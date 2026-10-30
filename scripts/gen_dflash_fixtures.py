# /// script
# requires-python = ">=3.10"
# dependencies = ["torch>=2.9", "transformers==4.57.3", "safetensors"]
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""Reference DFlash drafts, from a transcription of z-lab's implementation.

The drafter below transcribes the forward pass of the checkpoint's dflash.py (MIT license,
z-lab/Qwen3-4B-DFlash-b16) onto transformers' Qwen3 modules, rather than loading it with
trust_remote_code, so that no code from the checkpoint runs here.

Usage (CPU-only torch, Float32; uv installs it in an isolated environment):
    uv run scripts/gen_dflash_fixtures.py <target_dir> <drafter_dir> <name>
    uv run scripts/gen_dflash_fixtures.py <target_dir> <drafter_dir> --report "<prompt>" [steps]

The first form follows the reference `spec_generate` loop (greedy) for a few blocks and
writes test/fixtures/dflash/<name>.json and <name>.bin. For every block it records the
context the drafter reads (the target's states of the positions it just ran, rounded to
BFloat16 so that a test can feed exactly the same values; the reference drafter runs on the
rounded values too), the block's position, and the top-k logits of its drafts. The .bin file
holds the states of all blocks as BFloat16 (hidden x layers, row-major per position).
transformers is pinned to the version z-lab evaluated with.

`--report` runs the reference greedy speculative loop on a chat prompt instead and prints
the tokens per verify step, to compare with TileInfer's.
"""

import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, DynamicCache
from transformers.models.qwen3.modeling_qwen3 import (Qwen3MLP, Qwen3RMSNorm, Qwen3RotaryEmbedding,
                                                      eager_attention_forward, rotate_half)

FIXTURES = Path(__file__).resolve().parent.parent / "test" / "fixtures" / "dflash"
TOP_K = 10
BLOCKS = 3
PROMPT = [{"role": "user", "content": "List the first ten prime numbers."}]


# --- dflash.py, transcribed (eager attention, no dropout) ---

def apply_rotary_pos_emb(q, k, cos, sin):
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    q_len = q.size(-2)
    q_embed = (q * cos[..., -q_len:, :]) + (rotate_half(q) * sin[..., -q_len:, :])
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Attention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.training = False
        H, D = config.hidden_size, config.head_dim
        self.q_proj = nn.Linear(H, config.num_attention_heads * D, bias=False)
        self.k_proj = nn.Linear(H, config.num_key_value_heads * D, bias=False)
        self.v_proj = nn.Linear(H, config.num_key_value_heads * D, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * D, H, bias=False)
        self.q_norm = Qwen3RMSNorm(D, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(D, eps=config.rms_norm_eps)

    def forward(self, hidden_states, target_hidden, position_embeddings, past_key_values):
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden.shape[1]
        q = self.q_norm(self.q_proj(hidden_states).view(bsz, q_len, -1, self.head_dim)).transpose(1, 2)
        k = torch.cat([self.k_proj(target_hidden), self.k_proj(hidden_states)], dim=1)
        v = torch.cat([self.v_proj(target_hidden), self.v_proj(hidden_states)], dim=1)
        k = self.k_norm(k.view(bsz, ctx_len + q_len, -1, self.head_dim)).transpose(1, 2)
        v = v.view(bsz, ctx_len + q_len, -1, self.head_dim).transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        k, v = past_key_values.update(k, v, self.layer_idx)
        # No mask: the block sees the whole context and the whole block.
        out, _ = eager_attention_forward(self, q, k, v, None, dropout=0.0, scaling=self.scaling)
        return self.o_proj(out.reshape(bsz, q_len, -1))


class Layer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.self_attn = Attention(config, layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, target_hidden, position_embeddings, past_key_values):
        h = hidden_states + self.self_attn(self.input_layernorm(hidden_states), target_hidden,
                                           position_embeddings, past_key_values)
        return h + self.mlp(self.post_attention_layernorm(h))


class Drafter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([Layer(config, i) for i in range(config.num_hidden_layers)])
        self.target_layer_ids = config.dflash_config["target_layer_ids"]
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config)
        self.fc = nn.Linear(len(self.target_layer_ids) * config.hidden_size, config.hidden_size, bias=False)
        self.hidden_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.block_size = config.block_size
        self.mask_token_id = config.dflash_config["mask_token_id"]

    def forward(self, position_ids, noise_embedding, target_hidden, past_key_values):
        hidden_states = noise_embedding
        target_hidden = self.hidden_norm(self.fc(target_hidden))
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, target_hidden, position_embeddings, past_key_values)
        return self.norm(hidden_states)

# --- end of the transcription ---


def load(target_dir, drafter_dir):
    target = AutoModelForCausalLM.from_pretrained(target_dir, dtype=torch.float32,
                                                  attn_implementation="eager").eval()
    drafter = Drafter(AutoConfig.from_pretrained(drafter_dir)).float().eval()
    weights = load_file(str(Path(drafter_dir) / "model.safetensors"))
    drafter.load_state_dict({k: v.float() for k, v in weights.items()})
    return target, drafter, AutoTokenizer.from_pretrained(target_dir)


def encode(tok, messages):
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)
    return tok(text, add_special_tokens=True)["input_ids"]


def context_states(drafter, hidden_states):
    # As dflash.py's extract_context_feature, rounded to BFloat16.
    x = torch.cat([hidden_states[i + 1] for i in drafter.target_layer_ids], dim=-1)
    return x.to(torch.bfloat16).float()


@torch.no_grad()
def speculate(target, drafter, ids, blocks, on_block=None):
    """The greedy loop of dflash.py's spec_generate for `blocks` verify steps; returns the tokens."""
    B, mask = drafter.block_size, drafter.mask_token_id
    tcache, dcache = DynamicCache(), DynamicCache()
    out = target(torch.tensor([ids]), past_key_values=tcache, use_cache=True, logits_to_keep=1,
                 output_hidden_states=True)
    tokens = ids + [int(out.logits[0, -1].argmax())]
    states = context_states(drafter, out.hidden_states)            # (1, positions, layers x hidden)
    ctx_start, start = 0, len(ids)
    for _ in range(blocks):
        block = torch.tensor([[tokens[start]] + [mask] * (B - 1)])
        hidden = drafter(target_hidden=states, noise_embedding=target.model.embed_tokens(block),
                         position_ids=torch.arange(dcache.get_seq_length(), start + B).unsqueeze(0),
                         past_key_values=dcache)
        logits = target.lm_head(hidden[:, -B + 1:, :])[0]            # (B - 1, vocab)
        dcache.crop(start)
        drafts = logits.argmax(dim=-1).tolist()
        if on_block is not None:
            on_block(tokens[:start + 1], ctx_start, states, logits)

        out = target(torch.tensor([[tokens[start]] + drafts]), position_ids=torch.arange(start, start + B).unsqueeze(0),
                     past_key_values=tcache, use_cache=True, output_hidden_states=True)
        post = out.logits[0].argmax(dim=-1).tolist()
        accepted = 0
        while accepted < B - 1 and drafts[accepted] == post[accepted]:
            accepted += 1
        tokens = tokens[:start + 1] + drafts[:accepted] + [post[accepted]]
        ctx_start, start = start, start + accepted + 1
        tcache.crop(start)
        states = context_states(drafter, out.hidden_states)[:, :accepted + 1]
        print(f"block at {start - accepted - 1}: {accepted} drafts accepted", flush=True)
    return tokens


def fixture(target, drafter, tok):
    blocks, chunks = [], []

    def record(context, ctx_start, states, logits):
        top = torch.topk(logits, TOP_K, dim=-1)
        blocks.append({"context": context, "ctx_start": ctx_start, "ctx_len": states.shape[1],
                       "drafts": logits.argmax(dim=-1).tolist(), "topk_ids": top.indices.tolist(),
                       "topk_logits": top.values.tolist()})
        chunks.append(states[0].to(torch.bfloat16))

    speculate(target, drafter, encode(tok, PROMPT), BLOCKS, record)
    return {"block_size": drafter.block_size, "mask_id": drafter.mask_token_id,
            "layers": [i + 1 for i in drafter.target_layer_ids], "blocks": blocks}, torch.cat(chunks)


def report(target, drafter, tok, prompt, steps):
    ids = encode(tok, [{"role": "user", "content": prompt}])
    tokens = speculate(target, drafter, ids, steps)
    new = len(tokens) - len(ids) - 1                                 # the last bonus is not verified
    print(f"{new} tokens in {steps} verify steps: {new / steps:.2f} tokens per step")
    print(tok.decode(tokens[len(ids):]))


def main():
    target_dir, drafter_dir = sys.argv[1], sys.argv[2]
    target, drafter, tok = load(target_dir, drafter_dir)
    if sys.argv[3] == "--report":
        report(target, drafter, tok, sys.argv[4], int(sys.argv[5]) if len(sys.argv) > 5 else 40)
        return
    name = sys.argv[3]
    meta, states = fixture(target, drafter, tok)
    FIXTURES.mkdir(parents=True, exist_ok=True)
    (FIXTURES / f"{name}.json").write_text(json.dumps(meta))
    (FIXTURES / f"{name}.bin").write_bytes(states.view(torch.int16).numpy().tobytes())
    print(f"wrote {FIXTURES / name}.json, .bin ({states.shape[0]} positions)")


if __name__ == "__main__":
    main()
