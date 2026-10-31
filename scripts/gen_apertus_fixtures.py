# /// script
# requires-python = ">=3.10"
# dependencies = ["torch>=2.9", "transformers>=4.57", "safetensors"]
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""A tiny random Apertus checkpoint and its reference logits from transformers.

Usage (CPU-only torch; uv installs it in an isolated environment):
    uv run scripts/gen_apertus_fixtures.py                    # tiny random model, ~10 s
    uv run scripts/gen_apertus_fixtures.py real <model_dir>   # Apertus 1.5 8B, ~5 min, ~40 GB RAM

Apertus 8B does not fit the 16 GB GPU in BFloat16, so the forward pass is checked on a
small model with random weights instead. It runs on transformers' ApertusForCausalLM
(Apertus 1), whose decoder is the same as Apertus 1.5's (the swiss-ai fork's
Apertus1p5TextModel), and is saved the way Apertus 1.5 ships: decoder settings nested
under text_config, tensors under model.language_model., and an embedding with extra rows
after the text vocabulary (image and audio codes there) that lm_head does not predict.

Writes test/fixtures/apertus/tiny/{config.json,model.safetensors} (BFloat16 weights) and
test/fixtures/logits/apertus-tiny.json: for each prompt of random token ids and for both
float32 and bfloat16, the top-k logits at every position and a greedy continuation.

`real` runs the pulled Apertus 1.5 8B checkpoint the same way, in Float32 only: its decoder
tensors renamed to Apertus 1's and the embedding cut to the text rows, on the prompts of
gen_model_fixtures.py (the chat one through the checkpoint's own template), and writes
test/fixtures/logits/apertus-1.5-8b.json.
"""

import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import ApertusConfig, ApertusForCausalLM, AutoTokenizer
from transformers.models.apertus.modeling_apertus import ApertusRotaryEmbedding

FIXTURES = Path(__file__).resolve().parent.parent / "test" / "fixtures"
TOP_K = 10
NEW_TOKENS = 24
EXTRA_ROWS = 64          # input-only embedding rows, as Apertus 1.5's image/audio codes

TEXT_CONFIG = {
    "vocab_size": 512, "hidden_size": 128, "intermediate_size": 384, "num_hidden_layers": 2,
    "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 64, "rms_norm_eps": 1e-5,
    "max_position_embeddings": 262144, "tie_word_embeddings": False, "hidden_act": "xielu",
    "attention_bias": False, "mlp_bias": False, "bos_token_id": 1, "eos_token_id": [2, 68, 72],
    "pad_token_id": 3,
    # Apertus 1.5's RoPE.
    "rope_parameters": {"rope_type": "llama3", "rope_theta": 4000000, "factor": 32.0,
                        "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                        "original_max_position_embeddings": 8192},
}

# Raw (pre-softplus) xIELU parameters per layer: layer 0 like Apertus 1.5's first layer
# (αp ≈ 18, αn ≈ 11), layer 1 like its later ones.
XIELU_RAW = [(18.125, 10.5), (0.3, -0.4)]

PROMPT_LENGTHS = {"short": 5, "medium": 37, "long": 100}


def cast(model: ApertusForCausalLM, dtype: torch.dtype) -> ApertusForCausalLM:
    """`model.to(dtype)`, but with the RoPE buffers kept in Float32, as from_pretrained does."""
    rope = model.model.rotary_emb
    buffers = {name: b.clone() for name, b in rope.named_buffers()}
    model = model.to(dtype)
    for name, b in buffers.items():
        setattr(rope, name, b)
    return model


def build_model() -> ApertusForCausalLM:
    torch.manual_seed(0)
    model = ApertusForCausalLM(ApertusConfig(**TEXT_CONFIG)).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "act_fn" in name:
                continue
            if p.dim() == 2:     # linear weights (out, in) and the embedding
                std = 1.0 if "embed" in name else p.shape[1] ** -0.5
                p.copy_(torch.randn_like(p) * std)
            else:                # RMSNorm weights
                p.copy_(1 + 0.2 * torch.randn_like(p))
        for layer, (alpha_p, alpha_n) in zip(model.model.layers, XIELU_RAW):
            layer.mlp.act_fn.alpha_p.fill_(alpha_p)
            layer.mlp.act_fn.alpha_n.fill_(alpha_n)
    # Store what the checkpoint holds: BFloat16 values.
    return cast(cast(model, torch.bfloat16), torch.float32)


def save_checkpoint(model: ApertusForCausalLM, out: Path) -> None:
    tensors = {}
    for name, t in model.state_dict().items():
        name = name.replace("model.", "model.language_model.", 1) if name.startswith("model.") else name
        tensors[name] = t.to(torch.bfloat16).contiguous()
    embed = "model.language_model.embed_tokens.weight"
    extra = torch.randn(EXTRA_ROWS, TEXT_CONFIG["hidden_size"]).to(torch.bfloat16)
    tensors[embed] = torch.cat([tensors[embed], extra])
    text = dict(TEXT_CONFIG, model_type="apertus1p5_text", dtype="bfloat16", qk_norm=True, post_norm=False,
                vocab_size=TEXT_CONFIG["vocab_size"] + EXTRA_ROWS, output_vocab_size=TEXT_CONFIG["vocab_size"])
    config = {"model_type": "apertus1p5", "architectures": ["Apertus1p5ForConditionalGeneration"],
              "tie_word_embeddings": False, "text_config": text}
    out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, out / "model.safetensors")
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")


@torch.no_grad()
def reference(model, ids):
    input_ids = torch.tensor([ids])
    logits = model(input_ids).logits[0].float()             # (T, vocab)
    top = torch.topk(logits, TOP_K, dim=-1)
    greedy = model.generate(input_ids, max_new_tokens=NEW_TOKENS, do_sample=False, eos_token_id=None,
                            pad_token_id=0, temperature=None, top_p=None, top_k=None)[0, len(ids):]
    return {"topk_ids": top.indices.tolist(), "topk_logits": top.values.tolist(), "greedy": greedy.tolist()}


def main() -> None:
    model = build_model()
    save_checkpoint(model, FIXTURES / "apertus" / "tiny")
    g = torch.Generator().manual_seed(1)
    cases = {k: {"ids": torch.randint(0, TEXT_CONFIG["vocab_size"], (n,), generator=g).tolist()}
             for k, n in PROMPT_LENGTHS.items()}
    for dtype_name, dtype in (("float32", torch.float32), ("bfloat16", torch.bfloat16)):
        m = cast(model, dtype)
        m.config._attn_implementation = "eager"
        for case_name, case in cases.items():
            case[dtype_name] = reference(m, case["ids"])
            print(f"{dtype_name} {case_name}: {len(case['ids'])} tokens")
    out = FIXTURES / "logits" / "apertus-tiny.json"
    out.write_text(json.dumps({"top_k": TOP_K, "cases": cases}) + "\n")
    print(f"wrote {out}")


REAL_PROMPTS = {
    "chat": [{"role": "user", "content": "Explain in two sentences why the sky is blue."}],
    "prose": "The history of the printing press begins in the fifteenth century, when",
    "code": "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
}


def load_real(model_dir: Path) -> ApertusForCausalLM:
    text = json.loads((model_dir / "config.json").read_text())["text_config"]
    keys = ["hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
            "rms_norm_eps", "max_position_embeddings", "rope_parameters", "hidden_act", "tie_word_embeddings",
            "bos_token_id", "eos_token_id", "pad_token_id"]
    config = ApertusConfig(**{k: text[k] for k in keys}, vocab_size=text["output_vocab_size"])
    with torch.device("meta"):
        model = ApertusForCausalLM(config)
    state = {}
    for shard in sorted(model_dir.glob("model-apertus-model-*.safetensors")):
        with safe_open(shard, "pt") as f:
            for name in f.keys():
                t = f.get_tensor(name).float()
                if name == "model.language_model.embed_tokens.weight":
                    t = t[: config.vocab_size].clone()
                state[name.replace("model.language_model.", "model.", 1)] = t
    model.load_state_dict(state, strict=True, assign=True)
    model.model.rotary_emb = ApertusRotaryEmbedding(config)       # not in the checkpoint
    return model.eval()


def real(model_dir: str) -> None:
    model_dir = Path(model_dir)
    model = load_real(model_dir)
    model.config._attn_implementation = "eager"
    tok = AutoTokenizer.from_pretrained(model_dir)
    cases = {}
    for name, prompt in REAL_PROMPTS.items():
        if isinstance(prompt, list):
            text = tok.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            ids = tok(text, add_special_tokens=False)["input_ids"]
        else:
            ids = tok(prompt, add_special_tokens=True)["input_ids"]
        cases[name] = {"ids": ids, "float32": reference(model, ids)}
        print(f"float32 {name}: {len(ids)} tokens")
    out = FIXTURES / "logits" / "apertus-1.5-8b.json"
    out.write_text(json.dumps({"top_k": TOP_K, "cases": cases}) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "real":
        real(sys.argv[2])
    elif len(sys.argv) == 1:
        main()
    else:
        sys.exit(__doc__)
