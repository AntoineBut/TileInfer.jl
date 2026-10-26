# /// script
# requires-python = ">=3.10"
# dependencies = ["torch>=2.9", "transformers>=4.57"]
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""Reference logits and greedy continuations from Hugging Face transformers.

Usage (CPU-only torch; uv installs it in an isolated environment):
    uv run scripts/gen_model_fixtures.py <model_dir> <name>

writes test/fixtures/logits/<name>.json. For every prompt and for both float32 and
bfloat16, it stores the top-k logits at every position and a greedy continuation. The
model runs with eager attention, the plain matmul/softmax formulation that TileInfer's
reference backend follows.
"""

import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

FIXTURES = Path(__file__).resolve().parent.parent / "test" / "fixtures"
TOP_K = 10
NEW_TOKENS = 24

PROMPTS = {
    "chat": [{"role": "user", "content": "Explain in two sentences why the sky is blue."}],
    "prose": "The history of the printing press begins in the fifteenth century, when",
    "code": "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
}


def encode(tok, prompt):
    if isinstance(prompt, list):
        text = tok.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    else:
        text = prompt
    return tok(text, add_special_tokens=True)["input_ids"]


@torch.no_grad()
def reference(model, ids):
    input_ids = torch.tensor([ids])
    logits = model(input_ids).logits[0].float()             # (T, vocab)
    top = torch.topk(logits, TOP_K, dim=-1)
    greedy = model.generate(input_ids, max_new_tokens=NEW_TOKENS, do_sample=False,
                            temperature=None, top_p=None, top_k=None)[0, len(ids):]
    return {
        "topk_ids": top.indices.tolist(),
        "topk_logits": top.values.tolist(),
        "greedy": greedy.tolist(),
    }


def main(model_dir: str, name: str) -> None:
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(model_dir)
    cases = {k: {"ids": encode(tok, p)} for k, p in PROMPTS.items()}
    for dtype_name, dtype in (("float32", torch.float32), ("bfloat16", torch.bfloat16)):
        model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype,
                                                     attn_implementation="eager").eval()
        for case_name, case in cases.items():
            case[dtype_name] = reference(model, case["ids"])
            print(f"{dtype_name} {case_name}: {len(case['ids'])} tokens")
        del model
    out = FIXTURES / "logits" / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"top_k": TOP_K, "cases": cases}) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
