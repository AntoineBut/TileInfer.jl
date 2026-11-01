"""vLLM side of the speculative-decoding comparison; same workload, metric and output as
tileinfer_spec.jl (greedy replies of up to `gen` tokens, stopping at EOS, to the prompts of
spec_prompts.json).

    bench/ref/.venv/bin/python bench/ref/vllm_spec.py MODEL METHOD K WEIGHTS KV [--gen 256] [--trials 5]

METHOD: plain, ngram (prompt lookup, n-grams of 3 down to 1) or dflash
(z-lab/Qwen3-4B-DFlash-b16); WEIGHTS int8 runs the W8A16 copy from make_w8a16.py; KV int8
is kv_cache_dtype = int8_per_token_head. Prefix caching is off.
"""

import argparse
import json
import os
import statistics
import sys
import time

os.environ.setdefault("VLLM_CONFIGURE_LOGGING", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # every model is in the local cache
import nvidia  # noqa: E402

_cu = os.path.join(nvidia.__path__[0], "cu13")
os.environ.setdefault("CUDA_HOME", _cu)
os.environ["PATH"] = os.path.join(_cu, "bin") + os.pathsep + os.environ["PATH"]
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from vllm import LLM, SamplingParams  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def spec_counters(llm):
    """Cumulative (verify steps with drafts, accepted drafts) from vLLM's metrics."""
    drafts = accepted = 0
    for m in llm.get_metrics():
        if m.name == "vllm:spec_decode_num_drafts":
            drafts += m.value
        elif m.name == "vllm:spec_decode_num_accepted_tokens":
            accepted += m.value
    return drafts, accepted


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("method", choices=["plain", "ngram", "dflash"])
    p.add_argument("k", type=int)
    p.add_argument("weights", choices=["bf16", "int8"])
    p.add_argument("kv", choices=["bf16", "int8"])
    p.add_argument("--gen", type=int, default=256)
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--gpu-mem", type=float, default=0.75)
    a = p.parse_args()

    with open(os.path.join(HERE, "spec_prompts.json")) as f:
        spec = json.load(f)
    max_tokens = 256 * -(-(max(len(x["ids"]) for x in spec["prompts"]) + a.gen + 64) // 256)
    model = a.model if a.weights == "bf16" else os.path.join(HERE, "models", a.model.split("/")[-1] + "-w8a16")
    speculative = None
    if a.method == "ngram":
        speculative = {"method": "ngram", "num_speculative_tokens": a.k,
                       "prompt_lookup_max": 3, "prompt_lookup_min": 1}
    elif a.method == "dflash":
        speculative = {"method": "dflash", "model": "z-lab/Qwen3-4B-DFlash-b16", "num_speculative_tokens": a.k}
    llm = LLM(model=model, dtype="bfloat16", max_model_len=max_tokens, max_num_seqs=1,
              enable_prefix_caching=False, gpu_memory_utilization=a.gpu_mem, seed=0,
              kv_cache_dtype="int8_per_token_head" if a.kv == "int8" else "auto",
              speculative_config=speculative, disable_log_stats=False)

    def run(ids, n):
        sp = SamplingParams(temperature=0.0, max_tokens=n, stop_token_ids=spec["stop_ids"])
        t = time.perf_counter()
        out = llm.generate([TokensPrompt(prompt_token_ids=ids)], sp, use_tqdm=False)
        return time.perf_counter() - t, list(out[0].outputs[0].token_ids)

    results = []
    for pr in spec["prompts"]:
        ids = pr["ids"]
        t0 = time.perf_counter()
        run(ids, a.gen)
        run(ids, 1)
        while time.perf_counter() - t0 < 1.0:
            run(ids, 1)
        tn, t1 = [], []
        for _ in range(a.trials):
            before = spec_counters(llm)
            dt, out = run(ids, a.gen)
            after = spec_counters(llm)
            tn.append(dt)
            t1.append(run(ids, 1)[0])
        # vLLM's output excludes the stop token, like TileInfer's.
        out = [t for t in out if t not in spec["stop_ids"]]
        n = len(out)
        e2e, pre = statistics.median(tn), statistics.median(t1)
        r = {"prompt": pr["name"], "prompt_len": len(ids), "out": n, "e2e_s": e2e, "prefill_ms": 1e3 * pre,
             "decode_tok_s": (n - 1) / (e2e - pre), "trials_e2e_s": tn, "trials_prefill_s": t1,
             "first_tokens": out[:16]}
        if speculative is not None:
            steps = after[0] - before[0]
            r["steps"] = steps
            r["tokens_per_step"] = n / steps if steps else None
        results.append(r)
        print(f"{pr['name']:13s} out {n}  decode {r['decode_tok_s']:.1f} tok/s", file=sys.stderr, flush=True)
    print(json.dumps({"engine": "vllm", "model": a.model, "method": a.method, "k": a.k, "weights": a.weights,
                      "kv": a.kv, "gen": a.gen, "trials": a.trials, "results": results}, indent=2))


if __name__ == "__main__":
    main()
