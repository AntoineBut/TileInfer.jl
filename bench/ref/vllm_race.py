"""vLLM side of the end-to-end drag race: single sequence, greedy, fixed lengths.

    bench/ref/.venv/bin/python bench/ref/vllm_race.py [--model Qwen/Qwen3-0.6B]
        [--prompt-len 512] [--gen 256] [--trials 5] [--eager] [--out bench/results/vllm.json]

Prefill time  = latency of a request with max_tokens = 1 (prompt forward + 1 sample).
Decode tok/s  = (gen - 1) / (latency(gen) - latency(1)).
Each latency is the median over `--trials` runs after a 1 s warm-up: the GPU also drives the
desktop, so timings are heavy-tailed and a few congested trials should not move the
result. Every trial is recorded, with the minimum-based numbers alongside. Prefix caching
is off so every trial re-prefills.
"""

import argparse
import json
import os
import statistics
import time

os.environ.setdefault("VLLM_CONFIGURE_LOGGING", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # every model is in the local cache
# No system CUDA toolkit: torch.compile finds nvcc in the pip-installed nvidia/cu13 package.
import nvidia  # noqa: E402

_cu = os.path.join(nvidia.__path__[0], "cu13")
os.environ.setdefault("CUDA_HOME", _cu)
os.environ["PATH"] = os.path.join(_cu, "bin") + os.pathsep + os.environ["PATH"]
# FlashInfer's JIT sampler fails to build against that nvcc (CCCL header mismatch); greedy
# decoding does not need it.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from vllm import LLM, SamplingParams  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--prompt-len", type=int, default=512)
    p.add_argument("--gen", type=int, default=256)
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--eager", action="store_true", help="no CUDA graphs / torch.compile")
    p.add_argument("--gpu-mem", type=float, default=0.6)
    p.add_argument("--kv-cache-dtype", default="auto", help="e.g. int8_per_token_head")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    llm = LLM(model=a.model, dtype=a.dtype, max_model_len=a.prompt_len + a.gen + 16,
              max_num_seqs=1, enable_prefix_caching=False, enforce_eager=a.eager,
              gpu_memory_utilization=a.gpu_mem, seed=0,
              kv_cache_dtype=a.kv_cache_dtype)
    # Deterministic ordinary-token prompt ids, same recipe as bench/ref/tileinfer_race.jl.
    ids = [1000 + (7919 * i) % 140000 for i in range(a.prompt_len)]
    prompt = TokensPrompt(prompt_token_ids=ids)

    def latency(n):
        sp = SamplingParams(temperature=0.0, max_tokens=n, min_tokens=n, ignore_eos=True)
        # Warm-up: keep the GPU busy for a second so its clock has ramped up (from idle,
        # a 10 ms request runs at 900-1600 MHz and takes 2-3x longer).
        t0 = time.perf_counter()
        llm.generate([prompt], sp, use_tqdm=False)
        while time.perf_counter() - t0 < 1.0:
            llm.generate([prompt], sp, use_tqdm=False)
        times = []
        for _ in range(a.trials):
            t = time.perf_counter()
            out = llm.generate([prompt], sp, use_tqdm=False)
            times.append(time.perf_counter() - t)
        assert len(out[0].outputs[0].token_ids) == n
        return times, list(out[0].outputs[0].token_ids)

    times1, _ = latency(1)
    timesn, toks = latency(a.gen)

    def metrics(t1, tn, suffix=""):
        return {
            "prefill_ms" + suffix: 1e3 * t1, "prefill_tok_s" + suffix: a.prompt_len / t1,
            "decode_ms_per_tok" + suffix: 1e3 * (tn - t1) / (a.gen - 1),
            "decode_tok_s" + suffix: (a.gen - 1) / (tn - t1), "e2e_s" + suffix: tn,
        }

    res = {
        "engine": "vllm", "eager": a.eager, "kv_cache_dtype": a.kv_cache_dtype, "model": a.model, "dtype": a.dtype,
        "prompt_len": a.prompt_len, "gen": a.gen, "trials": a.trials, "statistic": "median",
        **metrics(statistics.median(times1), statistics.median(timesn)),
        **metrics(min(times1), min(timesn), "_min"),
        "trials_prefill_s": times1, "trials_e2e_s": timesn, "first_tokens": toks[:16],
    }
    print(json.dumps(res, indent=2))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
