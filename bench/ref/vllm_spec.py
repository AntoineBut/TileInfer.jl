"""vLLM side of the speculative-decoding comparison; same workload, metric and output as
tileinfer_spec.jl (greedy replies of up to `gen` tokens, stopping at EOS, to the prompts of
spec_prompts.json).

    bench/ref/.venv/bin/python bench/ref/vllm_spec.py MODEL METHOD K WEIGHTS KV [--gen 128] [--trials 5]
        [--prompts spec_prompts.json] [--gpu-mem 0.75] [--prefix-caching] [--profile]
        [--attention-backend FLASH_ATTN|FLASHINFER|TRITON_ATTN]

METHOD: plain, ngram (prompt lookup, n-grams of 3 down to 1) or dflash
(z-lab/Qwen3-4B-DFlash-b16); WEIGHTS int8 runs the W8A16 copy from make_w8a16.py; KV int8
is kv_cache_dtype = int8_per_token_head. Prefix caching is off unless --prefix-caching (then the
repeated requests hit the cache, so prefill_ms and decode_tok_s no longer mean the same).

--profile (under nsys, see profile.sh): after the same warm-up, one reply per prompt between
llm.start_profile()/stop_profile() (cudaProfilerStart/Stop in the engine-core process), each in
an NVTX range `request NAME`, with vLLM's NVTX scopes on (VLLM_NVTX_SCOPES_FOR_PROFILING), and
no timed trials.
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
# nvcc from the pip CUDA toolkit; ninja (FlashInfer's JIT builds) from the venv, which is not on
# PATH when its python is run directly.
os.environ["PATH"] = os.pathsep.join([os.path.join(_cu, "bin"), os.path.dirname(sys.executable), os.environ["PATH"]])
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
if "--profile" in sys.argv:
    os.environ.setdefault("VLLM_NVTX_SCOPES_FOR_PROFILING", "1")  # read in the engine-core process
    # py-cpuinfo (imported by vLLM) calls platform.architecture(), which runs `file` on the
    # interpreter; nsys injects itself into that child, which hangs in file's seccomp sandbox.
    import platform

    platform.architecture = lambda *args, **kwargs: ("64bit", "ELF")

import torch  # noqa: E402
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


def annotate_runner():
    """NVTX ranges around the target forward, sampling and drafting of vLLM's (V2) model runner,
    which has no NVTX scopes of its own; nsys_export.py splits each step into parts by them.
    The engine core is forked after this, so it inherits the wrappers."""
    import functools

    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator

    def wrap(cls, method, name):
        f = getattr(cls, method)

        @functools.wraps(f)
        def g(*args, **kwargs):
            torch.cuda.nvtx.range_push(name)
            try:
                return f(*args, **kwargs)
            finally:
                torch.cuda.nvtx.range_pop()

        setattr(cls, method, g)

    wrap(GPUModelRunner, "execute_model", "model_runner: forward")
    wrap(GPUModelRunner, "sample", "model_runner: sample")
    wrap(DFlashSpeculator, "propose", "model_runner: draft")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("method", choices=["plain", "ngram", "dflash"])
    p.add_argument("k", type=int)
    p.add_argument("weights", choices=["bf16", "int8"])
    p.add_argument("kv", choices=["bf16", "int8"])
    p.add_argument("--gen", type=int, default=128)
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--gpu-mem", type=float, default=0.75)
    p.add_argument("--prompts", default="spec_prompts.json", help="prompt file, relative to bench/ref/")
    p.add_argument("--prefix-caching", action="store_true")
    p.add_argument("--profile", action="store_true")
    p.add_argument("--attention-backend", default=None, help="vLLM attention backend (default: its choice, FLASH_ATTN here)")
    a = p.parse_args()

    with open(os.path.join(HERE, a.prompts)) as f:
        spec = json.load(f)
    max_tokens = 256 * -(-(max(len(x["ids"]) for x in spec["prompts"]) + a.gen + 64) // 256)
    model = a.model if a.weights == "bf16" else os.path.join(HERE, "models", a.model.split("/")[-1] + "-w8a16")
    speculative = None
    if a.method == "ngram":
        speculative = {"method": "ngram", "num_speculative_tokens": a.k,
                       "prompt_lookup_max": 3, "prompt_lookup_min": 1}
    elif a.method == "dflash":
        speculative = {"method": "dflash", "model": "z-lab/Qwen3-4B-DFlash-b16", "num_speculative_tokens": a.k}
    if a.profile:
        annotate_runner()
    llm = LLM(model=model, dtype="bfloat16", max_model_len=max_tokens, max_num_seqs=1,
              enable_prefix_caching=a.prefix_caching, gpu_memory_utilization=a.gpu_mem, seed=0,
              kv_cache_dtype="int8_per_token_head" if a.kv == "int8" else "auto",
              speculative_config=speculative, disable_log_stats=False,
              profiler_config={"profiler": "cuda"} if a.profile else None,
              attention_config={"backend": a.attention_backend} if a.attention_backend else None)

    def run(ids, n):
        sp = SamplingParams(temperature=0.0, max_tokens=n, stop_token_ids=spec["stop_ids"])
        t = time.perf_counter()
        out = llm.generate([TokensPrompt(prompt_token_ids=ids)], sp, use_tqdm=False)
        return time.perf_counter() - t, list(out[0].outputs[0].token_ids)

    def warm_up(ids):
        t0 = time.perf_counter()
        run(ids, a.gen)
        run(ids, 1)
        while time.perf_counter() - t0 < 1.0:
            run(ids, 1)

    results = []
    if a.profile:
        # One traced reply per prompt; the profile's own output comes from nsys_export.py.
        for pr in spec["prompts"]:
            warm_up(pr["ids"])
        llm.start_profile()
        for pr in spec["prompts"]:
            before = spec_counters(llm)
            torch.cuda.nvtx.range_push(f"request {pr['name']}")
            dt, out = run(pr["ids"], a.gen)
            torch.cuda.nvtx.range_pop()
            after = spec_counters(llm)
            out = [t for t in out if t not in spec["stop_ids"]]
            results.append({"prompt": pr["name"], "prompt_len": len(pr["ids"]), "out": len(out), "e2e_s": dt,
                            "steps": after[0] - before[0], "first_tokens": out[:16]})
        llm.stop_profile()
    for pr in [] if a.profile else spec["prompts"]:
        ids = pr["ids"]
        warm_up(ids)
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
    print(json.dumps({"engine": "vllm", "profile": a.profile, "attention_backend": a.attention_backend, "prefix_caching": a.prefix_caching, "prompts": a.prompts, "model": a.model, "method": a.method, "k": a.k, "weights": a.weights,
                      "kv": a.kv, "gen": a.gen, "trials": a.trials, "results": results}, indent=2))


if __name__ == "__main__":
    main()
