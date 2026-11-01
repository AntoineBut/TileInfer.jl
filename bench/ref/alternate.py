"""Alternate the vLLM and TileInfer races so that drift in GPU clocks or desktop load hits
both engines alike, then summarize each engine over all its runs.

    python3 bench/ref/alternate.py [--model REPO] [--rounds 3] [--trials 10] [--prompt-len 512] [--gen 256]
        [--weights bf16|int8] [--kv bf16|int8]
        [--out bench/results/race-alternate.json]

Round r runs vLLM then TileInfer when r is odd, TileInfer then vLLM when r is even (ABBA),
each in a fresh process via vllm_race.py / tileinfer_race.jl, which record every trial and
report medians (timings are heavy-tailed). The summary pools every trial of an engine
across rounds and reports medians as the primary result, with the minimum and the
quartiles. Before every run it records the other GPU processes and the SM clock, so a
disturbed run can be spotted afterwards.
"""

import argparse
import json
import os
import statistics
import subprocess
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
JULIA = os.environ.get("JULIA", os.path.expanduser("~/.juliaup/bin/julia"))


def gpu_state():
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                           "--format=csv,noheader"], capture_output=True, text=True).stdout
    clock = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,utilization.gpu", "--format=csv,noheader"],
                           capture_output=True, text=True).stdout.strip()
    names = [line.split(",")[1].strip().split()[0].split("/")[-1] for line in apps.splitlines() if line.strip()]
    return {"other_processes": names, "sm_clock_util": clock}


def vllm_model(a):
    if a.weights == "bf16":
        return a.model
    path = os.path.join(HERE, "models", a.model.split("/")[-1] + "-w8a16")
    if not os.path.isdir(path):
        raise SystemExit(f"{path} missing: run bench/ref/make_w8a16.py {a.model}")
    return path


def run_vllm(a):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out = f.name
    subprocess.run([os.path.join(HERE, ".venv", "bin", "python"), os.path.join(HERE, "vllm_race.py"),
                    "--model", vllm_model(a), "--kv-cache-dtype", "int8_per_token_head" if a.kv == "int8" else "auto",
                    "--prompt-len", str(a.prompt_len), "--gen", str(a.gen),
                    "--trials", str(a.trials), "--out", out],
                   check=True, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with open(out) as f:
        res = json.load(f)
    os.remove(out)
    return res


def run_tileinfer(a):
    proc = subprocess.run([JULIA, "--project=bench", os.path.join(HERE, "tileinfer_race.jl"), a.model,
                           str(a.prompt_len), str(a.gen), "tile", str(a.trials), a.weights, a.kv],
                          check=True, cwd=ROOT, capture_output=True, text=True)
    return json.loads(proc.stdout[proc.stdout.index("{"):])


def summary(values):
    q1, median, q3 = statistics.quantiles(values, n=4)
    return {"n": len(values), "min": min(values), "q1": q1, "median": median, "q3": q3, "max": max(values)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--prompt-len", type=int, default=512)
    p.add_argument("--gen", type=int, default=256)
    p.add_argument("--weights", default="bf16", choices=["bf16", "int8"],
                   help="int8: TileInfer quantize = :int8, vLLM the W8A16 copy from make_w8a16.py")
    p.add_argument("--kv", default="bf16", choices=["bf16", "int8"],
                   help="int8: TileInfer Int8Pages, vLLM kv_cache_dtype = int8_per_token_head")
    p.add_argument("--out", default=os.path.join(ROOT, "bench", "results", "race-alternate.json"))
    a = p.parse_args()

    engines = {"vllm": run_vllm, "tileinfer": run_tileinfer}
    runs = []
    for r in range(1, a.rounds + 1):
        order = ["vllm", "tileinfer"] if r % 2 else ["tileinfer", "vllm"]
        for name in order:
            state = gpu_state()
            t = time.time()
            res = engines[name](a)
            runs.append({"round": r, "engine": name, "gpu_before": state, "wall_s": time.time() - t, **res})
            print(f"round {r} {name:9s}  median: decode {res['decode_tok_s']:6.1f} tok/s  "
                  f"prefill {res['prefill_ms']:6.2f} ms  e2e {res['e2e_s']:.4f} s  "
                  f"(min-based: prefill {res['prefill_ms_min']:6.2f} ms)  "
                  f"first tokens {res['first_tokens'][:4]}  [{state['sm_clock_util']}; "
                  f"{', '.join(state['other_processes'])}]", flush=True)

    # Pool every trial of an engine across its runs. Decode comes from the pooled medians:
    # (median latency(gen) - median latency(1)) / (gen - 1).
    result = {"config": vars(a), "runs": runs, "summary": {}}
    print(f"\nall trials pooled ({a.rounds} runs x {a.trials} trials per engine); median [q1, q3], min")
    for name in engines:
        mine = [x for x in runs if x["engine"] == name]
        pre = summary([1e3 * t for x in mine for t in x["trials_prefill_s"]])
        e2e = summary([t for x in mine for t in x["trials_e2e_s"]])
        decode_ms = 1e3 * (e2e["median"] - pre["median"] / 1e3) / (a.gen - 1)
        result["summary"][name] = {"prefill_ms": pre, "e2e_s": e2e, "decode_ms_per_tok": decode_ms,
                                   "decode_tok_s": 1e3 / decode_ms}
        print(f"{name:10s} prefill {pre['median']:6.2f} ms [{pre['q1']:6.2f}, {pre['q3']:6.2f}], min {pre['min']:6.2f}"
              f"   e2e {e2e['median']:.4f} s [{e2e['q1']:.4f}, {e2e['q3']:.4f}], min {e2e['min']:.4f}"
              f"   decode {decode_ms:.3f} ms/tok = {1e3 / decode_ms:5.1f} tok/s")
    with open(a.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
