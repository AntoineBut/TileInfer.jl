"""Engine-agnostic export of a profile.sh trace (Nsight Systems) to CSV/JSON, for plots, and a
side-by-side text summary of one or more traces.

    python3 bench/ref/nsys_export.py REPORT.nsys-rep [REPORT2.nsys-rep ...] [--force] [--top 15]

Each report is exported (once; --force redoes it) to the directory of the same name:

- kernels.csv: one row per GPU activity (kernel, memcpy, memset) of the traced requests:
  engine, request, step, step_kind, part, category, name, start_us, dur_us, stream, graph,
  grid, block, regs, bytes, nvtx (the launching thread's NVTX ranges, outermost first).
- steps.csv: one row per engine step: engine, request, step, step_kind, marker (the NVTX range
  that starts it), position (approximate context length), start_us, period_us (to the next
  step's first activity), span_us (first start to last end), busy_us (union of activity
  intervals), host_us (the marker range's host time), kernels, busy_us by part (part_*), and
  by category (cat_*).
- summary.json: the run (the driver's JSON output: tokens and steps per request), per step
  kind the mean step and its parts and categories, and per kernel name µs and launches per
  decode step.

Times are in µs from the first traced activity. A step is one engine iteration: a target
pass (prefill, or verify of the drafts) and what follows it up to the next target pass (the
drafter's pass, sampling, bookkeeping). Each activity goes to the step whose marker started
last before its launch (host time of the API call), so a vLLM draft (launched from
sample_tokens, outside the execute range) joins the step of its target pass, as TileInfer's
draft does. `part` comes from the innermost NVTX range at launch, `category` from the
kernel's name (CATEGORIES below; kernels.csv keeps raw names to recategorize).
`position` is interpolated from the request's tokens and steps (prompt + i * out / steps):
exact on average, off by the accepted-length spread within a request.
"""

import argparse
import bisect
import csv
import glob
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
from collections import defaultdict

# Per engine: NVTX range names that start a step (outermost match), and which start a prefill step.
ENGINES = {
    "tileinfer": {"step": r"^(prefill|(replay|eager|capture) (verify|step))$", "prefill": r"^prefill$"},
    "vllm": {"step": r"^execute_", "prefill": r"^execute_context_[1-9]"},
}
# Part of a step from the launching thread's NVTX ranges (joined with " / ", outermost first);
# first match wins.
PARTS = [
    ("draft", r"draft"),
    ("sample", r"sample"),
    ("target", r"prefill|verify|forward|(replay|eager|capture) step"),
]
# Category of an activity from its kernel name (lowercase); first match wins.
CATEGORIES = [
    ("memcpy", r"^memcpy|^memset"),
    ("attention_inputs", r"q_inputs|kv_inputs|rotary|rope|reshape_and_cache|concat_and_cache|store_kv"),
    ("attention", r"attention|attn|flash|fmha|paged|mha"),
    ("linear", r"gemm|gemv|cutlass|xmma|marlin|matmul|cublas|^sm\d+_|ampere_|splitk|wgmma|humming"),
    ("norm", r"norm|rsqrt"),
    ("activation", r"silu|swiglu|act_and_mul|gelu"),
    ("sampling", r"sample|topk|top_k|argmax|softmax|rejection|penalt|counter"),
    ("embedding", r"embed"),
    ("elementwise", r"triton_|elementwise|broadcast|vectorized|unrolled|copy|fill|index|gather|scatter|cat"),
]

KERNEL_FIELDS = ["engine", "request", "step", "step_kind", "part", "category", "name", "start_us", "dur_us",
                 "stream", "graph", "grid", "block", "regs", "bytes", "nvtx"]


def find_nsys():
    path = os.environ.get("NSYS") or shutil.which("nsys")
    if path:
        return path
    found = sorted(glob.glob("/usr/local/NVIDIA-Nsight-Compute*/host/target-linux-x64/nsys"))
    if not found:
        sys.exit("nsys not found; set NSYS")
    return found[-1]


def to_sqlite(rep, force):
    db = rep[: -len(".nsys-rep")] + ".sqlite"
    if force or not os.path.exists(db) or os.path.getmtime(db) < os.path.getmtime(rep):
        p = subprocess.run([find_nsys(), "export", "--type", "sqlite", "--force-overwrite", "true", "-o", db, rep],
                           capture_output=True, text=True)
        if p.returncode != 0:
            sys.exit(f"nsys export failed:\n{p.stdout[-2000:]}{p.stderr[-2000:]}")
    return db


def read_run(path):
    """The driver's JSON output, which profile.sh captures along with nsys's own stdout lines."""
    lines = open(path).read().splitlines()
    try:
        i = lines.index("{")
        j = len(lines) - 1 - lines[::-1].index("}")
        return json.loads("\n".join(lines[i:j + 1]))
    except ValueError:
        print(f"warning: no run JSON in {path}", file=sys.stderr)
        return {}


def first_match(rules, text):
    for label, rx in rules:
        if re.search(rx, text):
            return label
    return "other"


def nvtx_ranges(con):
    """(start, end, globalTid, text) of every NVTX range, push/pop and start/end."""
    rows = con.execute("""
        SELECT n.start, n.end, n.globalTid, coalesce(n.text, s.value)
        FROM NVTX_EVENTS n LEFT JOIN StringIds s ON s.id = n.textId
        WHERE n.eventType IN (59, 60) AND n.end IS NOT NULL""").fetchall()
    return [r for r in rows if r[3] is not None]


def activities(con):
    """Kernels, memcpys and memsets with their launch (API start, globalTid)."""
    strings = dict(con.execute("SELECT id, value FROM StringIds"))
    launch = {}
    for corr, tid, start, name in con.execute(
            "SELECT correlationId, globalTid, start, nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME"):
        launch[((tid >> 24) << 24, corr)] = (start, tid, strings.get(name, ""))
    acts = []
    for (start, end, pid, stream, corr, graph, short, demangled, gx, gy, gz, bx, by, bz, regs) in con.execute("""
            SELECT start, end, globalPid, streamId, correlationId, graphId, shortName, demangledName,
                   gridX, gridY, gridZ, blockX, blockY, blockZ, registersPerThread
            FROM CUPTI_ACTIVITY_KIND_KERNEL"""):
        d = strings.get(demangled, "")
        acts.append(dict(start=start, end=end, launch=launch.get((pid, corr)),
                         name=kernel_name(strings.get(short, "?"), d), demangled=d,
                         stream=stream, graph=graph is not None, grid=f"{gx}x{gy}x{gz}", block=f"{bx}x{by}x{bz}",
                         regs=regs, bytes=""))
    kinds = dict(con.execute("SELECT id, label FROM ENUM_CUDA_MEMCPY_OPER"))
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
        for start, end, pid, stream, corr, nbytes, kind, node in con.execute("""
                SELECT start, end, globalPid, streamId, correlationId, bytes, copyKind, graphNodeId
                FROM CUPTI_ACTIVITY_KIND_MEMCPY"""):
            acts.append(dict(start=start, end=end, launch=launch.get((pid, corr)), name=f"memcpy {kinds.get(kind, kind)}",
                             demangled="", stream=stream, graph=node is not None, grid="", block="", regs="",
                             bytes=nbytes))
    if "CUPTI_ACTIVITY_KIND_MEMSET" in tables:
        for start, end, pid, stream, corr, nbytes, node in con.execute("""
                SELECT start, end, globalPid, streamId, correlationId, bytes, graphNodeId
                FROM CUPTI_ACTIVITY_KIND_MEMSET"""):
            acts.append(dict(start=start, end=end, launch=launch.get((pid, corr)), name="memset", demangled="",
                             stream=stream, graph=node is not None, grid="", block="", regs="", bytes=nbytes))
    return acts


def kernel_name(short, demangled):
    """A readable kernel name: nsys's short name, unless it is a generic template name (cuBLAS's
    CUTLASS `Kernel2`, `kernel`) or truncated (torch.compile's Triton kernels), then from the
    demangled name."""
    if not demangled or not (short.endswith("...") or short.lower() in ("kernel", "kernel2")):
        return short
    m = re.search(r"cutlass::Kernel2?<(\w+)", demangled)
    if m:
        return m.group(1)
    m = re.search(r"(\w+(?:::\w+)*)::kernel<", demangled)
    if m:
        return m.group(1).split("::")[-1] + "_kernel"
    return re.sub(r"^void ", "", demangled.split("(")[0].split("<")[0])


def thread_stacks(ranges, acts):
    """For each activity, the NVTX ranges open on its launching thread at launch, outermost first."""
    by_tid = defaultdict(list)
    for r in ranges:
        by_tid[r[2]].append(r)
    launches = defaultdict(list)
    for a in acts:
        a["stack"] = []
        if a["launch"] is not None:
            launches[a["launch"][1]].append(a)
    for tid, la in launches.items():
        rs = sorted(by_tid.get(tid, []), key=lambda r: (r[0], -r[1]))   # outer before inner
        la.sort(key=lambda a: a["launch"][0])
        open_, i = [], 0
        for a in la:
            t = a["launch"][0]
            while i < len(rs) and rs[i][0] <= t:
                open_.append(rs[i])
                i += 1
            open_ = [r for r in open_ if r[1] >= t]
            a["stack"] = [r[3] for r in open_]


def union_length(intervals):
    total, cur_s, cur_e = 0, None, None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    return total + (cur_e - cur_s if cur_e is not None else 0)


def export(rep, force):
    base = rep[: -len(".nsys-rep")]
    out_dir = base
    run_json = base + ".json"
    run = read_run(run_json) if os.path.exists(run_json) else {}
    engine = run.get("engine") or ("vllm" if "vllm" in os.path.basename(base) else "tileinfer")
    rules = ENGINES[engine]
    con = sqlite3.connect(to_sqlite(rep, force))
    ranges = nvtx_ranges(con)
    acts = activities(con)
    con.close()
    thread_stacks(ranges, acts)

    # Requests: `request NAME` ranges (any thread; vLLM's are in the driver process).
    requests = sorted((r[0], r[1], r[3][len("request "):]) for r in ranges if r[3].startswith("request "))
    # Step markers: outermost matching ranges, in time order.
    markers = sorted((r[0], r[1], r[3]) for r in ranges if re.search(rules["step"], r[3]))
    outer, last_end = [], -1
    for m in markers:
        if m[0] >= last_end:
            outer.append(m)
            last_end = m[1]
    marker_starts = [m[0] for m in outer]

    def request_of(t):
        for s, e, name in requests:
            if s <= t <= e:
                return name
        return ""

    t0 = min(a["start"] for a in acts)
    rows = []
    for a in acts:
        t = a["launch"][0] if a["launch"] else a["start"]
        req = request_of(t)
        if requests and not req:
            continue                                   # outside the traced requests
        i = bisect.bisect_right(marker_starts, t) - 1
        nvtx = " / ".join(s for s in a["stack"] if not s.startswith("request "))
        rows.append(dict(engine=engine, request=req, marker=i, name=a["name"],
                         part=first_match(PARTS, nvtx),
                         category=first_match(CATEGORIES, (a["name"] + " " + a["demangled"][:200]).lower()),
                         start=a["start"], end=a["end"], stream=a["stream"], graph=int(a["graph"]), grid=a["grid"],
                         block=a["block"], regs=a["regs"], bytes=a["bytes"], nvtx=nvtx))
    rows.sort(key=lambda r: r["start"])

    # Steps, numbered from 0 within each request.
    groups = defaultdict(list)
    for r in rows:
        groups[(r["request"], r["marker"])].append(r)
    results = {x["prompt"]: x for x in run.get("results", [])}
    steps, index = [], {}
    by_request = defaultdict(list)
    for key in sorted(groups, key=lambda k: groups[k][0]["start"]):
        by_request[key[0]].append(key)
    for req, keys in by_request.items():
        res = results.get(req, {})
        n_decode = sum(1 for k in keys if k[1] < 0 or not re.search(rules["prefill"], outer[k[1]][2]))
        rate = res.get("out", 0) / res["steps"] if res.get("steps") else (res.get("out", 0) / max(n_decode, 1))
        d = 0
        for j, key in enumerate(keys):
            m = outer[key[1]] if key[1] >= 0 else (None, None, "")
            kind = "prefill" if re.search(rules["prefill"], m[2]) else ("decode" if key[1] >= 0 else "none")
            g = groups[key]
            step = dict(engine=engine, request=req, step=j, step_kind=kind, marker=m[2],
                        position=round(res.get("prompt_len", 0) + d * rate) if kind == "decode" else res.get("prompt_len", ""),
                        start_us=(g[0]["start"] - t0) / 1e3,
                        period_us=(groups[keys[j + 1]][0]["start"] - g[0]["start"]) / 1e3 if j + 1 < len(keys) else "",
                        span_us=(max(r["end"] for r in g) - g[0]["start"]) / 1e3,
                        busy_us=union_length([(r["start"], r["end"]) for r in g]) / 1e3,
                        host_us=(m[1] - m[0]) / 1e3 if m[0] is not None else "",
                        kernels=len(g))
            d += kind == "decode"
            for label in [p for p, _ in PARTS] + ["other"]:
                step["part_" + label] = sum(r["end"] - r["start"] for r in g if r["part"] == label) / 1e3
            for label in [c for c, _ in CATEGORIES] + ["other"]:
                step["cat_" + label] = sum(r["end"] - r["start"] for r in g if r["category"] == label) / 1e3
            steps.append(step)
            index[key] = (j, kind)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "kernels.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, KERNEL_FIELDS)
        w.writeheader()
        for r in rows:
            j, kind = index[(r["request"], r["marker"])]
            w.writerow(dict(engine=engine, request=r["request"], step=j, step_kind=kind, part=r["part"],
                            category=r["category"], name=r["name"], start_us=f"{(r['start'] - t0) / 1e3:.3f}",
                            dur_us=f"{(r['end'] - r['start']) / 1e3:.3f}", stream=r["stream"], graph=r["graph"],
                            grid=r["grid"], block=r["block"], regs=r["regs"], bytes=r["bytes"], nvtx=r["nvtx"]))
    with open(os.path.join(out_dir, "steps.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, list(steps[0].keys()))
        w.writeheader()
        for s in steps:
            w.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in s.items()})

    summary = {"engine": engine, "report": os.path.basename(rep), "run": run, "by_step_kind": {}, "kernels_per_decode_step": {}}
    for kind in ("prefill", "decode"):
        ss = [s for s in steps if s["step_kind"] == kind]
        if not ss:
            continue
        mean = {k: sum(s[k] for s in ss if s[k] != "") / max(1, sum(1 for s in ss if s[k] != ""))
                for k in ss[0] if isinstance(ss[0][k], (int, float)) and k not in ("step", "position")}
        mean["steps"] = len(ss)
        summary["by_step_kind"][kind] = mean
    decode_keys = {k for k, (j, kind) in index.items() if kind == "decode"}
    nd = max(1, sum(1 for s in steps if s["step_kind"] == "decode"))
    per = defaultdict(lambda: [0, 0.0, "", ""])
    for r in rows:
        if (r["request"], r["marker"]) in decode_keys:
            p = per[(r["part"], r["name"])]
            p[0] += 1
            p[1] += (r["end"] - r["start"]) / 1e3
            p[2] = r["category"]
    summary["kernels_per_decode_step"] = sorted(
        ({"part": part, "name": name, "category": c, "launches": n / nd, "us": t / nd}
         for (part, name), (n, t, c, _) in per.items()), key=lambda x: -x["us"])
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    return summary


def label(summary):
    run = summary["run"]
    return f"{summary['engine']} {run.get('weights', '')}".strip()


def print_summary(summaries, top):
    cols = [label(s) for s in summaries]
    w = max(12, *(len(c) + 1 for c in cols))
    print("\n" + " " * 34 + "".join(c.rjust(w) for c in cols))

    def line(name, values, fmt="{:.1f}"):
        print(f"  {name:32s}" + "".join((fmt.format(v) if v is not None else "-").rjust(w) for v in values))

    for s in summaries:
        for r in s["run"].get("results", []):
            print(f"  [{label(s)}] {r['prompt']}: {r['prompt_len']} prompt tokens, {r['out']} out, {r.get('steps', '-')} steps")
    for kind in ("prefill", "decode"):
        ms = [s["by_step_kind"].get(kind) for s in summaries]
        if not any(ms):
            continue
        print(f"\n  {kind} steps, mean µs per step")
        get = lambda k: [m.get(k) if m else None for m in ms]
        line("steps", get("steps"), "{:.0f}")
        line("kernels", get("kernels"), "{:.0f}")
        for k in ("period_us", "span_us", "busy_us", "host_us"):
            line(k[:-3], get(k))
        keys = sorted({k for m in ms if m for k in m if k.startswith(("part_", "cat_"))},
                      key=lambda k: (not k.startswith("part_"), -max((m or {}).get(k, 0) for m in ms)))
        for k in keys:
            if max((m or {}).get(k, 0) for m in ms) >= 1:
                line("  " + k, get(k))
    for s in summaries:
        print(f"\n  [{label(s)}] top kernels per decode step: part, category, name, launches, µs")
        for k in s["kernels_per_decode_step"][:top]:
            print(f"    {k['part']:7s} {k['category']:17s} {k['name'][:60]:60s} {k['launches']:7.1f} {k['us']:9.1f}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("reports", nargs="+")
    p.add_argument("--force", action="store_true", help="redo the sqlite and CSV exports")
    p.add_argument("--top", type=int, default=15)
    a = p.parse_args()
    summaries = [export(r, a.force) for r in a.reports]
    print_summary(summaries, a.top)


if __name__ == "__main__":
    main()
