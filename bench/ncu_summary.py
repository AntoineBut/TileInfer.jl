"""Compact summary of Nsight Compute reports, for iterating on kernel performance.

Usage (no dependencies beyond the ncu install; NCU_HOME overrides its location):
    python3 bench/ncu_summary.py REPORT [--compare BASELINE] [--kernel REGEX]
                                        [--top N] [--table] [--full]

REPORT is a .ncu-rep(z) written by bench/profile.jl. For each kernel it prints the launch
shape, time, DRAM traffic, occupancy, warp stall reasons, ncu's own suggestions (sorted by
the speedup ncu estimates) and the instructions where sampled warps were stalled most.
Reports with more than a few kernels (a model step) get one line per kernel instead, and
a total per kernel name; --table forces that view.

--compare puts a baseline report's numbers next to each metric, matching kernels by name
and order of launch. Stall samples per instruction need `--set full` (or the
SourceCounters section) at profile time.
"""

import argparse
import os
import re
import sys
from collections import defaultdict

NCU_HOME = os.environ.get("NCU_HOME", "/usr/local/NVIDIA-Nsight-Compute")
sys.path.append(os.path.join(NCU_HOME, "extras", "python"))
import ncu_report  # noqa: E402

STALL_PREFIX = "smsp__average_warps_issue_stalled_"
STALL_SUFFIX = "_per_issue_active.ratio"
PC_SAMPLES = "smsp__pcsamp_sample_count"
PC_STALL_PREFIX = "smsp__pcsamp_warps_issue_stalled_"
# Not stalls: a warp that was picked to issue, or was ready but another warp was picked.
NOT_STALLS = {"selected", "not_selected"}


class Kernel:
    """The metrics of one profiled kernel launch."""

    def __init__(self, action, index):
        self.action = action
        self.name = action.name()
        self.index = index  # launches of this kernel name before this one
        self.names = set(action.metric_names())

    def get(self, name, default=None):
        if name not in self.names:
            return default
        m = self.action.metric_by_name(name)
        if m.num_instances() > 1 or not m.has_value():
            return default
        return m.value()

    def stalls(self):
        """Average cycles each issued instruction waited, per stall reason, largest first."""
        out = []
        for name in self.names:
            if name.startswith(STALL_PREFIX) and name.endswith(STALL_SUFFIX):
                reason = name[len(STALL_PREFIX) : -len(STALL_SUFFIX)]
                value = self.get(name, 0.0)
                if reason not in NOT_STALLS and value > 0:
                    out.append((reason, value))
        return sorted(out, key=lambda r: -r[1])

    def instanced(self, name):
        """{pc: value} of a metric sampled per instruction."""
        if name not in self.names:
            return {}
        m = self.action.metric_by_name(name)
        if not m.has_correlation_ids():
            return {}
        ids = m.correlation_ids()
        return {ids.as_uint64(i): m.as_double(i) for i in range(m.num_instances())}

    def hot_instructions(self, top):
        """[(pc, share of samples, sass, source line, top stall reasons)], hottest first."""
        samples = self.instanced(PC_SAMPLES)
        total = sum(samples.values())
        if not total:
            return []
        reasons = {}
        for name in self.names:
            if name.startswith(PC_STALL_PREFIX) and not name.endswith("_not_issued"):
                reason = name[len(PC_STALL_PREFIX) :]
                if reason not in NOT_STALLS:
                    reasons[reason] = self.instanced(name)
        out = []
        for pc, n in sorted(samples.items(), key=lambda s: -s[1])[:top]:
            why = sorted(((r, v.get(pc, 0.0)) for r, v in reasons.items()), key=lambda r: -r[1])
            why = [(r, v / n) for r, v in why[:2] if v > 0]
            info = self.action.source_info(pc)
            line = f"{short_path(info.file_name())}:{info.line()}" if info else ""
            out.append((pc, n / total, self.action.sass_by_pc(pc).strip(), line, why))
        return out

    def rules(self):
        """[(estimated speedup %, title, message)] of ncu's rule results, largest first."""
        out = []
        for rule in self.action.rule_results_as_dicts():
            msg = rule.get("rule_message") or {}
            text = strip_links(msg.get("message", ""))
            if not text or "speedup_estimation" not in rule:
                continue
            title = msg.get("title") or rule["name"]
            out.append((rule["speedup_estimation"]["speedup"], title, text))
        return sorted(out, key=lambda r: -r[0])


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def short_path(path):
    """Repo files relative to the repo, package files as package/path (no version slug)."""
    if path.startswith(REPO + os.sep):
        return os.path.relpath(path, REPO)
    m = re.search(r"/packages/([^/]+)/[^/]+/(.*)", path)
    return f"{m.group(1)}/{m.group(2)}" if m else os.path.basename(path)


def strip_links(text):
    """Rule messages link to report pages as @kind:target:text@; keep the text."""
    return re.sub(r"@\w+:(?:[^:@]*:)*([^:@]*)@", r"\1", text).strip()


def load(path, pattern):
    ctx = ncu_report.load_report(path)
    kernels, seen = [], defaultdict(int)
    for r in range(ctx.num_ranges()):
        rng = ctx.range_by_idx(r)
        for a in range(rng.num_actions()):
            action = rng.action_by_idx(a)
            name = action.name()
            if pattern and not re.search(pattern, name):
                continue
            kernels.append(Kernel(action, seen[name]))
            seen[name] += 1
    return kernels


def fmt_time(ns):
    return f"{ns:.0f} ns" if ns < 1e3 else f"{ns / 1e3:.2f} µs" if ns < 1e6 else f"{ns / 1e6:.2f} ms"


def fmt_bytes(b):
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if b >= scale:
            return f"{b / scale:.2f} {unit}"
    return f"{b:.0f} B"


def shorten(text, width):
    """The first sentence or two of `text`, at most `width` characters."""
    text = " ".join(text.split())
    if len(text) <= width:
        return text
    cut = text.rfind(". ", 0, width)
    return text[: cut + 1] if cut > width // 3 else text[: width - 1] + "…"


# (label, metric, formatter, lower is better) for the comparable part of the summary.
SUMMARY = [
    ("time", "gpu__time_duration.sum", fmt_time, True),
    ("DRAM read", "dram__bytes_read.sum", fmt_bytes, True),
    ("DRAM write", "dram__bytes_write.sum", fmt_bytes, True),
    ("DRAM bandwidth", "dram__bytes.sum.per_second", lambda v: f"{v / 1e9:.0f} GB/s", False),
    ("DRAM % of peak", "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed", lambda v: f"{v:.1f} %", False),
    ("SM % of peak", "sm__throughput.avg.pct_of_peak_sustained_elapsed", lambda v: f"{v:.1f} %", None),
    ("issue slots busy", "smsp__issue_active.avg.pct_of_peak_sustained_active", lambda v: f"{v:.1f} %", None),
    ("L1 hit rate", "l1tex__t_sector_hit_rate.pct", lambda v: f"{v:.1f} %", None),
    ("L2 hit rate", "lts__t_sector_hit_rate.pct", lambda v: f"{v:.1f} %", None),
    ("occupancy, theoretical", "sm__maximum_warps_per_active_cycle_pct", lambda v: f"{v:.1f} %", None),
    ("occupancy, achieved", "sm__warps_active.avg.pct_of_peak_sustained_active", lambda v: f"{v:.1f} %", None),
    ("instructions", "smsp__inst_executed.sum", lambda v: f"{v:,.0f}", True),
]

OCCUPANCY_LIMITS = ("registers", "shared_mem", "warps", "blocks")


def change(now, base, lower_better):
    if not base:
        return ""
    pct = 100 * (now - base) / base
    if abs(pct) < 0.5:
        return "  ="
    verdict = "" if lower_better is None else "  better" if (pct < 0) == lower_better else "  worse"
    return f"  {pct:+.1f} %{verdict}"


def details(k, base, args):
    g = k.get
    grid = "x".join(str(int(g(f"launch__grid_dim_{d}", 1))) for d in "xyz")
    block = "x".join(str(int(g(f"launch__block_dim_{d}", 1))) for d in "xyz")
    print(f"== {k.name} #{k.index}  grid {grid}  block {block}  "
          f"regs {g('launch__registers_per_thread', 0):.0f}  "
          f"smem {fmt_bytes(g('launch__shared_mem_per_block', 0))}/block  "
          f"waves/SM {g('launch__waves_per_multiprocessor', 0):.2f}")
    limits = {lim: g(f"launch__occupancy_limit_{lim}") for lim in OCCUPANCY_LIMITS}
    limits = {lim: v for lim, v in limits.items() if v is not None}
    if limits:
        lim, v = min(limits.items(), key=lambda kv: kv[1])
        print(f"   occupancy limited by {lim} ({v:.0f} blocks/SM); "
              f"clocks SM {g('sm__cycles_elapsed.avg.per_second', 0) / 1e9:.2f} GHz, "
              f"DRAM {g('dram__cycles_elapsed.avg.per_second', 0) / 1e9:.2f} GHz")
    if base:
        print(f"   {'':24} {'now':>12} {'baseline':>12}")
    for label, metric, fmt, lower_better in SUMMARY:
        v = g(metric)
        if v is None:
            continue
        line = f"   {label:24} {fmt(v):>12}"
        if base and base.get(metric) is not None:
            b = base.get(metric)
            line += f" {fmt(b):>12}{change(v, b, lower_better)}"
        print(line)

    stalls = k.stalls()
    if stalls:
        print("   stall cycles per issued instruction: "
              + ", ".join(f"{r} {v:.1f}" for r, v in stalls[:5]))
        if base and base.stalls():
            print("                              baseline: "
                  + ", ".join(f"{r} {v:.1f}" for r, v in base.stalls()[:5]))

    rules = [r for r in k.rules() if r[0] >= 1.0 or args.full]
    if rules:
        print("   ncu suggestions (estimated speedup):")
        for speedup, title, text in rules:
            text = text if args.full else shorten(text, 160)
            print(f"     {speedup:4.0f} %  {title}: {text}")

    hot = k.hot_instructions(args.top)
    if hot:
        start = min(k.instanced(PC_SAMPLES))
        print("   hottest instructions (share of stall samples, main reasons):")
        for pc, share, sass, line, why in hot:
            reasons = ", ".join(f"{r} {100 * v:.0f} %" for r, v in why)
            where = f"  [{line}]" if line else ""
            print(f"     {100 * share:4.1f} %  +{pc - start:#06x}  {sass:40}  {reasons}{where}")
    print()


def table(kernels, baseline):
    """One line per launch, then the total time per kernel name."""
    print(f"{'kernel':40} {'#':>3} {'time':>10} {'DRAM %':>7} {'SM %':>6} {'occ %':>6}  top stall")
    totals = defaultdict(lambda: [0, 0.0])
    for k in kernels:
        t = k.get("gpu__time_duration.sum", 0.0)
        totals[k.name][0] += 1
        totals[k.name][1] += t
        stall = k.stalls()[:1]
        print(f"{k.name[:40]:40} {k.index:>3} {fmt_time(t):>10} "
              f"{k.get('gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed', 0):7.1f} "
              f"{k.get('sm__throughput.avg.pct_of_peak_sustained_elapsed', 0):6.1f} "
              f"{k.get('sm__warps_active.avg.pct_of_peak_sustained_active', 0):6.1f}  "
              + (f"{stall[0][0]} {stall[0][1]:.1f}" if stall else ""))

    base_totals = defaultdict(float)
    for k in baseline or []:
        base_totals[k.name] += k.get("gpu__time_duration.sum", 0.0)
    grand = sum(t for _, t in totals.values())
    print(f"\n{'kernel':40} {'calls':>5} {'total':>10} {'share':>6}" + ("  vs baseline" if baseline else ""))
    for name, (n, t) in sorted(totals.items(), key=lambda kv: -kv[1][1]):
        line = f"{name[:40]:40} {n:5} {fmt_time(t):>10} {100 * t / grand:5.1f}%"
        if baseline:
            b = base_totals.get(name)
            line += f"  {fmt_time(b)}{change(t, b, True)}" if b else "  (new)"
        print(line)
    print(f"{'all':40} {len(kernels):5} {fmt_time(grand):>10}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("report")
    p.add_argument("--compare", metavar="BASELINE", help="baseline report to compare against")
    p.add_argument("--kernel", metavar="REGEX", help="only kernels whose name matches")
    p.add_argument("--top", type=int, default=6, help="hottest instructions to list (default 6)")
    p.add_argument("--table", action="store_true", help="one line per kernel")
    p.add_argument("--full", action="store_true", help="full suggestion texts, including small ones")
    args = p.parse_args()

    kernels = load(args.report, args.kernel)
    if not kernels:
        sys.exit("no kernel in the report matches")
    baseline = load(args.compare, args.kernel) if args.compare else None
    if args.table or len(kernels) > 4:
        table(kernels, baseline)
        return
    by_key = {(k.name, k.index): k for k in baseline or []}
    for k in kernels:
        details(k, by_key.get((k.name, k.index)), args)


if __name__ == "__main__":
    main()
