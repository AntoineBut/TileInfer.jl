# Profiling kernels with Nsight Compute

`bench/run.jl` says how fast a kernel is. Nsight Compute (`ncu`) says why: how much of the
DRAM bandwidth it uses, how many warps are resident, what they wait on, and which
instructions they wait at. This page covers the one-time setup, the tuning loop, and how to
read the output.

## Setup (once per machine)

1. Install Nsight Compute with NVIDIA's `.run` installer (`sudo sh nsight_compute-linux-*.run`):
   accept the default path and the symlink at `/usr/local/NVIDIA-Nsight-Compute`, then
   `fish_add_path /usr/local/NVIDIA-Nsight-Compute`. `ncu` is the profiler, `ncu-ui` the
   viewer (on Wayland, `QT_QPA_PLATFORM=xcb ncu-ui` if no window appears).
2. Let non-root users read the GPU performance counters. Without this, `ncu` fails with
   `ERR_NVGPUCTRPERM`. The NVIDIA module is in the initramfs, so rebuild it (this machine
   boots with Limine, so use `limine-mkinitcpio`, not `mkinitcpio -P`) and reboot:

   ```sh
   echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-profiling.conf
   sudo limine-mkinitcpio
   ```

   Check with `grep RmProfilingAdminOnly /proc/driver/nvidia/params`, which should print 0.

## The tuning loop

Profile any benchmark of `bench/suite.jl` by the key `bench/run.jl` prints:

```sh
ncu --profile-from-start off --set full -c 1 -o bench/results/down_before \
    julia --project=bench bench/profile.jl ops/linear_down/decode/tile
python3 bench/ncu_summary.py bench/results/down_before.ncu-repz
# ... change the kernel ...
ncu --profile-from-start off --set full -c 1 -o bench/results/down_after \
    julia --project=bench bench/profile.jl ops/linear_down/decode/tile
python3 bench/ncu_summary.py bench/results/down_after.ncu-repz --compare bench/results/down_before.ncu-repz
```

Then confirm the speedup with `bench/run.jl --compare`: `ncu` serializes kernels and locks
clocks, so its durations are only for comparing kernels with each other.

`bench/profile.jl` compiles and warms up the matching benchmarks, then runs one sample of
each between `cuProfilerStart`/`Stop`. `--profile-from-start off` makes `ncu` profile only
inside that range, so compilation and warm-up launches stay out of the report.

| `ncu` option | Use |
|---|---|
| `-o PATH` | write `PATH.ncu-repz`; keep reports in `bench/results/` (gitignored) |
| `--set full` | all sections; needed for stall reasons and hot instructions. The default set is enough for the `--table` view. |
| `-c N` | profile only the first N launches. An op sample is 20 calls of the same kernel, so use `-c 1`. |
| `-k regex:NAME` | only kernels whose name matches |
| `-f` | overwrite an existing report |

### Model steps

`model/decode_ctx1024/tile` and `model/prefill512/tile` profile every kernel of a step. `ncu`
replays each kernel (8 passes with the default set, ~45 with `--set full`) and saves and
restores device memory around each pass, about 0.65 s per kernel even when the kernel is
tiny. A decode step launches 543 kernels, about 6 minutes under `ncu`: per layer 14 cuTile
kernels (4 GEMVs, 4 RMSNorms, 2 ropes, store_kv, silu_mul, attention decode + combine)
plus 5 CUDA.jl copies and broadcasts (q/k/v slices of the fused projection, residual adds).
Profile only the cuTile kernels with `-k`; add `-c 15` to stop after the embedding and the
first layer (about a minute):

```sh
ncu --profile-from-start off -k regex:'^(embed|rmsnorm|gemv|rope|store_kv|silu_mul|attention_\w+)_kernel' \
    -o bench/results/decode julia --project=bench bench/profile.jl model/decode_ctx1024/tile
python3 bench/ncu_summary.py bench/results/decode.ncu-repz
```

## Reading `ncu_summary.py`

With up to 4 kernels in a report (or `--kernel REGEX` narrowing it), each kernel gets:

- **Launch line:** grid, block, registers per thread, shared memory per block, waves per SM
  (below 1 means fewer blocks than the GPU can hold at once), and what limits occupancy.
- **Metrics:** time, DRAM bytes and bandwidth, DRAM and SM throughput as % of peak, cache
  hit rates, occupancy. A memory-bound kernel such as a decode GEMV should approach 90 %
  DRAM throughput; a tiny kernel (decode rmsnorm, rope) mostly measures launch overhead,
  and fusing it into a neighbour helps more than tuning it.
- **Stall cycles per issued instruction**, by reason. The common ones:

  | Reason | The warp is waiting for |
  |---|---|
  | `long_scoreboard` | a global (DRAM/L2) load; normal for memory-bound kernels, reduced by more loads in flight (more warps, or more independent loads per thread) |
  | `short_scoreboard` | shared memory or a special function (`exp`, `rsqrt`) |
  | `barrier` | other warps of the block at a synchronization |
  | `wait` | a fixed-latency dependency on the previous instruction |
  | `math_pipe_throttle` | a busy arithmetic unit (compute bound) |
  | `imc_miss`, `no_instructions` | instruction or constant cache misses, usually only at kernel start |

- **ncu suggestions:** `ncu`'s rule results with its estimated speedup, largest first
  (`--full` for complete texts). The estimates are upper bounds and often overlap.
- **Hottest instructions:** the SASS instructions where sampled warps were stalled, with the
  source line they come from. The stall is charged to the instruction that *uses* a value,
  so the hot line for a slow load is usually its first use, not the load itself. Lines in
  `cuTile/src/language/...` are cuTile's lowering of an op (`operations.jl` for loads and
  conversions); find the call in `src/ops/tile/` that produces it. `ncu` also has a Tile IR
  source view (`--print-source tileir`), but it reports no Tile IR source for cuTile.jl
  kernels yet.

Reports with more than 4 kernels (or `--table`) get one line per launch and a total per
kernel name, which is the place to start for a model step.

## Snapshot: decode step, 2026-10-27 (M5)

`model/decode_ctx1024/tile` (1,024 cached positions), default section set, all kernels:

| Kernels | Launches | GPU time | Share |
|---|---|---|---|
| cuTile `gemv_kernel` (4 per layer + lm_head) | 113 | 2.00 ms | 56.5 % |
| cuTile rmsnorm, rope, store_kv, silu_mul, embed | 226 | 0.79 ms | 22.4 % |
| cuTile `attention_decode_kernel` + `attention_combine_kernel` | 56 | 0.37 ms | 10.5 % |
| CUDA.jl `gpu_getindex_kernel` (q/k/v slices, cos/sin) and residual-add broadcasts | 144 | 0.38 ms | 10.6 % |

Times are at `ncu`'s locked clocks, without launch gaps, so only the shares are meaningful.
Before M5 the reference attention took ~89 % (~5,350 launches, ~24 ms). Now the GEMVs
dominate, and the many tiny kernels (~3 µs each here) are the case for fusion and CUDA graphs
(M6): in `bench/run.jl` the step takes 3.8 ms against 3.5 ms of kernel time, because two
cuTile launches already cost ~15 µs of host time.
