# cuTile codegen quirks

Measured behaviour of cuTile.jl and `tileiras` on the 4080S, with a reproducer for each:
what to report upstream, and what TileInfer's kernels must work around. Probes live in
`bench/quirks/probes/`; each runs kernels that compute the same thing written differently.

```sh
julia --project=bench bench/quirks/run.jl [NAME_FILTER...]    # all probes ~11 min
```

For each variant the harness (`bench/quirks/harness.jl`) prints the GPU time (CUPTI trace,
minimum and mean after a warm-up), block size, registers, local memory (spills), shared
memory, and the code at both compiler stages, saved in `bench/results/quirks/<probe>/`:
the Tile IR cuTile.jl emits (`.tileir`) and the SASS `tileiras` produces (`.sass`, from
`nvdisasm`). Comparing them tells whose behaviour a difference is:

| Tile IR of the variants | SASS / time | Origin | Report to |
|---|---|---|---|
| Differs beyond names and shapes | differs | cuTile.jl (Julia → Tile IR) | cuTile.jl |
| Same up to names, or only the tile shape | differs | `tileiras` (Tile IR → SASS) | NVIDIA (CUDA Tile IR) |
| Same | same | noise | — |

Environment (2026-10-29, re-run 2026-10-31): cuTile.jl 1.0.0 (the latest registered), tileiras 13.3.36,
CUDA.jl 6.3, Julia 1.13.3, RTX 4080 SUPER (sm_89, Ada). Raw output of the last full run:
`bench/results/quirks/report.txt`.

## Findings

| # | Quirk | Probe | Origin | Status |
|---|---|---|---|---|
| 1 | `mmaf` narrower than the 16 × 8 MMA shape is up to 8.4× slower and spills | `narrow_matmul`, `narrow_matmul_t` | tileiras | to report; TileInfer pads |
| 2 | Every block-wide reduction costs ~0.15 µs: 3 barriers and a shared-memory atomic | `reduce_kind`, `reduce_size` | tileiras | to report |
| 3 | Always 128 threads per block, with no hint to change it | `reduce_size`, `hint_block_size` | tileiras / cuTile.jl | to ask |
| 4 | `argmax` pays for NaN handling even with `propagate_nan = false` | `reduce_kind` | cuTile.jl | to report; TileInfer has its own reducer |
| 5 | `x >> 16` and `x << 3` on a scalar `UInt32` fail to compile | `shift_count_type` | cuTile.jl | to report (one-line fix) |
| 6 | `for X in (A, B)` over arrays fails to compile | `tuple_loop` | cuTile.jl | known limitation; workaround below |
| 7 | CUPTI reports 0 bytes of local memory for kernels that spill | `narrow_matmul_t` | CUPTI / CUDA.jl | harness reads the driver attribute |
| 8 | The `occupancy` hint alone can change results in the last bit | `knobs_attention_prefill_*` | tileiras | expected (summation order); noted |
| 9 | Converting two (64, 128) Int8 tiles to BF16 for MMAs in one loop body spills; halves fit | `int8_attention` | tileiras | expected (register pressure); TileInfer splits the tiles |

### 1. Narrow matrix products (tileiras)

`muladd(K, q, acc)` with K (64 × 128) and q (128 × R), as in decode attention, over 16
pages per block (160 blocks). cuTile.jl emits the column-major product as a row-major
`mmaf` of (R × 128) · (128 × 64), so R is the op's M; `muladd(qᵀ, Kᵀ)` puts R on N. The Tile
IR of the variants differs only in the shape.

| R | as M: µs | regs | spills | as N: µs | regs | spills |
|---|---|---|---|---|---|---|
| 32 | 17.9 | 140 | 0 | 17.9 | 138 | 0 |
| 16 | 14.1 | 104 | 0 | 13.8 | 111 | 0 |
| 8 | **115.4** | 255 | 212 LDL / 171 STL | 12.9 | 91 | 0 |
| 4 | **96.6** | 255 | 827 LDL / 223 STL | **52.2** | 174 | 0 (62 SHFL) |
| 2 | 22.2 | 206 | 0 | **47.5** | 255 | 904 B/thread |

Below Ada's m16n8k16 MMA shape (M < 16 or N < 8), tileiras does not pad to it but emits a
different code path, which spills at M = 4–8 and N = 2. Padding by hand to M = 16 is
6–8× faster than M = 4–8 although it does 2–4× the arithmetic.

TileInfer already pads: decode attention packs at least 16 query rows (`DECODE_MIN_ROWS`,
found by tuning: 15 → 10 µs) and GEMM tiles are never narrower than 16 tokens
(`gemm_config`), which covers spec-decoding verify steps of 2–8 tokens. **Rule: keep the M
side of every `muladd` ≥ 16 and the N side ≥ 8** (in Julia's terms: the second operand's
columns ≥ 16, the first operand's rows ≥ 8).

### 2. Cost of a block-wide reduction (tileiras)

20 dependent reductions over a Float32 tile in registers (each removes the element found,
as a top-k selection does), 38 blocks. Per reduction, after subtracting the 0-iteration
load/store baseline:

| Reduction (4096 elements) | µs per reduction |
|---|---|
| `sum` | ~0.12 |
| `maximum` | ~0.16 |
| `mapreduce` of (value, index) without NaN checks | ~0.24 |
| `argmax` | ~0.29 |
| `maximum`, then `minimum` of the positions holding it | ~0.31 |

By size, `maximum` costs ~0.13 µs at 256 elements, ~0.16 at 4,096 and ~0.42 at 16,384:
mostly a fixed cost. The SASS of one `maximum` (`reduce_size/N = 256.sass`): a warp
butterfly (5 `SHFL.BFLY`), then the cross-warp step through a single shared word:
initialize it, `BAR`, `REDUX.MAX` (recomputing the warp maximum the butterfly already
has) and `ATOMS.MAX` on the float encoded as an integer, `BAR`, `LDS`, `BAR`. Three
barriers and a shared atomic per reduction; with 4 warps, one slot per warp and a single
barrier would do. This bounds any kernel that reduces in a loop: TileInfer's top-k
sampling costs ~2 × top_k reductions (17 µs per token at top_k = 20).

### 3. Fixed block size (tileiras / cuTile.jl)

Every kernel probed runs 128 threads, from 256-element tiles to 16,384-element ones (155
registers per thread). `occupancy` changes only the register budget (16 caps it at 32
registers and runs 1.8× slower), `opt_level = 1` changes nothing measurable, and
`num_worker_warps` applies only to warp-specialized kernels. So a latency-bound kernel
(one warp per scheduler, ~8 % achieved occupancy under `ncu` for the sampling kernels)
cannot trade tile size for more threads. Question for upstream: is there, or will there
be, a way to request warps per block for non-warp-specialized kernels on sm_8x?

### 4. `argmax` NaN handling (cuTile.jl)

`ArgMaxReducer` (`src/language/operations.jl`) tests `isnan` on both operands of every
combine step whatever `propagate_nan` says; the flag only picks which side wins. For
logits, which are never NaN, a reducer without the checks (TileInfer's `FirstMax` in
`src/ops/tile/sampling.jl`, ties to the smaller index as `argmax`) is 1.2× faster per
reduction (0.24 vs 0.29 µs), and took the sampling kernels from 19.2 to 17.2 µs. Suggestion
for upstream: a NaN-free fast path (e.g. a keyword to skip the checks).

### 5. Scalar shifts by an `Int` (cuTile.jl)

```julia
x = X[Int32(1)]            # UInt32
Y[Int32(1)] = x >> 16      # Binary op type mismatch: lhs element type UInt32 != rhs element type Int64
Y[Int32(1)] = x >> UInt32(16)   # works
```

`<<` fails the same way, while the broadcast `tile .>> 16` works. The overlays in
`src/language/arithmetic.jl` pass the count straight to `shri`/`shli`; converting it to
the operand's type would fix it (Julia also defines counts ≥ the bit width, which give 0,
and negative counts, which shift the other way).

### 6. Loops over tuples of arrays (cuTile.jl)

`for X in (A, B); acc = acc .+ ct.load(X; ...); end` fails with "Cannot resolve LoopOp
init value". Recursion through an `@inline` helper compiles and works:
`add_all(acc, X, rest...) = add_all(acc .+ ct.load(X; ...), rest...)`.

### 7. Spills in the CUPTI trace (CUPTI / CUDA.jl)

`CUDA.@profile trace = true` reports `local_mem.thread = 0` for kernels whose SASS spills
(`narrow_matmul_t`, R = 2: 253 `LDL`, 185 `STL`). The driver attribute
`CUDA.FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES` of the compiled function gives 904 bytes. CUPTI
kernel durations also come in steps of ~0.24 µs, so compare means, not minima, for short
kernels.

### 8. Occupancy and bit-exactness (tileiras)

The same Tile IR of the prefill attention kernel compiled with `occupancy = 3` or `4`
(168 and 128 registers instead of 216) differs from occupancies 1–2 in a handful of
outputs, depending on the data (2 of 1,048,576 in one check, none in another), by one BF16
ulp, with the same error against the reference: a different summation order, not a
miscompile. Occupancies 1 and 2 (same registers, half the shared memory) are
bit-identical. So results are bit-reproducible only for a given set of compiler hints.

### 9. Int8 → BF16 conversion of MMA operands (tileiras)

A tile converted from Int8 to BF16 for an `mmaf` is not staged through shared memory as a
BF16 tile loaded from memory is: the bytes are shuffled into the MMA's register layout and
converted there, so the whole converted tile is live in registers. Decode attention over
Int8 keys and values (4B shapes, 32k positions, `int8_attention`): converting a (64, 128)
key page uses 178 registers, a value page 232, both 255 with 56 bytes of spills per thread
and 3.7× the SASS of the BF16 kernel. Smaller conversions fit: 32-key tiles (211
registers), K and V in halves of the head dimension (190), or both (184, kept in
TileInfer); each removes the spills and runs ~5–10 % faster in the probe (1.26× in the real
kernel, which spilled more). An Int8 × Int8 `mmai` for QKᵀ avoids the key conversion
(206 registers), at the cost of quantizing the queries. Storing V transposed, so the
contraction axis is contiguous for the PV product, did not help (255 registers, more
shuffles). Same class of cost as the INT8 GEMM's spills at 512 tokens (audit below).

## Not reproduced

Notes in CLAUDE.md came from the OKC experiments without reproducers; on sm_89 with
cuTile 1.0.0 they do not hold:

| Claim | Probes | Result |
|---|---|---|
| Logic moved into an `@inline` helper ran 1.7× slower | `helpers_gemv`, `helpers_topk`, `audit_decode_attention`, `audit_q_inputs` | Same SASS (or the same instruction counts and time) inline, in an `@inline` helper, a plain helper or a closure |
| A 3-D tile reduced over its third axis ran 60× slower | `reduce_axis`, `audit_combine` | Same time as the 2-D reduction (2.15 vs 2.15 µs; combine 1.91 vs 1.91). Loading with the reduced axis first is the slow form (1.4×: it transposes through shared memory) |
| Dropping a trivial size-1 axis ran 5× slower | `size1_axis` | Same time and instruction counts with `(H, 1)`, `reshape` or `dropdims` |

Also checked: a Float64 literal in Float32 code (`x .* 0.7`) compiles to the same IR as
`x .* 0.7f0`. cuTile converts scalars to the tile's element type, unlike Julia, where
`Float32 .* 0.7` is Float64.

## TileInfer audit

**Code layout.** TileInfer kernels against copies that differ only in code layout
(`05_tileinfer_audit.jl`): decode attention with `attend_page` (now `attend`) and `page_column` inlined
(same SASS), q inputs with `norm_rope` and its closures inlined (same instruction counts
and time), the attention combine as a 3-D reduction (same time), and RMSNorm with its
loads through a closure or reordered (same or slightly worse). The helpers in the GEMV and
sampling kernels compile to the same SASS too (`helpers_*`).

**Census** (`06_census.jl`, in `bench/results/quirks/report.txt`): every kernel of a decode
step (1 token at 1,024 positions), a spec-decoding verify step (8 and 16 tokens at 1,024
positions), and a 64- and 512-token prefill step, at the real shapes of Qwen3-0.6B and 4B
in BFloat16 and INT8 and of Apertus 1.5 8B in INT8. None uses a narrow `mmaf`. The slow
ones are mostly known design limits ([Performance](performance.md), "What to optimize"), not codegen:
`q_inputs_kernel` at 512 tokens (one block per token and head, ~330 GB/s), the prefill
GEMMs (70–75 % of the tensor-core peak). One oddity without cost: `rmsnorm_kernel`
compiles to twice the SASS of `add_rmsnorm_kernel` (tileiras unrolls its loops twice: 26
vs 13 global loads), runs once per step, and is ≤ 0.5 µs slower. New on 2026-10-31:

- **`gemm_int8_kernel` spills** at 512 tokens, on every model: 231–255 registers and 16
  bytes of local memory per thread, where the BF16 `gemm_kernel` with the same tiles uses
  222–236 and none (the Int8 → BF16 conversion of the weight tile adds live registers).
  The INT8 prefill-512 GEMMs are 1.06–1.26× slower than BF16 (4B down: 327 vs 259 µs)
  although they read half the weight bytes.
- **Attention for 9–16 tokens leaves the decode kernels**: `DECODE_MAX_TOKENS = 8`, so a
  16-token verify step (DFlash's block) runs the prefill kernel, 16–32 blocks over 1,024
  positions: 27–28 µs on every model against 6–9 µs at 1 and 8 tokens. Not a codegen
  quirk; a dispatch limit that costs ~0.6–0.7 ms per 16-token verify step (28–36 layers),
  ~5 % of a 4B verify. Now up to 16 tokens use the decode kernels, with 8 splits
  ([Performance](performance.md), "16-token verify attention").

**Knobs** (`07_knobs.jl`), never swept before on the heaviest kernels:

| Kernel | `latency` hint on the weight loads | `occupancy` |
|---|---|---|
| Decode GEMV (0.6B qkv, down; 4B gate_up, down) | no effect (1, 4, 10) | no effect (2, 4) |
| Prefill GEMM (n = 64, 512) | none better; `latency = 1` 1.2× slower | **2: up to 1.12× when the grid has ≥ 2 blocks per SM**, 0.86× when it has fewer |
| Verify-sized GEMM (n = 2–8) | — | 2 within 1 % of 1 |
| Prefill attention (asks for 2) | — | 1: 1.5× slower; 3: same in run.jl; 4: 1.3× slower |

The GEMM now launches with `occupancy = 2` when it has ≥ 2 blocks per SM
(`gemm_occupancy` in `src/ops/tile/linear.jl`): prefill 512 on 4B 1.06× faster, other
prefill steps 1.01–1.04×, see [Performance](performance.md).

No layout quirk costs TileInfer time in the kernels probed; the known cliff (narrow
`mmaf`) is already avoided by padding.
