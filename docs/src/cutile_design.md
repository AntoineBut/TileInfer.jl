# cuTile design choices

Some of TileInfer's kernels are written in ways that would look arbitrary without context: a
query tile padded with empty rows, a hand-written argmax, loops unrolled through recursion,
arrays reshaped before every launch. Each of these follows from measured behaviour of
cuTile.jl, its compiler `tileiras` or CUDA.jl on the RTX 4080 SUPER. This page lists those
choices with the measurements behind them, so that they are not "simplified" back.

Measured with cuTile.jl 1.0.0, tileiras 13.3, CUDA.jl 6.3 and Julia 1.13.3 (sm_89, Ada).
These are properties of those versions, and each is worth re-measuring after an upgrade.

## Pad matrix products to at least 16 × 8

**Choice.** Decode attention packs at least 16 query rows into a tile (`DECODE_MIN_ROWS`),
even when a step has 2 query heads per KV head and one token; GEMM tiles are never narrower
than 16 tokens (`gemm_config`), which covers verify steps of 2–8 tokens.

**Why.** Below Ada's 16 × 8 tensor-core shape, the compiler does not pad but takes another
code path, which spills registers. A product `K (64 × 128) · q (128 × R)` as in decode
attention, with R on the MMA's M side:

| R | µs | registers | spills |
|---|---|---|---|
| 32 | 17.9 | 140 | none |
| 16 | 14.1 | 104 | none |
| 8 | 115.4 | 255 | 212 loads, 171 stores |
| 4 | 96.6 | 255 | 827 loads, 223 stores |

Padding to 16 rows does 2–4× the arithmetic and runs 6–8× faster. With R on the N side,
the cliff is below 8. In the real kernel, padding from 2 to 16 rows cut decode attention
from 15 to 10 µs. **Rule:** keep the M side of every `muladd` ≥ 16 and the N side ≥ 8 (in
Julia's terms, the second operand's columns and the first operand's rows).

## Keep launches under 32 kernel parameters

**Choice.** Kernels take 2-D views of the KV pools and per-head tensors
(`(head_dim, page_size × kv_heads × pages)` rather than a 4-D array), and an INT8 layer's
keys and values share one Int8 array and one scale array.

**Why.** A `TileArray{T,N}` argument flattens to 1 + 2N parameters (pointer, sizes,
strides). From 32 parameters on, CUDA.jl packs the arguments through an untyped generic path
that costs ~15 µs and ~430 allocations per launch. The KV-cache write kernel became 1.7×
faster with 2-D views, and separate key and value arrays would take the INT8 one to 36
parameters.

## Keep array shapes stable

**Choice.** The block table on the device has a fixed length, padded with zeros to a multiple
of 16 pages, rather than growing with the sequence.

**Why.** cuTile specializes a kernel on the divisibility and contiguity of each array, not
only on its type, so a new length can mean a new compilation. Before the padding, the
attention kernels recompiled at page boundaries during decoding: a 50-token reply ran at
55 tok/s instead of 133. A captured CUDA graph also needs its arrays to keep their size.

## Few reductions inside loops

**Choice.** Sampling selects the top-k candidates of each chunk of the vocabulary in parallel
blocks, then merges the chunks' candidates in one block, rather than reducing over the whole
vocabulary k times in one block.

**Why.** Each block-wide reduction has a mostly fixed cost, whatever the tile size:

| Reduction over 4,096 elements | µs |
|---|---|
| `sum` | ~0.12 |
| `maximum` | ~0.16 |
| (value, index) maximum without NaN checks | ~0.24 |
| `argmax` | ~0.29 |

`maximum` costs ~0.13 µs at 256 elements and ~0.16 at 4,096: the compiled code is a warp
shuffle tree followed by a cross-warp step with three barriers and a shared-memory atomic.
A top-k selection by repeated argmax pays this about 2 × k times, which bounds the sampling
kernels (~17 µs per token at top-k = 20).

## A hand-written argmax for logits

**Choice.** The sampling kernels reduce with their own `FirstMax` (`src/ops/tile/sampling.jl`)
instead of `argmax`: the larger (value, position) pair, ties to the smaller position.

**Why.** `argmax` checks both operands for NaN at every combine step, also with
`propagate_nan = false`. Logits are never NaN (padding is -Inf), and without the checks a
reduction is 1.2× faster (0.24 vs 0.29 µs). It took the sampling kernels from 19.2 to
17.2 µs.

## Block size, occupancy and latency hints

**Choice.** Kernels are tuned through tile shapes and the `occupancy` hint, never the block
size. GEMMs launch with `occupancy = 2` only when the grid has at least two blocks per SM
(`gemm_occupancy`); no kernel uses `latency` hints.

**Why.** Every kernel runs 128 threads per block, from 256-element tiles to 16,384-element
ones, and there is no hint to change that outside warp-specialized kernels; `occupancy` only
changes the register budget. Sweeps of the hints on the heaviest kernels:

| Kernel | `occupancy` | `latency` on weight loads |
|---|---|---|
| prefill GEMM (64, 512 tokens) | 2: up to 1.12× faster with ≥ 2 blocks per SM, 0.86× with fewer | no gain; 1 is 1.2× slower |
| decode GEMV | no effect (2, 4) | no effect (1, 4, 10) |
| verify-sized GEMM (2–8 tokens) | 2 within 1 % of 1 | — |
| prefill attention (uses 2) | 1: 1.5× slower; 4: 1.3× slower | — |

Bit-exactness depends on these hints: the same prefill attention kernel compiled with
`occupancy` 3 or 4 differs in a handful of outputs by one BF16 ulp (a different summation
order), so two kernels can only be expected to agree bit for bit at the same hints.

## Convert Int8 tiles in small pieces

**Choice.** INT8 attention converts 32 keys at a time, and splits keys, values, queries and
the accumulator into two halves of the head dimension. INT8 GEMMs have tiles of their own
(`gemm_int8_config`), narrower than the BF16 ones at 512 tokens.

**Why.** An Int8 tile converted to BF16 for the tensor cores is converted in registers, so the
whole converted tile is live at once. Converting both a `(64, 128)` key page and a value page
in one loop body needs 255 registers and spills (178 and 232 for each alone). The smaller
pieces need 184 registers without spills, and made the real decode kernel 1.26× faster.

## Language workarounds

- **Scalar shifts need a count of the operand's type**: `x >> UInt32(16)` for a `UInt32`;
  `x >> 16` fails to compile on a scalar, while the broadcast `tile .>> 16` works.
- **No `for` loops over tuples of arrays** (`for X in (A, B)` fails with "Cannot resolve
  LoopOp init value"). Kernels unroll through an `@inline` recursive helper instead:
  `add_all(acc, X, rest...) = add_all(acc .+ ct.load(X; ...), rest...)`.
- **Float64 literals are fine**: `x .* 0.7` compiles to the same code as `x .* 0.7f0`,
  because cuTile converts scalars to the tile's element type.

## What does not matter

These were measured not to change the compiled code or its speed, so kernels are free to use
them for readability:

- moving logic into helper functions, `@inline` or not, or into closures (same SASS);
- reducing a 3-D tile over its third axis instead of a 2-D one (same time);
- keeping or dropping a size-1 axis (`(H, 1)`, `reshape` or `dropdims`: same code).

Loading a tile with the reduced axis first is slower (1.4×): it transposes through shared
memory.

## How these were measured

Each finding has a probe in `bench/codegen/probes/`: kernels that compute the same thing
written differently, measured side by side.

```sh
julia --project=bench bench/codegen/run.jl [NAME_FILTER...]    # all probes, ~11 min
```

For each variant the harness (`bench/codegen/harness.jl`) reports the GPU time (minimum and
mean over many launches after a warm-up), registers, shared memory, local memory (spills,
read from the driver: the CUPTI trace reports 0 for kernels that spill), and the code at
both compiler stages, saved in `bench/results/codegen/<probe>/`: the Tile IR cuTile.jl emits
and the SASS `tileiras` produces. Comparing the two stages shows where a difference comes
from. Probes other than the TileInfer audit use only CUDA.jl and cuTile.jl, so each kernel
pair is a self-contained reproducer.

The census probe also lists every kernel of real decode, verify and prefill steps of
Qwen3-0.6B, Qwen3-4B and Apertus 1.5 8B with its time, bandwidth, registers and spills; none
uses a matrix product narrower than 16 × 8.
