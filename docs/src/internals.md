# Internals

```@meta
CurrentModule = TileInfer
```

Everything that is documented but not public, grouped by source file. None of it is a stable
interface: it is here because this project is meant to be read. The design behind it is in
[Architecture](architecture.md); measured kernel rules are in [cuTile quirks](cutile_quirks.md).

## Configuration and loading

```@autodocs
Modules = [TileInfer]
Public = false
Pages = ["config.jl", "hub/store.jl", "hub/loader.jl"]
```

## Model

```@autodocs
Modules = [TileInfer]
Public = false
Pages = ["model/weights.jl", "model/rope.jl", "model/decoder.jl"]
```

## KV cache

```@autodocs
Modules = [TileInfer]
Public = false
Pages = ["kv/block_manager.jl", "kv/paged_cache.jl"]
```

## Tokenizer and chat templates

```@autodocs
Modules = [TileInfer]
Public = false
Pages = ["tokenizer/bpe.jl", "tokenizer/chat_template.jl"]
```

## Model runner and sampling

```@autodocs
Modules = [TileInfer]
Public = false
Pages = ["engine/runner.jl", "engine/sampler.jl", "engine/generate.jl", "engine/chat.jl"]
```

## Speculative decoding

```@autodocs
Modules = [TileInfer]
Public = false
Pages = ["engine/speculative.jl", "engine/dflash.jl"]
```

## Op interface, reference backend and quantization

```@autodocs
Modules = [TileInfer]
Public = false
Pages = ["ops/interface.jl", "ops/reference.jl", "ops/quant.jl"]
```

## cuTile kernels

One file per kernel family in `src/ops/tile/`. Each `*_kernel` function runs on the GPU; the
launchers around it pick tile shapes and grids.

### Normalization

```@autodocs
Modules = [TileInfer]
Public = false
Pages = ["ops/tile/norm.jl"]
```

### Element-wise: RoPE, activations, KV writes, embedding

```@autodocs
Modules = [TileInfer]
Public = false
Pages = ["ops/tile/elementwise.jl"]
```

### Linear layers: GEMV and GEMM

```@autodocs
Modules = [TileInfer]
Public = false
Pages = ["ops/tile/linear.jl"]
```

### Fused attention inputs

```@autodocs
Modules = [TileInfer]
Public = false
Pages = ["ops/tile/fused.jl"]
```

### Paged flash-attention

```@autodocs
Modules = [TileInfer]
Public = false
Pages = ["ops/tile/attention.jl"]
```

### Device sampling

```@autodocs
Modules = [TileInfer]
Public = false
Pages = ["ops/tile/sampling.jl"]
```

## Command line

```@autodocs
Modules = [TileInfer]
Public = false
Pages = ["cli.jl"]
```
