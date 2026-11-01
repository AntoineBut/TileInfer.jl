# Public API

```@meta
CurrentModule = TileInfer
```

The names below are declared `public` in `src/TileInfer.jl`. Only the backend types are
exported, so qualify the rest (`TileInfer.generate`) or import them
(`using TileInfer: generate, load_model`). See [Getting started](usage.md) for how they fit
together, and [Internals](internals.md) for everything else.

Conventions shared by every function here: arrays are column-major with PyTorch's shapes
reversed (hidden states `(hidden, tokens)`, linear weights `(in, out)` applied as `y = Wᵀx`),
token ids are 0-based as in the Hugging Face vocabularies, and the element type `T` of a
model (BFloat16 by default) is a type parameter.

```@docs
TileInfer
```

## Models and the hub

```@docs
pull
model_dir
list_models
remove_model
CachedModel
ModelConfig
load_config
DecoderLM
load_model
max_length
forward!
```

## Tokenizer and chat templates

```@docs
BPETokenizer
load_tokenizer
encode
decode
ChatMessage
chat_prompt
```

## KV cache

```@docs
PagedKVCache
Sequence
new_sequence
truncate!
release!
OutOfPagesError
```

## Generation

```@docs
SamplingParams
generate
generate!
generation_defaults
TextStream
ModelRunner
ChatSession
reply!
reset!
TurnStats
```

## Speculative decoding

```@docs
NgramProposer
DraftProposer
DFlashProposer
DFlashDrafter
load_dflash
propose
SpecStats
acceptance_rate
```

## Backends and ops

Every op has a [`RefBackend`](@ref) method (CUDA.jl broadcasts and cuBLAS) and a
[`TileBackend`](@ref) method (cuTile kernels), and an in-place form that the model's forward
step calls on preallocated buffers.

```@docs
Backend
RefBackend
TileBackend
```

### Primitive ops

```@docs
embed
rmsnorm
linear
rope
silu_mul
xielu
XIELU
store_kv!
attention
attention_workspace
```

### In-place ops

```@docs
embed!
rmsnorm!
linear!
rope!
silu_mul!
xielu!
attention!
```

### Fused ops

```@docs
attention_inputs!
add_rmsnorm!
linear_swiglu!
linear_xielu!
```

### Sampling

```@docs
sample_token!
sample_tokens!
accept_token!
```

### Quantization

```@docs
Int8Weight
quantize_int8
Int8Pages
```

### Capabilities

```@docs
supports_graphs
```

## Command line

```@docs
cli
```
