# Getting started

```@meta
CurrentModule = TileInfer
```

## Requirements

- Julia 1.13.
- An NVIDIA GPU that cuTile.jl supports, with a recent driver. TileInfer is developed and
  tuned on an RTX 4080 SUPER (Ada, sm_89, 16 GB); other cards should work but are untested.
- Enough GPU memory for the model in BFloat16: Qwen3-0.6B fits anywhere, Qwen3-4B needs
  about 9 GB, and 8B models need `--int8` on a 16 GB card.

## Installation

TileInfer is not registered. Clone it and instantiate the environment:

```sh
git clone https://github.com/AntoineBut/TileInfer.jl
cd TileInfer.jl
julia --project -e 'using Pkg; Pkg.instantiate()'
```

The command line can run from the clone with `julia --project -m TileInfer <command>`, or be
installed as a `tileinfer` executable in `~/.julia/bin` (a Julia Pkg app):

```sh
julia --project -e 'using Pkg; Pkg.Apps.develop(path = ".")'
```

## Command line

```sh
tileinfer pull Qwen/Qwen3-0.6B                     # download weights, configs, tokenizer
tileinfer ls                                       # list pulled models
tileinfer run Qwen/Qwen3-0.6B "Why is the sky blue?"
tileinfer chat Qwen/Qwen3-4B --thinking off        # multi-turn; /reset clears, /exit quits
tileinfer chat Qwen/Qwen3-4B --dflash z-lab/Qwen3-4B-DFlash-b16   # speculative decoding
tileinfer rm Qwen/Qwen3-0.6B
```

Models go to the Hugging Face hub cache (`~/.cache/huggingface/hub`), shared with Python's
`huggingface_hub`. Gated repositories need a token in `HF_TOKEN` or
`~/.cache/huggingface/token`. `tileinfer help` lists every option: sampling
(`--temperature`, `--top-k`, `--top-p`, `--seed`), context size, `--int8` weights,
`--kv-int8` cache, the reference backend, and the speculative decoding proposers.

The first reply in a fresh process includes compiling the kernels (a few seconds); the
statistics line printed after each reply measures decode speed from the second token on.

## Julia API

The command line is a thin layer over the functions below; they are all listed with their
docstrings in the [Public API](api.md).

### Generate a reply

```julia
using TileInfer
using TileInfer: pull, load_model, load_tokenizer, generation_defaults, chat_prompt,
                 ChatMessage, encode, decode, generate, TextStream

dir = pull("Qwen/Qwen3-0.6B")                    # or model_dir(...) once pulled
model = load_model(dir; backend = TileBackend())  # the default backend is RefBackend()
tok = load_tokenizer(dir)
sampling, stop_ids = generation_defaults(dir, model.config)   # the authors' sampling settings

prompt = chat_prompt(model.config.model_type, [ChatMessage("user", "Name three primary colors.")];
                     enable_thinking = false)
ids = encode(tok, prompt; add_special_tokens = false)  # the template already holds any BOS
out = generate(model, ids; max_new_tokens = 256, sampling, stop_ids,
               on_token = TextStream(tok))       # streams the text to stdout
decode(tok, out)
```

[`load_model`](@ref) takes `T` (the element type, `BFloat16` by default), `max_len` (the
longest sequence, which sizes the RoPE tables) and `quantize = :int8`. [`generate`](@ref)
allocates a KV cache for one prompt; pass `kv_quantize = :int8` for an INT8 cache.

### Chat

A [`ChatSession`](@ref) keeps one KV cache for the whole conversation and only runs the
tokens a new turn adds:

```julia
using TileInfer: ChatSession, reply!, reset!

session = ChatSession(model, tok; sampling, stop_ids, enable_thinking = false)
reply, stats = reply!(session, "Name three primary colors.")
reply, stats = reply!(session, "And three secondary ones?")   # reuses the first turn's cache
stats.reused, stats.prefilled, stats.generated
reset!(session)
```

### Speculative decoding

Pass a proposer as `speculative`; the output is the same as without it (exactly for greedy
decoding, in distribution for sampling), only faster when drafts are accepted:

```julia
using TileInfer: NgramProposer, DraftProposer, DFlashProposer, load_dflash, SpecStats

stats = SpecStats()
generate(model, ids; sampling, stop_ids, speculative = NgramProposer(k = 4), spec_stats = stats)

# For a Qwen3-4B target: a smaller model with the same vocabulary drafts, or a DFlash
# drafter trained for that target drafts a block of tokens in one pass.
target = load_model("Qwen/Qwen3-4B"; backend = TileBackend())
draft = DraftProposer(load_model("Qwen/Qwen3-0.6B"; backend = TileBackend()); k = 4)
dflash = DFlashProposer(load_dflash("z-lab/Qwen3-4B-DFlash-b16", target))
generate(target, ids; sampling, stop_ids, speculative = dflash)
```

### Lower level

[`ModelRunner`](@ref), [`PagedKVCache`](@ref), [`Sequence`](@ref) and
[`forward!`](@ref) expose single forward steps, and the ops ([`linear`](@ref),
[`attention`](@ref), …) can be called directly on device arrays with either backend; see
the [Public API](api.md). The tests in `test/` are good examples of both.

## Development

Tests, benchmarks and profiling scripts live in `test/` and `bench/`, which are workspace
projects sharing the root manifest:

```sh
julia --project=test test/runtests.jl            # full suite (~4 min, needs the GPU)
julia --project=test test/runtests.jl kernels    # only files whose path contains "kernels"
julia --project=bench bench/run.jl --filter decode
julia --project=docs docs/make.jl                # this site, into docs/build/
```

Tests that need a real checkpoint use Qwen/Qwen3-0.6B and are skipped when it is not
pulled. The [Profiling](profiling.md) page describes the kernel profiling workflow.
