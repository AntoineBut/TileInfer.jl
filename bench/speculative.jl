# Speculative decoding on chat prompts: plain decoding (pipelined) against n-gram lookup,
# a draft model and a DFlash drafter, for several draft lengths k (DFlash also with its
# whole block). Acceptance depends on the text, so unlike bench/run.jl this generates
# replies to fixed prompts (bench/prompts.jl): two tasks on a Julia source file, one that
# copies its input and one open-ended, cut to fit --context prompt tokens. The defaults are
# the reference point of docs/src/performance.md. --verify prefill runs every step of more
# than one token (verify steps, the drafters' passes) on the prefill attention kernel
# instead of the split-KV decode kernels (see bench/variants.jl).
#
#   julia --project=bench bench/speculative.jl [--model REPO] [--draft REPO|none] [--dflash REPO|none]
#       [--k 1,2,4,7] [--tokens N] [--context N] [--trials N] [--sampled [--pq]] [--filter REGEX]
#       [--verify decode|prefill]
#
# Defaults: Qwen/Qwen3-4B drafted by n-gram lookup and z-lab/Qwen3-4B-DFlash-b16 (for
# Qwen3-4B; none otherwise), k = 2, 4, 7; a draft model only with --draft (Qwen/Qwen3-0.6B
# for Qwen3-4B); 1,024-token prompts, 128 new tokens, greedy, best of 5
# (--sampled: the model's recommended sampling, same seed for every method; --pq: the model
# proposers sample their drafts too, verified by the p/q rule). Decode tok/s
# runs from the first new token to the last, so it excludes the target's prefill but includes
# the draft model's (it prefills during the first proposal). Prefill is the time to the first
# token: the target's prefill and first sample. Each case is the best of
# `trials` runs after a warm-up run that compiles and captures every step.

include("variants.jl")
include("prompts.jl")
using Printf: @printf, @sprintf
using Random: Xoshiro
using TileInfer
using TileInfer: DecoderLM, DFlashProposer, DraftProposer, NgramProposer, SpecStats,
                 generation_defaults, load_dflash, load_tokenizer, model_dir

function main(args)
    _, opts = TileInfer.parse_args(args; flags = ("sampled", "pq"))
    target_name = get(opts, "model", "Qwen/Qwen3-4B")
    draft_name = get(opts, "draft", "none")
    dflash_name = get(opts, "dflash", target_name == "Qwen/Qwen3-4B" ? "z-lab/Qwen3-4B-DFlash-b16" : "none")
    ks = parse.(Int, split(get(opts, "k", "2,4,7"), ','))
    n = parse(Int, get(opts, "tokens", "128"))
    trials = parse(Int, get(opts, "trials", "5"))
    filter = Regex(get(opts, "filter", ""))
    verify = get(opts, "verify", "decode")
    verify in ("decode", "prefill") || throw(ArgumentError("--verify decode or prefill"))
    backend = verify == "decode" ? TileBackend() : Variant(verify = :prefill)

    dir = model_dir(target_name)
    len = parse(Int, get(opts, "context", string(PROMPT_TOKENS)))
    prompts = [p for p in chat_prompts(load_tokenizer(dir), TileInfer.load_config(dir).model_type; len)
               if occursin(filter, first(p))]
    # Room for the longest prompt, the reply and a last draft beyond it.
    max_tokens = 256 * cld(maximum(length ∘ last, prompts) + n + 64, 256)

    target = TileInfer.load_model(target_name; backend, max_len = max_tokens)
    draft = draft_name == "none" ? nothing : TileInfer.load_model(draft_name; backend, max_len = max_tokens)
    defaults, stop_ids = generation_defaults(dir, target.config)
    sampling = haskey(opts, "sampled") ? defaults : TileInfer.SamplingParams()
    runner = TileInfer.ModelRunner(target, TileInfer.PagedKVCache(target; max_tokens))
    # DFlash makes the target's steps keep hidden states: a runner of its own keeps the other
    # methods' steps as they are.
    dflash = dflash_name == "none" ? nothing : load_dflash(dflash_name, target; max_len = max_tokens)
    dflash_runner = dflash === nothing ? nothing : TileInfer.ModelRunner(target, TileInfer.PagedKVCache(target; max_tokens))

    methods = Tuple{String,Any,Any}[("plain", nothing, runner)]
    sample = haskey(opts, "pq")
    sample && sampling.temperature == 0 && throw(ArgumentError("--pq needs --sampled"))
    pq = sample ? " pq" : ""
    for k in ks
        push!(methods, ("ngram k=$k", NgramProposer(; k), runner))
        draft === nothing ||
            push!(methods, ("draft$pq k=$k", DraftProposer(draft; k, sample, max_tokens), runner))
    end
    if dflash !== nothing
        for k in unique!([[k for k in ks if k < dflash.block_size]; dflash.block_size - 1])
            push!(methods, ("dflash$pq k=$k", DFlashProposer(dflash; k, sample, max_tokens), dflash_runner))
        end
    end

    println("target $target_name, draft $draft_name, DFlash $dflash_name, ",
            sampling.temperature == 0 ? "greedy" : "sampled $sampling", sample ? ", sampled drafts" : "",
            ", $n tokens, best of $trials", verify == "decode" ? "" : ", prefill kernel for verify steps")
    @printf("%-12s %6s %-14s %8s %7s %11s %6s %6s %8s %11s\n", "prompt", "len", "method", "tok/s", "speedup",
            "prefill ms", "out", "steps", "accept", "tokens/step")
    for (name, ids) in prompts
        base = NaN
        for (method, proposer, runner) in methods
            cache = runner.cache
            function run()
                seq = TileInfer.new_sequence(cache)
                stats = SpecStats()
                first = Ref(0.0)
                on_token(_) = first[] == 0 && (first[] = time())
                start = time()
                out = TileInfer.generate!(runner, seq, ids; max_new_tokens = n, stop_ids, sampling, rng = Xoshiro(1),
                                          on_token, speculative = proposer, spec_stats = stats)
                rate = (length(out) - 1) / (time() - first[])
                TileInfer.release!(cache, seq)
                return rate, first[] - start, out, stats
            end
            run()                                    # compile and capture
            runs = [run() for _ in 1:trials]
            best = maximum(r[1] for r in runs)
            prefill = minimum(r[2] for r in runs)
            _, _, out, stats = run()
            proposer === nothing && (base = best)
            @printf("%-12s %6d %-14s %8.1f %6.2f× %11.1f %6d %6s %8s %11s\n", name, length(ids), method, best, best / base,
                    1e3 * prefill, length(out),
                    proposer === nothing ? "" : string(stats.steps),
                    proposer === nothing ? "" : @sprintf("%.2f", TileInfer.acceptance_rate(stats)),
                    proposer === nothing ? "" : @sprintf("%.2f", length(out) / stats.steps))
        end
    end
end

main(ARGS)
