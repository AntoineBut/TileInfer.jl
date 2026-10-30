# Speculative decoding on real chat prompts: plain decoding (pipelined) against n-gram lookup,
# a draft model, an EAGLE-3 drafter and a DFlash drafter, for several draft lengths k
# (DFlash also with its whole block). Acceptance depends on the text, so unlike
# bench/run.jl this generates replies to fixed prompts, from one that copies its input to
# open-ended ones.
#
#   julia --project=bench bench/speculative.jl [--model REPO] [--draft REPO|none] [--eagle REPO|none]
#       [--dflash REPO|none] [--k 1,2,4,7] [--tokens N] [--trials N] [--sampled] [--filter REGEX]
#
# Defaults: Qwen/Qwen3-4B drafted by Qwen/Qwen3-0.6B, AngelSlim/Qwen3-4B_eagle3 and
# z-lab/Qwen3-4B-DFlash-b16 (the last two for Qwen3-4B; none otherwise), k = 2, 4, 7,
# 256 new tokens, greedy
# (--sampled: the model's recommended sampling, same seed for every method). Decode tok/s
# runs from the first new token to the last, so it excludes the target's prefill but includes
# the draft model's (it prefills during the first proposal). Each case is the best of
# `trials` runs after a warm-up run that compiles and captures every step.

using CUDA
using Printf: @printf, @sprintf
using Random: Xoshiro
using TileInfer
using TileInfer: ChatMessage, DecoderLM, DFlashProposer, DraftProposer, Eagle3Proposer, NgramProposer, SpecStats,
                 chat_prompt, encode, generation_defaults, load_dflash, load_eagle3, load_tokenizer, model_dir

const CODE = """
function rmsnorm!(y, x, w, eps)
    for j in axes(x, 2)
        s = zero(eltype(x))
        for i in axes(x, 1)
            s += x[i, j]^2
        end
        r = 1 / sqrt(s / size(x, 1) + eps)
        for i in axes(x, 1)
            y[i, j] = x[i, j] * r * w[i]
        end
    end
    return y
end
"""

const PROMPTS = [
    "rename" => "Rename the variable `s` to `acc` and `r` to `scale` in this Julia function. Reply with the full code only.\n\n" * CODE,
    "document" => "Add a docstring and a comment on each loop to this Julia function. Reply with the full code only.\n\n" * CODE,
    "explain" => "Explain how a hash map handles collisions, with an example.",
    "story" => "Write a short story about a lighthouse keeper who finds a strange machine on the beach.",
]

const MAX_TOKENS = 2048

function main(args)
    _, opts = TileInfer.parse_args(args; flags = ("sampled",))
    target_name = get(opts, "model", "Qwen/Qwen3-4B")
    draft_name = get(opts, "draft", "Qwen/Qwen3-0.6B")
    eagle_name = get(opts, "eagle", target_name == "Qwen/Qwen3-4B" ? "AngelSlim/Qwen3-4B_eagle3" : "none")
    dflash_name = get(opts, "dflash", target_name == "Qwen/Qwen3-4B" ? "z-lab/Qwen3-4B-DFlash-b16" : "none")
    ks = parse.(Int, split(get(opts, "k", "2,4,7"), ','))
    n = parse(Int, get(opts, "tokens", "256"))
    trials = parse(Int, get(opts, "trials", "3"))
    filter = Regex(get(opts, "filter", ""))

    target = TileInfer.load_model(target_name; backend = TileBackend(), max_len = MAX_TOKENS)
    draft = draft_name == "none" ? nothing : TileInfer.load_model(draft_name; backend = TileBackend(), max_len = MAX_TOKENS)
    dir = model_dir(target_name)
    tok = load_tokenizer(dir)
    defaults, stop_ids = generation_defaults(dir, target.config)
    sampling = haskey(opts, "sampled") ? defaults : TileInfer.SamplingParams()
    runner = TileInfer.ModelRunner(target, TileInfer.PagedKVCache(target; max_tokens = MAX_TOKENS))
    # EAGLE and DFlash make the target's steps keep hidden states (of different layers): a
    # runner of their own each keeps the others' steps as they are.
    aux_runner() = TileInfer.ModelRunner(target, TileInfer.PagedKVCache(target; max_tokens = MAX_TOKENS))
    eagle = eagle_name == "none" ? nothing : load_eagle3(eagle_name, target; max_len = MAX_TOKENS)
    eagle_runner = eagle === nothing ? nothing : aux_runner()
    dflash = dflash_name == "none" ? nothing : load_dflash(dflash_name, target; max_len = MAX_TOKENS)
    dflash_runner = dflash === nothing ? nothing : aux_runner()

    methods = Tuple{String,Any,Any}[("plain", nothing, runner)]
    for k in ks
        push!(methods, ("ngram k=$k", NgramProposer(; k), runner))
        draft === nothing || push!(methods, ("draft k=$k", DraftProposer(draft; k, max_tokens = MAX_TOKENS), runner))
        eagle === nothing || push!(methods, ("eagle k=$k", Eagle3Proposer(eagle; k, max_tokens = MAX_TOKENS), eagle_runner))
    end
    if dflash !== nothing
        for k in unique!([[k for k in ks if k < dflash.block_size]; dflash.block_size - 1])
            push!(methods, ("dflash k=$k", DFlashProposer(dflash; k, max_tokens = MAX_TOKENS), dflash_runner))
        end
    end

    println("target $target_name, draft $draft_name, EAGLE-3 $eagle_name, DFlash $dflash_name, ",
            sampling.temperature == 0 ? "greedy" : "sampled $sampling",
            ", $n tokens, best of $trials")
    @printf("%-9s %-11s %8s %7s %6s %6s %8s %11s\n", "prompt", "method", "tok/s", "speedup", "out", "steps",
            "accept", "tokens/step")
    for (name, text) in PROMPTS
        occursin(filter, name) || continue
        ids = encode(tok, chat_prompt(target.config.model_type, [ChatMessage("user", text)]; enable_thinking = false))
        base = NaN
        for (method, proposer, runner) in methods
            cache = runner.cache
            function run()
                seq = TileInfer.new_sequence(cache)
                stats = SpecStats()
                first = Ref(0.0)
                on_token(_) = first[] == 0 && (first[] = time())
                out = TileInfer.generate!(runner, seq, ids; max_new_tokens = n, stop_ids, sampling, rng = Xoshiro(1),
                                          on_token, speculative = proposer, spec_stats = stats)
                rate = (length(out) - 1) / (time() - first[])
                TileInfer.release!(cache, seq)
                return rate, out, stats
            end
            run()                                    # compile and capture
            best = maximum(first(run()) for _ in 1:trials)
            _, out, stats = run()
            proposer === nothing && (base = best)
            @printf("%-9s %-11s %8.1f %6.2f× %6d %6s %8s %11s\n", name, method, best, best / base, length(out),
                    proposer === nothing ? "" : string(stats.steps),
                    proposer === nothing ? "" : @sprintf("%.2f", TileInfer.acceptance_rate(stats)),
                    proposer === nothing ? "" : @sprintf("%.2f", length(out) / stats.steps))
        end
    end
end

main(ARGS)
