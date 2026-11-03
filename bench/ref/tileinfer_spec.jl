# TileInfer side of the speculative-decoding comparison; same workload and output as
# vllm_spec.py. Greedy replies of up to `gen` tokens (stopping at EOS) to the prompts of
# spec_prompts.json, with one configuration:
#
#   julia --project=bench bench/ref/tileinfer_spec.jl model method k weights kv [gen] [trials] [prompts] [--profile]
#
# method: plain, ngram (NgramProposer, n-grams of 3 down to 1) or dflash (z-lab drafter
# for Qwen3-4B); weights and kv: bf16 or int8. Per prompt, after a 1 s warm-up (compiles,
# captures the graphs, ramps the GPU clock up): median latency of the reply and of a
# 1-token request over `trials`; decode tok/s = (n - 1) / (latency(n) - latency(1)), so
# the drafter's prefill, wherever an engine does it, counts once in either term.
#
# --profile (under nsys, see profile.sh): after the same warm-up, one reply per prompt between
# cuProfilerStart/Stop, each in an NVTX range `request NAME`, and no timed trials.

using TileInfer, CUDA, JSON
using TileInfer: DFlashProposer, NgramProposer, SpecStats, load_dflash, NVTX

median(v) = (s = sort(v); n = length(s); isodd(n) ? s[(n + 1) ÷ 2] : (s[n ÷ 2] + s[n ÷ 2 + 1]) / 2)
quant(s) = s == "int8" ? :int8 : (s == "bf16" ? nothing : error("unknown dtype $s"))

profile = "--profile" in ARGS
args = filter(!=("--profile"), ARGS)
model_name, method = args[1], args[2]
k = parse(Int, args[3])
weights, kv = args[4], args[5]
gen = parse(Int, get(args, 6, "128"))
trials = profile ? 1 : parse(Int, get(args, 7, "5"))
prompts_file = get(args, 8, "spec_prompts.json")          # relative to bench/ref/

spec = JSON.parsefile(joinpath(@__DIR__, prompts_file))
stop_ids = Int.(spec["stop_ids"])
prompts = [(p["name"], Int.(p["ids"])) for p in spec["prompts"]]
max_tokens = 256 * cld(maximum(length(last(p)) for p in prompts) + gen + 64, 256)

model = TileInfer.load_model(model_name; backend = TileBackend(), max_len = max_tokens, quantize = quant(weights))
cache = TileInfer.PagedKVCache(model; max_tokens, quantize = quant(kv))
runner = TileInfer.ModelRunner(model, cache)
proposer = method == "plain" ? nothing :
           method == "ngram" ? NgramProposer(; k) :
           method == "dflash" ? DFlashProposer(load_dflash("z-lab/Qwen3-4B-DFlash-b16", model; max_len = max_tokens); k, max_tokens) :
           error("unknown method $method")

function run(ids, n)
    seq = TileInfer.new_sequence(cache)
    stats = SpecStats()
    t = time_ns()
    out = TileInfer.generate!(runner, seq, ids; max_new_tokens = n, stop_ids, speculative = proposer, spec_stats = stats)
    CUDA.synchronize()
    dt = (time_ns() - t) / 1e9
    TileInfer.release!(cache, seq)
    return dt, out, stats
end

function warm_up(ids)
    t0 = time_ns()
    run(ids, gen); run(ids, 1)
    while time_ns() - t0 < 1e9
        run(ids, 1)
    end
end

# One traced reply per prompt; the profile's own output comes from nsys_export.py.
function profile_replies()
    foreach(warm_up ∘ last, prompts)
    results = []
    CUDA.Profile.start()
    for (name, ids) in prompts
        dt, out, stats = NVTX.@range "request $name" run(ids, gen)
        push!(results, Dict("prompt" => name, "prompt_len" => length(ids), "out" => length(out), "e2e_s" => dt,
                            "steps" => stats.steps, "first_tokens" => out[1:min(16, end)]))
    end
    CUDA.Profile.stop()
    return results
end

results = profile ? profile_replies() : []
for (name, ids) in (profile ? () : prompts)
    warm_up(ids)
    tn = Float64[]; t1 = Float64[]
    out, stats = Int[], SpecStats()
    for _ in 1:trials
        dt, out, stats = run(ids, gen)
        push!(tn, dt)
        push!(t1, run(ids, 1)[1])
    end
    n = length(out)
    r = Dict("prompt" => name, "prompt_len" => length(ids), "out" => n,
             "e2e_s" => median(tn), "prefill_ms" => 1e3 * median(t1),
             "decode_tok_s" => (n - 1) / (median(tn) - median(t1)),
             "trials_e2e_s" => tn, "trials_prefill_s" => t1, "first_tokens" => out[1:min(16, end)])
    if proposer !== nothing
        r["steps"] = stats.steps
        r["tokens_per_step"] = n / stats.steps
    end
    push!(results, r)
    println(stderr, rpad(name, 13), " out ", n, "  decode ", round(r["decode_tok_s"], digits = 1), " tok/s")
end
JSON.print(stdout, Dict("engine" => "tileinfer", "profile" => profile, "prompts" => prompts_file, "model" => model_name, "method" => method, "k" => k,
                        "weights" => weights, "kv" => kv, "gen" => gen, "trials" => trials, "results" => results), 2)
println()
