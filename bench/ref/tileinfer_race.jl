# TileInfer side of the end-to-end drag race; same workload and output as vllm_race.py.
#
#   julia --project=bench bench/ref/tileinfer_race.jl [model] [prompt_len] [gen] [backend] [trials] [weights] [kv]
#
# `weights` and `kv` are bf16 (default) or int8 (`load_model(...; quantize = :int8)`,
# `PagedKVCache(...; quantize = :int8)`).
#
# Prefill = latency of generate(max_new_tokens = 1); decode tok/s = (gen - 1) /
# (latency(gen) - latency(1)). Each latency is the median over `trials` after a 1 s warm-up
# (timings are heavy-tailed); every trial is recorded, with the minimum-based numbers
# alongside. EOS is ignored.

using TileInfer, CUDA, JSON

median(v) = (s = sort(v); n = length(s); isodd(n) ? s[(n + 1) ÷ 2] : (s[n ÷ 2] + s[n ÷ 2 + 1]) / 2)

model_name = get(ARGS, 1, "Qwen/Qwen3-0.6B")
prompt_len = parse(Int, get(ARGS, 2, "512"))
gen        = parse(Int, get(ARGS, 3, "256"))
backend    = get(ARGS, 4, "tile") == "ref" ? TileInfer.RefBackend() : TileInfer.TileBackend()
trials     = parse(Int, get(ARGS, 5, "10"))
weights    = get(ARGS, 6, "bf16")
kv         = get(ARGS, 7, "bf16")
quant(s)   = s == "int8" ? :int8 : (s == "bf16" ? nothing : error("unknown dtype $s"))

model = TileInfer.load_model(model_name; backend, max_len = prompt_len + gen + 16, quantize = quant(weights))
ids = [1000 + (7919 * i) % 140000 for i in 0:prompt_len-1]   # same ids as vllm_race.py

# One cache and runner for all trials, so decode graphs are captured once during warm-up
# (as vLLM does at startup) rather than inside every timed run.
cache = TileInfer.PagedKVCache(model; max_tokens = prompt_len + gen, quantize = quant(kv))
runner = TileInfer.ModelRunner(model, cache)

function latency(n)
    function run()
        seq = TileInfer.new_sequence(cache)
        toks = TileInfer.generate!(runner, seq, ids; max_new_tokens = n, stop_ids = Int[])
        TileInfer.release!(cache, seq)
        return toks
    end
    # Warm-up: compile, then keep the GPU busy for a second so its clock has ramped up
    # (from idle, a 10 ms request runs at 900-1600 MHz and takes 2-3x longer).
    t0 = time_ns()
    run()
    while time_ns() - t0 < 1e9
        run()
    end
    CUDA.synchronize()
    times, toks = Float64[], Int[]
    for _ in 1:trials
        t = time_ns()
        toks = run()
        CUDA.synchronize()
        push!(times, (time_ns() - t) / 1e9)
    end
    @assert length(toks) == n
    return times, toks
end

times1, _ = latency(1)
timesn, toks = latency(gen)
metrics(t1, tn, suffix = "") = Dict(
    "prefill_ms" * suffix => 1e3 * t1, "prefill_tok_s" * suffix => prompt_len / t1,
    "decode_ms_per_tok" * suffix => 1e3 * (tn - t1) / (gen - 1), "decode_tok_s" * suffix => (gen - 1) / (tn - t1),
    "e2e_s" * suffix => tn)
res = merge(Dict(
    "engine" => "tileinfer", "backend" => string(typeof(backend)), "graphs" => runner.graphs, "model" => model_name,
    "dtype" => "bfloat16", "weights" => weights, "kv" => kv, "prompt_len" => prompt_len, "gen" => gen, "trials" => trials, "statistic" => "median",
    "trials_prefill_s" => times1, "trials_e2e_s" => timesn, "first_tokens" => toks[1:min(16, end)]),
    metrics(median(times1), median(timesn)), metrics(minimum(times1), minimum(timesn), "_min"))
JSON.print(stdout, res, 2)
println()
