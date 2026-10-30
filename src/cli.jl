# `tileinfer` command line. Installed as a Pkg app (see [apps] in Project.toml); also
# callable in-process as `TileInfer.cli(["ls"])`.

const USAGE = """
    usage: tileinfer <command> [arguments]

    commands:
      pull <repo> [--revision REV]   download a model's safetensors, configs and tokenizer
      ls                             list pulled models
      rm <repo> [--revision REV]     remove a pulled model (all revisions by default)
      run <model> <prompt> [options] generate a reply to `prompt` (a pulled repo id or a directory)
      chat <model> [options]         interactive multi-turn chat; /reset clears it, /exit quits

    run and chat options:
      --max-tokens N        tokens to generate per reply at most (default 512 for run, 2048 for chat)
      --max-context N       longest conversation in tokens; sizes the KV cache (default 4096)
      --temperature T       0 for greedy (defaults come from the model's generation_config.json)
      --top-k K, --top-p P  sampling filters
      --seed S              random seed
      --thinking on|off     Qwen3 reasoning mode (default on)
      --backend tile|ref    cuTile kernels (default) or the reference ops
      --raw                 run only: send the prompt as-is, without the chat template
      --speculate K         speculative decoding with up to K drafts per step (1-7), by
                            n-gram lookup in the conversation, or by --draft
      --draft <model>       a smaller model with the same vocabulary drafts (K = 4 by default)
      --eagle <drafter>     an EAGLE-3 drafter trained for the model drafts (K = 4 by default)
      --dflash <drafter>    a DFlash drafter trained for the model drafts a block in one pass
                            (K up to the block size - 1, all of it by default)
      --sample-drafts       with a drafter and a temperature > 0: the drafter samples its drafts
                            too, which are accepted by the p/q rule (default: greedy drafts)
    """

# Splits `args` into positionals and `--name value` options; names in `flags` take no
# value and are set to "true".
function parse_args(args; flags = ())
    positional = String[]
    options = Dict{String,String}()
    i = 1
    while i <= length(args)
        arg = args[i]
        if startswith(arg, "--") && arg[3:end] in flags
            options[arg[3:end]] = "true"
            i += 1
        elseif startswith(arg, "--")
            i < length(args) || throw(ArgumentError("option $arg needs a value"))
            options[arg[3:end]] = args[i+1]
            i += 2
        else
            push!(positional, arg)
            i += 1
        end
    end
    return positional, options
end

function check_options(options, allowed)
    for name in keys(options)
        name in allowed || throw(ArgumentError("unknown option --$name"))
    end
end

function format_bytes(n)
    units = ("B", "KB", "MB", "GB", "TB")
    i = n == 0 ? 0 : min(floor(Int, log(1000, n)), length(units) - 1)
    return i == 0 ? "$n B" : string(round(n / 1000^i; digits = 2), " ", units[i+1])
end

function cmd_pull(positional, options)
    check_options(options, ("revision",))
    length(positional) == 1 || throw(ArgumentError("pull takes exactly one repo id"))
    dir = pull(only(positional); revision = get(options, "revision", "main"))
    println("pulled to $dir")
end

function cmd_ls(positional, options)
    check_options(options, ())
    isempty(positional) || throw(ArgumentError("ls takes no arguments"))
    models = list_models()
    isempty(models) && return println("no models pulled")
    width = maximum(m -> length(m.repo_id), models)
    for m in models
        println(rpad(m.repo_id, width), "  ", rpad(m.revision, 10), "  ", format_bytes(m.bytes))
    end
end

function cmd_rm(positional, options)
    check_options(options, ("revision",))
    length(positional) == 1 || throw(ArgumentError("rm takes exactly one repo id"))
    remove_model(only(positional); revision = get(options, "revision", nothing))
end

function parse_option(::Type{T}, options, name, default) where {T}
    haskey(options, name) || return default
    value = tryparse(T, options[name])
    value === nothing && throw(ArgumentError("--$name expects a $(T == Int ? "integer" : "number"), got $(options[name])"))
    return value
end

const GENERATION_OPTIONS = ("max-tokens", "max-context", "temperature", "top-k", "top-p", "seed", "thinking",
                            "backend", "speculate", "draft", "eagle", "dflash", "sample-drafts")
const BACKENDS = Dict("tile" => TileBackend(), "ref" => RefBackend())

# Model, tokenizer and generation settings shared by `run` and `chat`.
function load_for_generation(model_name, options)
    thinking = get(options, "thinking", "on")
    thinking in ("on", "off") || throw(ArgumentError("--thinking expects on or off"))
    backend = get(BACKENDS, get(options, "backend", "tile"), nothing)
    backend === nothing && throw(ArgumentError("--backend expects tile or ref"))
    dir = model_dir(model_name)
    model = load_model(dir; backend, max_len = parse_option(Int, options, "max-context", 4096))
    tok = load_tokenizer(dir)
    defaults, stop_ids = generation_defaults(dir, model.config)
    sampling = SamplingParams(temperature = parse_option(Float64, options, "temperature", defaults.temperature),
                              top_k = parse_option(Int, options, "top-k", defaults.top_k),
                              top_p = parse_option(Float64, options, "top-p", defaults.top_p))
    rng = Random.Xoshiro(parse_option(Int, options, "seed", rand(UInt32)))
    return (; model, tok, sampling, stop_ids, rng, enable_thinking = thinking == "on",
            speculative = load_proposer(model, options))
end

# The proposer of --speculate / --draft / --eagle / --dflash, or nothing.
function load_proposer(model::DecoderLM, options)
    drafters = filter(o -> haskey(options, o), ["draft", "eagle", "dflash"])
    sample = haskey(options, "sample-drafts")
    sample && isempty(drafters) && throw(ArgumentError("--sample-drafts needs --draft, --eagle or --dflash"))
    haskey(options, "speculate") || !isempty(drafters) || return nothing
    length(drafters) > 1 && throw(ArgumentError("use only one of --draft, --eagle and --dflash"))
    if haskey(options, "dflash")
        d = load_dflash(options["dflash"], model)
        k = parse_option(Int, options, "speculate", d.block_size - 1)
        1 <= k < d.block_size || throw(ArgumentError("--speculate expects 1 to $(d.block_size - 1) drafts with DFlash, got $k"))
        return DFlashProposer(d; k, sample)
    end
    k = parse_option(Int, options, "speculate", 4)
    1 <= k < MAX_GRAPH_TOKENS || throw(ArgumentError("--speculate expects 1 to $(MAX_GRAPH_TOKENS - 1) drafts, got $k"))
    haskey(options, "eagle") && return Eagle3Proposer(load_eagle3(options["eagle"], model); k, sample)
    haskey(options, "draft") || return NgramProposer(; k)
    draft = load_model(model_dir(options["draft"]); T = eltype(model), backend = model.backend,
                       max_len = max_length(model))
    draft.config.vocab_size == model.config.vocab_size ||
        throw(ArgumentError("the draft model's vocabulary ($(draft.config.vocab_size)) differs from the model's ($(model.config.vocab_size))"))
    return DraftProposer(draft; k, sample)
end

# Streams tokens to `io` and records when the first two arrived.
mutable struct TimedStream
    stream::TextStream
    start::Float64
    first_token::Float64
    second_token::Float64
end

TimedStream(tok::BPETokenizer, io::IO) = TimedStream(TextStream(tok, io), time(), 0.0, 0.0)

function (t::TimedStream)(id)
    now = time()
    if t.first_token == 0
        t.first_token = now
    elseif t.second_token == 0
        t.second_token = now
    end
    t.stream(id)
end

# In a fresh process, time to first token includes compiling the prefill kernels, and the
# first decode step compiles the single-token ones. Decode speed is measured from the
# second token on so that it excludes both.
function timing_summary(t::TimedStream, generated::Integer)
    stop = time()
    rate = generated > 2 ? (generated - 2) / (stop - t.second_token) : NaN
    ttft = t.first_token == 0 ? stop - t.start : t.first_token - t.start
    return string("first token after ", round(ttft; digits = 2), " s, then ", round(rate; digits = 1), " tok/s")
end

spec_summary(s::SpecStats, generated::Integer) = s.steps == 0 ? "" :
    string("; ", round(generated / s.steps; digits = 2), " tokens per step, ",
           round(100 * acceptance_rate(s); digits = 1), " % of drafts accepted")

function cmd_run(positional, options)
    check_options(options, (GENERATION_OPTIONS..., "raw"))
    length(positional) == 2 || throw(ArgumentError("run takes a model and a prompt"))
    model_name, prompt = positional
    g = load_for_generation(model_name, options)
    text = haskey(options, "raw") ? prompt :
        chat_prompt(g.model.config.model_type, [ChatMessage("user", prompt)];
                    enable_thinking = g.enable_thinking)
    ids = encode(g.tok, text)
    stream = TimedStream(g.tok, stdout)
    spec = SpecStats()
    generated = generate(g.model, ids; max_new_tokens = parse_option(Int, options, "max-tokens", 512),
                         g.sampling, g.stop_ids, g.rng, on_token = stream, g.speculative, spec_stats = spec)
    println(); flush(stdout)
    println(stderr, "[", length(ids), " prompt tokens, ", length(generated), " generated; ",
            timing_summary(stream, length(generated)), spec_summary(spec, length(generated)), "]")
end

"""
    chat_loop(session, input, output; max_new_tokens)

Read user messages line by line from `input` and stream replies to `output` until EOF or
`/exit`. `/reset` starts a new conversation. Statistics go to stderr.
"""
function chat_loop(session::ChatSession, input::IO, output::IO; max_new_tokens::Integer)
    while true
        print(output, "> "); flush(output)
        line = readline(input)
        text = strip(line)
        ((eof(input) && isempty(line)) || text == "/exit") && break
        isempty(text) && continue
        if text == "/reset"
            reset!(session)
            println(output, "[conversation cleared]")
            continue
        end
        stream = TimedStream(session.tokenizer, output)
        try
            _, stats = reply!(session, text; max_new_tokens, on_token = stream)
            println(output); flush(output)
            println(stderr, "[", stats.reused, " tokens reused, ", stats.prefilled, " prefilled, ",
                    stats.generated, " generated; ", timing_summary(stream, stats.generated),
                    spec_summary(stats.spec, stats.generated), "]")
        catch e
            e isa OutOfPagesError || e isa ArgumentError || rethrow()
            println(output)
            println(stderr, "error: ", sprint(showerror, e), "; /reset to start over")
        end
    end
end

function cmd_chat(positional, options)
    check_options(options, GENERATION_OPTIONS)
    length(positional) == 1 || throw(ArgumentError("chat takes exactly one model"))
    g = load_for_generation(only(positional), options)
    session = ChatSession(g.model, g.tok; g.sampling, g.stop_ids, g.enable_thinking, g.rng, g.speculative)
    chat_loop(session, stdin, stdout; max_new_tokens = parse_option(Int, options, "max-tokens", 2048))
end

const COMMANDS = Dict("pull" => cmd_pull, "ls" => cmd_ls, "rm" => cmd_rm, "run" => cmd_run,
                      "chat" => cmd_chat)
const FLAGS = ("raw", "sample-drafts")

"Run the CLI on `args`; returns the process exit code."
function cli(args::Vector{String})
    if isempty(args) || args[1] in ("-h", "--help", "help")
        print(USAGE)
        return isempty(args) ? 1 : 0
    end
    command = get(COMMANDS, args[1], nothing)
    if command === nothing
        println(stderr, "unknown command: $(args[1])\n")
        print(stderr, USAGE)
        return 1
    end
    try
        command(parse_args(args[2:end]; flags = FLAGS)...)
    catch e
        e isa ArgumentError || e isa ErrorException || rethrow()
        println(stderr, "error: ", sprint(showerror, e))
        return 1
    end
    return 0
end
