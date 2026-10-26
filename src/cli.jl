# `tileinfer` command line. Installed as a Pkg app (see [apps] in Project.toml); also
# callable in-process as `TileInfer.cli(["ls"])`.

const USAGE = """
    usage: tileinfer <command> [arguments]

    commands:
      pull <repo> [--revision REV]   download a model's safetensors, configs and tokenizer
      ls                             list pulled models
      rm <repo> [--revision REV]     remove a pulled model (all revisions by default)
      run <model> <prompt> [options] generate a reply to `prompt` (a pulled repo id or a directory)

    run options:
      --max-tokens N        tokens to generate at most (default 512)
      --temperature T       0 for greedy (defaults come from the model's generation_config.json)
      --top-k K, --top-p P  sampling filters
      --seed S              random seed
      --thinking on|off     Qwen3 reasoning mode (default on)
      --raw                 send the prompt as-is, without the chat template
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

function cmd_run(positional, options)
    check_options(options, ("max-tokens", "temperature", "top-k", "top-p", "seed", "thinking", "raw"))
    length(positional) == 2 || throw(ArgumentError("run takes a model and a prompt"))
    model_name, prompt = positional
    thinking = get(options, "thinking", "on")
    thinking in ("on", "off") || throw(ArgumentError("--thinking expects on or off"))

    dir = model_dir(model_name)
    model = load_model(dir)
    tok = load_tokenizer(dir)
    defaults, stop_ids = generation_defaults(dir, model.config)
    sampling = SamplingParams(temperature = parse_option(Float64, options, "temperature", defaults.temperature),
                              top_k = parse_option(Int, options, "top-k", defaults.top_k),
                              top_p = parse_option(Float64, options, "top-p", defaults.top_p))
    rng = Random.Xoshiro(parse_option(Int, options, "seed", rand(UInt32)))
    text = haskey(options, "raw") ? prompt :
        chat_prompt(model.config.model_type, [ChatMessage("user", prompt)];
                    enable_thinking = thinking == "on")
    ids = encode(tok, text)

    stream = TextStream(tok)
    first_token = Ref(0.0)
    on_token(id) = (first_token[] == 0 && (first_token[] = time()); stream(id))
    start = time()
    generated = generate(model, ids; max_new_tokens = parse_option(Int, options, "max-tokens", 512),
                         sampling, stop_ids, rng, on_token)
    stop = time()
    println(); flush(stdout)
    # Time to first token includes JIT compilation in a fresh process; decode speed does not.
    decode_rate = length(generated) > 1 ? (length(generated) - 1) / (stop - first_token[]) : NaN
    println(stderr, "[", length(ids), " prompt tokens, ", length(generated), " generated; first token after ",
            round(first_token[] - start; digits = 2), " s, then ", round(decode_rate; digits = 1), " tok/s]")
end

const COMMANDS = Dict("pull" => cmd_pull, "ls" => cmd_ls, "rm" => cmd_rm, "run" => cmd_run)
const FLAGS = ("raw",)

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
