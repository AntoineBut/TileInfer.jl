# `tileinfer` command line. Installed as a Pkg app (see [apps] in Project.toml); also
# callable in-process as `TileInfer.cli(["ls"])`.

const USAGE = """
    usage: tileinfer <command> [arguments]

    commands:
      pull <repo> [--revision REV]   download a model's safetensors, configs and tokenizer
      ls                             list pulled models
      rm <repo> [--revision REV]     remove a pulled model (all revisions by default)
    """

# Splits `args` into positionals and `--name value` options.
function parse_args(args)
    positional = String[]
    options = Dict{String,String}()
    i = 1
    while i <= length(args)
        arg = args[i]
        if startswith(arg, "--")
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

const COMMANDS = Dict("pull" => cmd_pull, "ls" => cmd_ls, "rm" => cmd_rm)

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
        command(parse_args(args[2:end])...)
    catch e
        e isa ArgumentError || e isa ErrorException || rethrow()
        println(stderr, "error: ", sprint(showerror, e))
        return 1
    end
    return 0
end
