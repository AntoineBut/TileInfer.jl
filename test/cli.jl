# Runs `cli` in-process with output captured; returns (exit code, stdout, stderr).
function run_cli(args...)
    out, err = Pipe(), Pipe()
    code = redirect_stdout(out) do
        redirect_stderr(() -> TileInfer.cli(collect(String, args)), err)
    end
    close(out.in); close(err.in)
    return code, read(out, String), read(err, String)
end

@testset "cli" begin
    @test TileInfer.parse_args(["Qwen/Qwen3-0.6B", "--revision", "abc"]) ==
          (["Qwen/Qwen3-0.6B"], Dict("revision" => "abc"))
    @test_throws ArgumentError TileInfer.parse_args(["--revision"])
    @test TileInfer.parse_args(["m", "--raw", "p", "--seed", "3"]; flags = ("raw",)) ==
          (["m", "p"], Dict("raw" => "true", "seed" => "3"))
    @test TileInfer.format_bytes(0) == "0 B"
    @test TileInfer.format_bytes(1_503_000_000) == "1.5 GB"

    code, out, _ = run_cli("--help")
    @test code == 0 && occursin("usage: tileinfer", out)
    @test run_cli()[1] == 1
    code, _, err = run_cli("bogus")
    @test code == 1 && occursin("unknown command", err)
    code, _, err = run_cli("pull")
    @test code == 1 && occursin("exactly one repo id", err)
    code, _, err = run_cli("ls", "--color", "red")
    @test code == 1 && occursin("unknown option --color", err)
    @test run_cli("ls")[1] == 0

    code, _, err = run_cli("run", "only-a-model")
    @test code == 1 && occursin("a model and a prompt", err)
    code, _, err = run_cli("chat")
    @test code == 1 && occursin("exactly one model", err)
    code, _, err = run_cli("run", "nobody/not-pulled", "hi")
    @test code == 1 && occursin("is not pulled", err)
    if test_model_dir() !== nothing
        code, _, err = run_cli("run", TEST_MODEL, "hi", "--backend", "gpu")
        @test code == 1 && occursin("--backend expects tile or ref", err)
        code, _, err = run_cli("run", TEST_MODEL, "hi", "--top-p", "lots")
        @test code == 1 && occursin("--top-p expects a number", err)
        # Greedy raw completion; the same continuation as the float32 HF fixture would be
        # too strict in bfloat16, so only check the text and the stats line.
        code, out, err = run_cli("run", TEST_MODEL, "The capital of France is", "--raw",
                                 "--temperature", "0", "--max-tokens", "4")
        @test code == 0
        @test occursin("Paris", out)
        @test occursin("4 generated", err)
        # Speculative decoding, by n-gram lookup and with the model drafting for itself.
        specs = [("--speculate", "3"), ("--draft", TEST_MODEL)]
        try TileInfer.model_dir(EAGLE_TEST_HEAD); push!(specs, ("--eagle", EAGLE_TEST_HEAD)) catch end
        for spec in specs
            code, out, err = run_cli("run", TEST_MODEL, "The capital of France is", "--raw",
                                     "--temperature", "0", "--max-tokens", "12", "--max-context", "256", spec...)
            @test code == 0
            @test occursin("Paris", out)
            @test occursin("tokens per step", err)
        end
        code, _, err = run_cli("run", TEST_MODEL, "hi", "--speculate", "8")
        @test code == 1 && occursin("--speculate expects 1 to 7", err)
        code, _, err = run_cli("run", TEST_MODEL, "hi", "--draft", TEST_MODEL, "--dflash", TEST_MODEL)
        @test code == 1 && occursin("only one of --draft, --eagle and --dflash", err)
    end
end
