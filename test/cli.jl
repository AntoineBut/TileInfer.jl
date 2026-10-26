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
end
