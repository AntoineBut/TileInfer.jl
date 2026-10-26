# Local line-coverage report for src/ (not uploaded anywhere).
#
#   julia --project=test test/coverage.jl [--uncovered] [test filters...]
#
# Runs test/runtests.jl (with the same file filters) in a subprocess that instruments
# only src/, prints per-file coverage, writes coverage/lcov.info (shown in the editor by
# e.g. VS Code's Coverage Gutters extension) and removes the .cov files. `--uncovered`
# also lists the line ranges that never ran. Device code (cuTile kernels) is compiled by
# GPUCompiler and is not instrumented: its lines show as unexecuted only if the host code
# around them is.

using CoverageTools
using Printf: @printf

const ROOT = dirname(@__DIR__)
const SRC = joinpath(ROOT, "src")
const OUT = joinpath(ROOT, "coverage")

# "1-3, 7, 9-12" for the uncovered lines of a file.
function uncovered_ranges(fc::FileCoverage)
    lines = [i for (i, c) in enumerate(fc.coverage) if c !== nothing && c == 0]
    ranges = String[]
    i = 1
    while i <= length(lines)
        j = i
        while j < length(lines) && lines[j+1] == lines[j] + 1
            j += 1
        end
        push!(ranges, i == j ? string(lines[i]) : "$(lines[i])-$(lines[j])")
        i = j + 1
    end
    return join(ranges, ", ")
end

function main(args)
    show_uncovered = "--uncovered" in args
    filters = filter(!=("--uncovered"), args)

    clean_folder(SRC)                                  # stale .cov files from earlier runs
    julia = Base.julia_cmd()
    cmd = `$julia --project=$(joinpath(ROOT, "test")) --code-coverage=@$SRC
           $(joinpath(ROOT, "test", "runtests.jl")) $filters`
    tests_passed = success(pipeline(cmd; stdout, stderr))

    coverage = process_folder(SRC)
    clean_folder(SRC)
    mkpath(OUT)
    LCOV.writefile(joinpath(OUT, "lcov.info"), coverage)

    println()
    @printf("%-36s %8s %8s %7s\n", "file", "covered", "lines", "%")
    for fc in sort(coverage; by = fc -> fc.filename)
        covered, total = get_summary(fc)
        total == 0 && continue
        @printf("%-36s %8d %8d %6.1f%%\n", relpath(fc.filename, ROOT), covered, total, 100 * covered / total)
        show_uncovered && covered < total && println("    uncovered: ", uncovered_ranges(fc))
    end
    covered, total = get_summary(coverage)
    @printf("%-36s %8d %8d %6.1f%%\n", "total", covered, total, 100 * covered / total)
    println("lcov: ", joinpath(OUT, "lcov.info"))
    tests_passed || (println(stderr, "tests failed; coverage covers the tests that ran"); exit(1))
end

main(ARGS)
