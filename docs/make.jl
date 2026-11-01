# Builds the documentation site: julia --project=docs docs/make.jl
# The output goes to docs/build/ (open docs/build/index.html). On GitHub Actions the same
# script also deploys it to the gh-pages branch (see .github/workflows/Documentation.yml).
# Building needs no GPU: TileInfer loads without one, and nothing here runs a kernel.

using Documenter
using TileInfer

makedocs(;
    modules = [TileInfer],
    sitename = "TileInfer.jl",
    authors = "Antoine Buttier",
    repo = Remotes.GitHub("AntoineBut", "TileInfer.jl"),
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://antoinebut.github.io/TileInfer.jl",
        edit_link = "main",
        size_threshold_warn = 200 * 1024,     # the performance log is a long page
        size_threshold = 400 * 1024,
    ),
    pages = [
        "Home" => "index.md",
        "Getting started" => "usage.md",
        "Design" => [
            "Architecture" => "architecture.md",
            "KV cache" => "kv_cache.md",
            "Attention kernels" => "attention.md",
            "Speculative decoding" => "speculative.md",
        ],
        "History and roadmap" => "roadmap.md",
        "Performance" => [
            "Performance log" => "performance.md",
            "Profiling" => "profiling.md",
            "cuTile design choices" => "cutile_design.md",
        ],
        "API" => [
            "Public API" => "api.md",
            "Internals" => "internals.md",
        ],
        "References" => "references.md",
    ],
    # Every name declared `public` in src/TileInfer.jl must appear in the manual.
    checkdocs = :public,
)

deploydocs(;
    repo = "github.com/AntoineBut/TileInfer.jl.git",
    devbranch = "main",
    push_preview = false,
)
