# Runs the cuTile codegen probes of bench/quirks/probes (see harness.jl).
#
#   julia --project=bench bench/quirks/run.jl [NAME_FILTER...]

include("harness.jl")

for file in sort(readdir(joinpath(@__DIR__, "probes"); join = true))
    endswith(file, ".jl") && include(file)
end

selected(p) = isempty(ARGS) || any(a -> occursin(a, p.name), ARGS)
cap = CUDA.capability(CUDA.device())
println("cuTile $(pkgversion(cuTile)), tileiras $(cuTile.tileiras_version()), $(CUDA.name(CUDA.device())) (sm_$(cap.major)$(cap.minor))")
for p in [filter(selected, COMPILE_PROBES); filter(selected, PROBES)]
    try
        run_probe(p)
    catch e
        println("  ERROR: ", first(split(sprint(showerror, e), '\n')))
    end
end
