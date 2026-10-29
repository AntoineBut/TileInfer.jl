# Probes of cuTile codegen: pairs (or more) of kernels that compute the same thing written
# differently, measured side by side (see docs/CUTILE_QUIRKS.md). For each variant the
# harness reports the GPU time of its kernels (CUPTI trace: minimum and mean over the second
# half of ~0.3 s of eager launches, after a warm-up; CUPTI times come in steps of ~0.24 µs,
# which the mean averages out), the block size, registers and shared memory of the launch,
# local memory (spills, from the driver), and the code at both compiler stages:
#
#   Tile IR  emitted by cuTile.jl (`ct.code_tiled`), with SSA and kernel names normalized,
#            so variants can be compared textually: "same IR" means cuTile.jl produced the
#            same program and any difference comes from tileiras (or noise);
#   SASS     the cubin tileiras produced, disassembled with nvdisasm: instruction count and
#            a few telling opcodes (barriers, shuffles, shared and local memory traffic).
#
# Files land in bench/results/quirks/<probe>/<variant>.{tileir,sass}. Probes other than
# the TileInfer audit use only CUDA and cuTile, so a kernel pair can be pasted into an
# upstream issue as is.

using CUDA
using BFloat16s: BFloat16
using cuTile: cuTile
import cuTile as ct
using Printf: @printf, @sprintf

const OUT = joinpath(@__DIR__, "..", "results", "quirks")
const NVDISASM = let a = joinpath(dirname(cuTile.tileiras_path()), "nvdisasm")
    isfile(a) ? a : nothing
end

"A way of computing the probe's result: `run()` launches its kernel(s) once, eagerly."
struct Variant
    name::String
    run::Function
end

"""
    Probe(name, question, variants; output = nothing, rtol = 0, bytes = nothing)

`output()` returns the result (host array) after a variant ran; every variant must match
the first one within `rtol` (0: bit-identical). `bytes[i]`, the minimum memory traffic of
variant i, adds its bandwidth (from the mean time) to the report.
"""
struct Probe
    name::String
    question::String
    variants::Vector{Variant}
    output::Union{Nothing,Function}
    rtol::Float64
    bytes::Union{Nothing,Vector{Float64}}
end

const PROBES = Probe[]

probe!(name, question, variants::Pair...; output = nothing, rtol = 0.0, bytes = nothing) =
    push!(PROBES, Probe(name, question, [Variant(n, f) for (n, f) in variants], output, rtol,
                        bytes === nothing ? nothing : Float64.(bytes)))

const L2 = CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_L2_CACHE_SIZE)
const ROTATIONS = Base.RefValue{Int}[]

"""
    rotating(make, bytes) -> () -> args

Enough copies of `make()` (each `bytes` large) to total > 2× L2, returned in turn on each
call, so that data read through them comes from DRAM. Every variant's correctness run
starts from the first copy.
"""
function rotating(make, bytes)
    sets = [make() for _ in 1:clamp(cld(2L2, bytes), 1, 16)]
    i = Ref(0)
    push!(ROTATIONS, i)
    return () -> (i[] = mod1(i[] + 1, length(sets)); sets[i[]])
end

struct KernelStats
    name::String
    min_us::Float64
    mean_us::Float64
    block::Int
    registers::Int
    shared_bytes::Int
end

struct CodeStats
    name::String            # kernel function name, as the trace reports it
    tileir::String          # normalized Tile IR
    sass::String            # the SASS instructions, without addresses
    tileir_ops::Int
    local_bytes::Int        # per thread (spills), from the driver: CUPTI reports 0
    sass_instructions::Int
    sass_counts::Dict{String,Int}
end

const SASS_OPS = ("BAR", "SHFL", "LDS", "STS", "LDL", "STL", "LDG", "STG", "BRA")

# Run `f` once with cuTile's compile hook set; returns the (kernel function, argument types)
# of every kernel compiled meanwhile.
function compiled_kernels(f)
    seen = Tuple{Any,Any}[]
    cuTile.compile_hook[] = (k, tt) -> ((k, tt) in seen || push!(seen, (k, tt)); nothing)
    try
        f()
        CUDA.synchronize()
    finally
        cuTile.compile_hook[] = nothing
    end
    isempty(seen) && error("no cuTile kernel compiled: give each variant kernels of its own")
    return seen
end

# SSA value names and the kernel's own name differ between otherwise identical programs:
# number values in order of use and drop the entry name.
function normalize_ir(text)
    names = Dict{String,String}()
    text = replace(text, r"entry @[A-Za-z0-9_]+" => "entry @kernel")
    return replace(text, r"%[A-Za-z0-9_]+" => s -> get!(() -> "%v$(length(names))", names, s))
end

function code_stats(f, tt, dir, label)
    io = IOBuffer()
    ct.code_tiled(io, f, Tuple(tt.parameters))
    ir = String(take!(io))
    write(joinpath(dir, "$label.tileir"), ir)
    ops = count(l -> occursin(r"^\s+(%\S+(, %\S+)* = )?[a-z_]+", l), split(ir, '\n'))
    sass_n, counts, body = 0, Dict{String,Int}(), IOBuffer()
    if NVDISASM !== nothing
        argtypes, consts = cuTile.unwrap_argtypes(f, tt)
        key = cuTile.TileCacheKey(cuTile.default_sm_arch(), cuTile.bytecode_version(), nothing, nothing, nothing, nothing)
        _, _, _, res = cuTile.compile(f, argtypes, consts, key)
        cubin = joinpath(dir, "$label.cubin")
        write(cubin, res.cuda_bin)
        sass = read(`$NVDISASM -c $cubin`, String)
        write(joinpath(dir, "$label.sass"), sass)
        for l in split(sass, '\n')
            m = match(r"/\*[0-9a-f]{4,}\*/\s+(@!?U?P\S+\s+)?([A-Z0-9]+)", l)
            m === nothing && continue
            sass_n += 1
            println(body, strip(replace(l, r"/\*[0-9a-f]{4,}\*/" => "", string(nameof(f)) => "kernel")))
            op = m.captures[2]
            op in SASS_OPS && (counts[op] = get(counts, op, 0) + 1)
        end
    end
    local_bytes = CUDA.attributes(cuTile.cufunction(f, tt).fun)[CUDA.FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES]
    return CodeStats(string(nameof(f)), normalize_ir(ir), String(take!(body)), ops, local_bytes, sass_n, counts)
end

# GPU time per kernel from a CUPTI trace of ~`seconds` of launches, after as long a warm-up.
# Starting the profiler idles the GPU and drops its clocks, so only the second half counts.
function time_kernels(f; seconds = 0.3)
    t0 = time()
    while time() - t0 < seconds
        f()
    end
    CUDA.synchronize()
    r = CUDA.@profile trace = true begin
        t0 = time()
        while time() - t0 < seconds
            f()
        end
        CUDA.synchronize()
    end
    d = r.device
    rows = [i for i in eachindex(d.name) if d.grid[i] !== missing]
    stats = KernelStats[]
    for name in unique(first(split(d.name[i], '(')) for i in rows)
        idx = [i for i in rows if first(split(d.name[i], '(')) == name]
        idx = idx[end÷2+1:end]
        us = sort([(d.stop[i] - d.start[i]) * 1e6 for i in idx])
        i = first(idx)
        push!(stats, KernelStats(String(name), us[1], sum(us) / length(us), prod(Int.((d.block[i].x, d.block[i].y, d.block[i].z))),
                                 Int(d.registers[i]),
                                 Int(coalesce(d.shared_mem[i].static, 0)) + Int(coalesce(d.shared_mem[i].dynamic, 0))))
    end
    return stats
end

function run_probe(p::Probe)
    dir = mkpath(joinpath(OUT, p.name))
    println("\n== $(p.name): $(p.question)")
    @printf("  %-34s %9s %9s %6s %5s %6s %6s %8s %6s  %s\n", "variant / kernel", "min µs", "mean µs", "block", "regs",
            "local", "smem", "IR ops", "SASS", "vs first variant")
    ref_out, ref_code = nothing, nothing
    for (j, v) in enumerate(p.variants)
        foreach(i -> i[] = 0, ROTATIONS)
        kernels = compiled_kernels(v.run)
        if p.output !== nothing
            out = p.output()
            if ref_out === nothing
                ref_out = out
            elseif !(p.rtol == 0 ? out == ref_out : isapprox(out, ref_out; rtol = p.rtol))
                @printf("  %-34s RESULT DIFFERS from %s\n", v.name, p.variants[1].name)
            end
        end
        codes = [code_stats(k, tt, dir, length(kernels) == 1 ? v.name : "$(v.name)_$i") for (i, (k, tt)) in enumerate(kernels)]
        same = ref_code === nothing ? "" :
               string(all(c -> c.tileir in first.(ref_code), codes) ? "same IR" : "IR differs", ", ",
                      all(c -> c.sass in last.(ref_code), codes) ? "same SASS" : "SASS differs")
        j == 1 && (ref_code = [(c.tileir, c.sass) for c in codes])
        stats = time_kernels(v.run)
        total, mean = sum(s -> s.min_us, stats), sum(s -> s.mean_us, stats)
        bw = p.bytes === nothing ? "" : @sprintf("%.0f GB/s", p.bytes[j] / mean / 1e3)
        @printf("  %-34s %9.2f %9.2f %6s %5s %6s %6s %8d %6d  %s\n", v.name, total, mean, "", "", "", "",
                sum(c -> c.tileir_ops, codes), sum(c -> c.sass_instructions, codes), join(filter(!isempty, [bw, same]), ", "))
        for s in stats
            c = something(findfirst(c -> c.name == s.name, codes), 0)
            c = c == 0 ? nothing : codes[c]
            ops = c === nothing ? "" : join(("$k $(c.sass_counts[k])" for k in SASS_OPS if haskey(c.sass_counts, k)), ", ")
            @printf("    %-32s %9.2f %9.2f %6d %5d %6d %6d %8s %6s  %s\n", s.name, s.min_us, s.mean_us, s.block, s.registers,
                    c === nothing ? -1 : c.local_bytes, s.shared_bytes, c === nothing ? "" : c.tileir_ops,
                    c === nothing ? "" : c.sass_instructions, ops)
        end
    end
end

# Probes of what compiles: each variant is run once, and its outcome (or the first line of
# its error) is reported.
struct CompileProbe
    name::String
    question::String
    variants::Vector{Variant}
end

const COMPILE_PROBES = CompileProbe[]

compile_probe!(name, question, variants::Pair...) =
    push!(COMPILE_PROBES, CompileProbe(name, question, [Variant(n, f) for (n, f) in variants]))

function run_probe(p::CompileProbe)
    println("\n== $(p.name): $(p.question)")
    for v in p.variants
        outcome = try
            v.run()
            CUDA.synchronize()
            "compiles and runs"
        catch e
            msg = sprint(showerror, e)
            m = match(r"Reason: (.*)", msg)
            "FAILS: " * (m === nothing ? first(split(msg, '\n')) : m.captures[1])
        end
        @printf("  %-40s %s\n", v.name, outcome)
    end
end
