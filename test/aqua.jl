# Package hygiene with Aqua.jl: method ambiguities, unbound type parameters, undefined
# exports, stale dependencies, compat bounds and type piracy.
#
# - The stale-dependency check only sees packages nothing else loads: a dependency that
#   CUDA.jl also pulls in (as PrecompileTools was) passes even when TileInfer never uses it.
# - The persistent-task check is off. It precompiles TileInfer in a scratch environment
#   whose manifest Aqua writes from each dependency's Project.toml, and CUDA_Runtime_jll
#   0.24.1 imports CUDA_Compiler_jll without listing it there (only the registry does), so
#   precompilation fails on machines with a driver. TileInfer starts no tasks when loaded
#   (it has no `__init__`), which is what that check guards against.

using Aqua

Aqua.test_all(TileInfer; persistent_tasks = false)
