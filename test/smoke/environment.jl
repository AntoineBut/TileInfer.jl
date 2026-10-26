# Toolchain and device requirements for Tile IR on this machine.

@testset "environment" begin
    @test CUDA.functional()
    # Tile IR supports Ampere/Ada (sm_80..sm_89) from bytecode v13.2 onwards.
    @test capability(device()) >= v"8.0"
    @test CUDA.driver_version() >= v"13.0"
end
