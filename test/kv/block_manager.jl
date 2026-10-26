@testset "block manager" begin
    b = TileInfer.BlockManager(6, 4)                  # 6 pages of 4 positions
    s = TileInfer.Sequence(Int32[], Int[])
    TileInfer.reserve!(b, s, 5)
    @test s.block_table == [1, 2] && TileInfer.num_free_pages(b) == 4
    append!(s.tokens, 1:5)
    TileInfer.reserve!(b, s, 3)                       # positions 5..7 still fit in page 2
    @test s.block_table == [1, 2]
    TileInfer.reserve!(b, s, 4)
    @test s.block_table == [1, 2, 3]

    # Position p is offset p % 4 of page block_table[p ÷ 4 + 1]: slot (page - 1) * 4 + offset + 1.
    other = TileInfer.Sequence(Int32[], Int[])
    TileInfer.reserve!(b, other, 1)                   # page 4
    @test TileInfer.slots(b, s, 3, 3) == Int32[4, 5, 6]
    @test TileInfer.slots(b, other, 0, 1) == Int32[13]

    TileInfer.truncate!(b, s, 4)                      # keeps exactly one page
    @test s.block_table == [1] && s.tokens == 1:4
    @test TileInfer.num_free_pages(b) == 4
    TileInfer.reserve!(b, s, 1)
    @test s.block_table == [1, 2]                     # freed pages come back in order

    @test_throws TileInfer.OutOfPagesError TileInfer.reserve!(b, s, 100)
    @test s.block_table == [1, 2]                     # failed reserve changes nothing
    @test_throws ArgumentError TileInfer.truncate!(b, s, 10)
    TileInfer.release!(b, s); TileInfer.release!(b, other)
    @test TileInfer.num_free_pages(b) == 6 && isempty(s.block_table)
end
