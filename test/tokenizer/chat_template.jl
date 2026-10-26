# Compares against Qwen3's own Jinja template rendered by jinja2
# (scripts/gen_fixtures.py chat <model_dir> qwen3). Needs no model download.

@testset "chat template" begin
    cases = TileInfer.JSON.parsefile(fixture("chat", "qwen3.json"))["cases"]
    @testset "matches Qwen3 Jinja template" begin
        for case in cases
            messages = [TileInfer.ChatMessage(m["role"], m["content"]) for m in case["messages"]]
            @test TileInfer.chat_prompt("qwen3", messages;
                                        add_generation_prompt = case["add_generation_prompt"],
                                        enable_thinking = case["enable_thinking"]) == case["prompt"]
        end
    end
    @test_throws ErrorException TileInfer.chat_prompt("gemma3", TileInfer.ChatMessage[])
    @test_throws ErrorException TileInfer.chat_prompt("qwen3", [TileInfer.ChatMessage("tool", "x")])
end
