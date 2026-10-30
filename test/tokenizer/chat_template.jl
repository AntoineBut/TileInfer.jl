# Compares against the models' own Jinja templates rendered by jinja2
# (scripts/gen_fixtures.py chat <model_dir> qwen3, and apertus-1.5). Needs no model download.

@testset "chat template" begin
    @testset "matches the $model_type Jinja template" for (model_type, name) in (("qwen3", "qwen3"),
                                                                                ("apertus1p5", "apertus-1.5"))
        for case in TileInfer.JSON.parsefile(fixture("chat", "$name.json"))["cases"]
            messages = [TileInfer.ChatMessage(m["role"], m["content"]) for m in case["messages"]]
            render() = TileInfer.chat_prompt(model_type, messages;
                                             add_generation_prompt = case["add_generation_prompt"],
                                             enable_thinking = case["enable_thinking"])
            if haskey(case, "prompt")
                @test render() == case["prompt"]
            else
                @test_throws ErrorException render()     # the template rejects it too
            end
        end
    end
    @test_throws ErrorException TileInfer.chat_prompt("gemma3", TileInfer.ChatMessage[])
    @test_throws ErrorException TileInfer.chat_prompt("qwen3", [TileInfer.ChatMessage("tool", "x")])
end
