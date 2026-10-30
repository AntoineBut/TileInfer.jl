# EAGLE-3 drafting (Li et al., https://arxiv.org/abs/2503.01840): a one-layer drafter trained
# against one target model, which reads the target's own hidden states instead of only its
# tokens. Layout and conventions follow vLLM's `llama_eagle3.py` and its EAGLE proposer.
#
# The target keeps the residual stream after three of its layers (by default 2, L/2 and
# L - 3 of L, counted from 1) for every token it runs; `fc` maps the three, stacked, to one
# hidden state g(p) per position p. The drafter at position p reads the token at p + 1 and a
# hidden state:
#
#     x = [rmsnorm(embed(token_{p+1}), input_norm); rmsnorm(hin, hidden_norm)]     (2 hidden)
#     r = hin + attention(x) (one Llama attention layer, RoPE at p, a KV cache of its own)
#     r = r + mlp(rmsnorm(r, mlp_norm))
#     logits = lm_head(rmsnorm(r, norm))    over a reduced vocabulary; d2t maps it back
#
# and predicts the token at p + 2. `hin` is g(p) where the target has run position p, and
# otherwise the drafter's own r of the previous position. So a proposal first runs the
# positions the target has just verified (the "first pass", with g) and then k - 1 positions
# on its own outputs (the chain). Draft-cache positions written by the chain are run again
# with g once the target has verified them.

"""
    Eagle3Drafter

An EAGLE-3 drafter on the GPU: its layer, the target's embedding table (shared), its RoPE
table, the target layers it reads (`aux_layers`, 1-based) and the map from its vocabulary to
the target's. Load with [`load_eagle3`](@ref).
"""
struct Eagle3Drafter{T,M<:AbstractMatrix{T},V<:AbstractVector{T},R<:RotaryTable{T},B<:Backend}
    config::ModelConfig              # vocab_size is the draft vocabulary
    aux_layers::Vector{Int}
    fc::M                            # (aux layers × target hidden, hidden)
    embed::M                         # the target's (hidden, target vocab)
    input_norm::V                    # for the token embedding
    hidden_norm::V                   # for hin
    wqkv::M                          # (2 hidden, (heads + 2 kv_heads) head_dim)
    wo::M
    mlp_norm::V
    w_gate_up::M
    w_down::M
    norm::V
    lm_head::M                       # (hidden, draft vocab)
    d2t::CuVector{Int32}             # draft id d → target id d + d2t[d + 1]
    rope::R
    backend::B
end

max_length(d::Eagle3Drafter) = length(d.rope)

"""
    load_eagle3(drafter, target::DecoderLM; max_len = max_length(target)) -> Eagle3Drafter

Load an EAGLE-3 drafter for `target` (a pulled repo id or a directory), in the format of
SpecForge and AngelSlim checkpoints (`midlayer.*`, `fc`, `d2t`).
"""
function load_eagle3(name::AbstractString, target::DecoderLM{T}; max_len::Integer = max_length(target)) where {T}
    dir = model_dir(name)
    raw = JSON.parsefile(joinpath(dir, "config.json"))
    cfg = parse_config(raw)
    cfg.num_layers == 1 || error("EAGLE-3 drafters with $(cfg.num_layers) layers are not supported")
    tcfg = target.config
    cfg.vocab_size == tcfg.vocab_size || error("the drafter's vocabulary differs from the target's")
    vocab = something(get(raw, "draft_vocab_size", nothing), cfg.vocab_size)
    cfg = ModelConfig(; (f => getfield(cfg, f) for f in fieldnames(ModelConfig) if f != :vocab_size)..., vocab_size = vocab)
    aux = eagle3_layers(raw, tcfg.num_layers)

    c = Checkpoint(dir)
    p = haskey(c, "midlayer.input_layernorm.weight") ? "midlayer." : "layers.0."
    H, I, D, TH = cfg.hidden_size, cfg.intermediate_size, cfg.head_dim, tcfg.hidden_size
    H == TH || error("drafter hidden size $H differs from the target's $TH")
    up(n, dims) = check_size(upload(T, c, n), dims, n)
    qkv = [p * "self_attn.$(x)_proj.weight" for x in ("q", "k", "v")]
    d2t = haskey(c, "d2t") ? CuVector{Int32}(Int32.(file_layout(c, "d2t"))) : CUDA.zeros(Int32, vocab)
    return Eagle3Drafter(cfg, aux, up("fc.weight", (length(aux) * TH, H)), target.weights.embed,
                         up(p * "input_layernorm.weight", (H,)), up(p * "hidden_norm.weight", (H,)),
                         check_size(upload_fused(T, c, qkv), (2H, (cfg.num_heads + 2cfg.num_kv_heads) * D), "wqkv"),
                         up(p * "self_attn.o_proj.weight", (cfg.num_heads * D, H)),
                         up(p * "post_attention_layernorm.weight", (H,)),
                         check_size(upload_fused(T, c, [p * "mlp.gate_proj.weight", p * "mlp.up_proj.weight"]),
                                    (H, 2I), "w_gate_up"),
                         up(p * "mlp.down_proj.weight", (I, H)), up("norm.weight", (H,)),
                         up("lm_head.weight", (H, vocab)), d2t, RotaryTable{T}(cfg, max_len), target.backend)
end

# The target layers whose outputs the drafter reads, counted from 1 (the residual stream after
# that many layers). vLLM's `eagle_aux_hidden_state_layer_ids` already count that way;
# SpecForge's `target_layer_ids` number layer outputs from 0.
function eagle3_layers(raw::AbstractDict, num_layers::Integer)
    ids = get(get(raw, "eagle_config", Dict()), "eagle_aux_hidden_state_layer_ids", nothing)
    ids === nothing || return collect(Int, ids)
    ids = get(raw, "target_layer_ids", nothing)
    ids === nothing || return collect(Int, ids) .+ 1
    return [2, num_layers ÷ 2, num_layers - 3]
end

# Buffers of one drafter step of n positions: the step's inputs and activations (b.x holds
# hin, then r), the concatenated layer input, and the target's stacked hidden states for a
# first pass.
struct EagleStep{T}
    b::StepBuffers{T}
    xcat::CuMatrix{T}                # (2 hidden, n)
    aux::CuMatrix{T}                 # (aux layers × hidden, n)
end

EagleStep(d::Eagle3Drafter{T}, cache::PagedKVCache{T}, n::Integer) where {T} =
    EagleStep{T}(StepBuffers{T}(d.config, d.backend, cache, n), CuMatrix{T}(undef, 2d.config.hidden_size, n),
                 CuMatrix{T}(undef, size(d.fc, 1), n))

# The drafter's layer on a prepared step whose b.x holds hin; leaves r in b.x and the
# logits of the last position in b.logits.
function eagle_layer!(d::Eagle3Drafter{T}, cache::PagedKVCache{T}, s::EagleStep{T}) where {T}
    cfg, be, b = d.config, d.backend, s.b
    H, n, eps = cfg.hidden_size, num_tokens(b), cfg.rms_norm_eps
    embed!(be, b.h, d.embed, b.ids)
    rmsnorm!(be, b.proj, b.h, d.input_norm, eps)
    view(s.xcat, 1:H, :) .= b.proj
    rmsnorm!(be, b.proj, b.x, d.hidden_norm, eps)
    view(s.xcat, H+1:2H, :) .= b.proj
    linear!(be, b.qkv, d.wqkv, s.xcat)
    attention_inputs!(be, b.q, cache.k[1], cache.v[1], b.qkv, nothing, nothing, d.rope, b.positions, b.slots, eps)
    attention!(be, b.attn, b.attn_workspace, b.q, cache.k[1], cache.v[1], b.block_table, b.positions,
               1 / sqrt(cfg.head_dim))
    linear!(be, b.proj, d.wo, reshape(b.attn, :, n))
    add_rmsnorm!(be, b.x, b.h, b.proj, d.mlp_norm, eps)
    linear_swiglu!(be, b.act, b.gate_up, d.w_gate_up, b.h)
    linear!(be, b.proj, d.w_down, b.act)
    add_rmsnorm!(be, b.x, b.h, b.proj, d.norm, eps)
    return linear!(be, b.logits, d.lm_head, view(b.h, :, n:n))
end

"""
    Eagle3Proposer(drafter; k = 4, sample = false, max_tokens = max_length(drafter),
                   graphs = supports_graphs(drafter.backend))

Proposes the next `k` tokens that the EAGLE-3 `drafter` decodes greedily (see
[`load_eagle3`](@ref)), or samples with the target's parameters when `sample` and the target
samples (see [`DraftProposer`](@ref)), on a KV cache of its own for up to `max_tokens`
positions. Reads the
hidden states of the target's latest step: generation enables them on the target's runner
([`enable_aux!`](@ref)).
"""
mutable struct Eagle3Proposer{D<:Eagle3Drafter,T}
    const k::Int
    const sample::Bool
    const dists::DraftDists                      # of the drafts, when sampled
    const drafter::D
    const cache::PagedKVCache{T}
    const seq::Sequence                          # tokens[p + 1]: the token read at position p
    const graphs::Bool
    const first::Dict{Int,Any}                   # (EagleStep, graph) of first passes by length
    const chain::Vector{EagleStep{T}}            # step i runs draft i, for i in 1:k-1
    const chain_graphs::Vector{Union{Nothing,CUDA.CuGraphExec}}
    const sink::CuVector{Int32}                  # ids written by the last step
    const mirror::CuVector{Int32}                # host-mapped: the drafts
    const mirror_host::Vector{Int32}
    const sampler::TokenSampler                  # over the draft vocabulary
    target::Any                                  # the target's ModelRunner
end

function Eagle3Proposer(d::Eagle3Drafter{T}; k::Integer = 4, sample::Bool = false, max_tokens::Integer = max_length(d),
                        graphs::Bool = supports_graphs(d.backend)) where {T}
    1 <= k < MAX_GRAPH_TOKENS || throw(ArgumentError("k must be in 1:$(MAX_GRAPH_TOKENS - 1)"))
    cache = PagedKVCache{T}(d.config, cld(min(max_tokens, max_length(d)), DEFAULT_PAGE_SIZE))
    mirror = CuArray{Int32,1,CUDA.HostMemory}(undef, k)
    return Eagle3Proposer{typeof(d),T}(k, sample, DraftDists(k), d, cache, new_sequence(cache), graphs, Dict{Int,Any}(),
                                       [EagleStep(d, cache, 1) for _ in 1:k-1],
                                       fill!(Vector{Union{Nothing,CUDA.CuGraphExec}}(undef, k - 1), nothing),
                                       CuVector{Int32}(undef, 1), mirror, unsafe_wrap(Array, mirror),
                                       TokenSampler(d.config.vocab_size), nothing)
end

function attach!(p::Eagle3Proposer, r::ModelRunner)
    size(p.drafter.fc, 1) == length(p.drafter.aux_layers) * r.model.config.hidden_size ||
        throw(ArgumentError("the EAGLE drafter does not fit this target model"))
    enable_aux!(r, p.drafter.aux_layers)
    reset_draft_sampler!(p.sampler, p.sample, r)
    p.target = r
    return p
end

draft_dists(p::Eagle3Proposer, r::ModelRunner) = samples_for(p.sample, r) ? p.dists : nothing

# Sample the draft of step i (i = 0: the first pass) from step s into the next chain step's
# inputs: its token, and r of the last position as its hin.
function sample_eagle!(p::Eagle3Proposer, i, s::EagleStep)
    be, b = p.drafter.backend, s.b
    last = i < p.k - 1 ? p.chain[i+1] : nothing
    ids = last === nothing ? p.sink : last.b.ids
    mirror = view(p.mirror, i+1:i+1)
    if p.sample                                  # q too (see DraftProposer), mapped like the token
        q = column(p.dists, i + 1)
        sample_token!(be, ids, mirror, b.logits, p.sampler, q)
        map_token!(be, ids, mirror, p.drafter.d2t, q)
    else
        sample_token!(be, ids, mirror, b.logits, p.sampler)
        map_token!(be, ids, mirror, p.drafter.d2t)
    end
    if last !== nothing
        H, n = size(b.x)
        copyto!(last.b.x, 1, b.x, (n - 1) * H + 1, H)
    end
end

function propose(p::Eagle3Proposer, context::AbstractVector{<:Integer}, k::Integer)
    k = min(k, p.k)
    k >= 1 || return Int[]
    r, d, seq = p.target::ModelRunner, p.drafter, p.seq
    L, start = length(context), r.last_start
    tb = r.last_step::StepBuffers
    # The target ran positions start:L-2 last (L - 1 is the newest token, not yet run). Keep
    # the draft positions before that which still read the context's tokens; positions from
    # `start` on are run again with the target's hidden states.
    keep = min(common_prefix(seq.tokens, view(context, 2:L)), start)
    truncate!(p.cache, seq, keep)
    if keep < start
        # The target ran positions the drafter never saw (a prefix reused without drafting):
        # their hidden states are gone, so their draft keys and values are left stale. Drafts
        # get worse; the tokens stay correct.
        reserve!(p.cache.blocks, seq, start - keep)
        append!(seq.tokens, view(context, keep+2:start+1))
    end
    n = L - 1 - start
    ids = view(context, start+2:L)
    NVTX.@range "draft" begin
        s, graph = n > MAX_GRAPH_TOKENS ? (EagleStep(d, p.cache, n), nothing) :
                   get!(() -> (EagleStep(d, p.cache, n), nothing), p.first, n)
        prepare_step!(s.b, d, p.cache, seq, ids)
        copyto!(s.aux, 1, tb.aux, 1, length(s.aux))      # the target's states of positions start:L-2
        first_pass() = (linear!(d.backend, s.b.x, d.fc, s.aux); eagle_layer!(d, p.cache, s); sample_eagle!(p, 0, s))
        if n > MAX_GRAPH_TOKENS
            first_pass()
        else
            p.first[n] = (s, run_captured!(first_pass, p, graph))
        end
        append!(seq.tokens, ids)
        for i in 1:k-1
            c = p.chain[i]
            prepare_step!(c.b, d, p.cache, seq, nothing)
            push!(seq.tokens, -1)                         # draft i, read below
            chain_step() = (eagle_layer!(d, p.cache, c); sample_eagle!(p, i, c))
            p.chain_graphs[i] = run_captured!(chain_step, p, p.chain_graphs[i])
        end
        synchronize()
    end
    drafts = Int.(p.mirror_host[1:k])
    seq.tokens[end-k+2:end] .= drafts[1:k-1]
    return drafts
end
