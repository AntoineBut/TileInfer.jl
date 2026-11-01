# Speculative decoding

```@meta
CurrentModule = TileInfer
```

At batch size 1, a decode step reads every weight of the model to produce one token, and
the GPU's compute units mostly wait for memory. A step of 8 or 16 tokens reads the same
weights once and costs little more. Speculative decoding exploits that: a cheap **proposer**
guesses the next k tokens, and one **verify step** of the model on all of them keeps the
guesses up to the first wrong one, plus the model's own token at that point. Each step emits
1 to k + 1 tokens for about the price of one.

Code: `src/engine/speculative.jl` (verify steps, the generation loop, n-gram and draft-model
proposers) and `src/engine/dflash.jl` (the DFlash proposer).

```julia
generate(model, ids; sampling, stop_ids, speculative = NgramProposer(k = 4))
```

or `--speculate K`, `--draft MODEL` or `--dflash DRAFTER` on the command line.

## One speculative step

With `last` the latest token, sampled but not yet run through the model:

1. **Propose** drafts d₁ … dₖ to follow the context.
2. **Verify**: run `[last, d₁, …, dₖ]` through the model in one step of k + 1 tokens. The step
   computes the logits of every position and samples the model's own choice tᵢ after each
   prefix, all on the device.
3. **Accept** drafts while dᵢ = tᵢ. If the first a drafts match, the step emits d₁ … dₐ and
   then tₐ₊₁, the model's own token where the drafts went wrong (or after the last draft):
   the "bonus" token, which becomes the next `last`.
4. **Roll back**: [`truncate!`](@ref) drops the keys and values of the rejected drafts from
   the cache. Their pages go back to the pool and later steps overwrite their slots, so this
   costs nothing on the GPU.

Verify steps of each length (up to 16 tokens) are captured as CUDA graphs, like decode steps,
and their logits and samples stay on the device: the host reads only the k + 1 sampled tokens.

## Why the output does not change

**Greedy decoding.** tᵢ is the argmax after each prefix, exactly the token plain decoding
would pick, so the output is plain greedy decoding's. (In BFloat16 a verify step runs other
kernel shapes than a decode step, so the logits can differ in the last bits and the outputs
part at near-ties; in Float32 they are identical, which the tests check.)

**Sampling.** The verify step samples each position with the same sampler and the same
counter-based random numbers plain decoding would use for that token. For a deterministic
proposer, "sample t from p and accept d if t = d" is exactly rejection sampling (Leviathan et
al., Chen et al.) with a one-hot proposal: d is accepted with probability p(d), and otherwise
the token comes from p without d. So sampled speculative decoding produces the tokens of plain
sampled decoding, not only the same distribution.

**Sampled drafts.** A model proposer can also sample its drafts from its own distribution q
(`sample = true`, `--sample-drafts`). Then each drafted position applies the general rule on
the device ([`accept_token!`](@ref)): accept d with probability min(1, p(d)/q(d)), otherwise
sample from max(p − q, 0), renormalized. The tokens follow the model's distribution, but are
no longer plain decoding's. With Qwen3's recommended sampling the gain was mixed (+8 % on
one open prompt, −3 to −12 % on a code prompt), so greedy drafts stay the default.

## Proposers

| Proposer | How it drafts | Cost of k drafts | Best at |
|---|---|---|---|
| [`NgramProposer`](@ref) | prompt lookup: the tokens that followed the latest earlier occurrence of the last 1–3 tokens | free (host) | replies that copy the prompt (code edits) |
| [`DraftProposer`](@ref) | a smaller model with the same vocabulary decodes k tokens on its own cache | k small decode steps | open text, small k |
| [`DFlashProposer`](@ref) | a block-diffusion drafter drafts a whole block in one pass | one pass, whatever k | everything tested |

**Draft model.** Qwen3-0.6B drafting for Qwen3-4B. Each proposal first rewinds the draft's
sequence to what the target accepted, then decodes k − 1 steps as a chain of CUDA graphs:
step i samples draft i + 1 on the device straight into the inputs of step i + 1, so the host
enqueues the whole chain and waits once. A draft step costs about a fifth of a 4B decode
step, so each draft must be accepted with probability ~0.2 or more to pay, and k = 2 to 4 is
best.

**DFlash** ([Chen et al., 2026](https://arxiv.org/abs/2602.06036)), with z-lab's
`Qwen3-4B-DFlash-b16`. The target's steps keep the residual stream after five of its layers
([`ModelRunner`](@ref) does this when a DFlash proposer is attached). The drafter is five
Qwen3 layers: the keys and values of context positions come from a projection of those
target states and stay in the drafter's own cache, and a block of 16 positions (the next
token, then 15 mask tokens) attends to the whole context and to itself without a causal mask.
The target's own lm_head over the block gives 15 drafts at once. A drafter pass costs 3.2 ms
on Qwen3-4B, a 16-token verify 13.3 ms against 12.7 ms for one token: a step costs ~1.3 decode
steps and pays from 1.3 tokens per step on. The drafts are written on the device into the
verify step's inputs, so the host waits once per step.

Unlike chained drafters, whose acceptance falls along the chain, a longer DFlash block costs
nothing more to draft and little more to verify, so drafting the whole block (k = 15) wins on
short contexts, even though per-draft acceptance drops.

An EAGLE-3 proposer (one drafter layer on three target layers' states, chained one token at
a time) was built and removed: DFlash, which reads the same kind of states, beat it on every
prompt. Its results are kept in the [Performance log](performance.md).

## Results

Qwen3-4B target, 256-token chat replies (`bench/speculative.jl`), speedup in decode tokens
per second over plain decoding (78–80 tok/s), best k of each method:

| Prompt | n-gram | 0.6B draft | DFlash (k = 15) | DFlash tokens per step |
|---|---|---|---|---|
| rename two variables in a function | 3.92× | 3.36× | **8.50×** | 11.0 |
| add a docstring and comments | 1.55× | 1.80× | **3.00×** | 3.94 |
| explain hash-map collisions | 1.12× | 1.63× | **2.50×** | 3.24 |
| short story | 1.04× | 1.39× | **1.92×** | 2.51 |

With Qwen3's recommended sampling (temperature 0.6, top-k 20, top-p 0.95) the picture is the
same: DFlash gains 1.75× to 3.17× on the three prompts other than the rename. On a 2.2k-token prompt, the code edit still
reaches 5.9× with the whole block, while explaining it is faster at k = 7 (2.4×) than at
k = 15 (2.2×): at long contexts verify steps cost more, and the late drafts of open text are
rarely accepted. More numbers, and what each change gained, are in the
[Performance log](performance.md).

## Not built yet

- **Adaptive k**: choose k, or the proposer, per step from the measured acceptance rate and
  step costs, as DeepInfer's scheduler does. The long-prompt results above show what it
  would gain.
- **Tree verification**: verify several candidate continuations in one step with a tree
  attention mask, as EAGLE and SpecInfer do.
