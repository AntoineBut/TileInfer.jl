# Byte-level BPE tokenizer read from a Hugging Face `tokenizer.json`, the scheme used by
# GPT-2, Llama 3, Qwen, Mistral-Nemo and Apertus. Encoding a string:
#
#   1. split out added/special tokens (e.g. "<|im_start|>"), which map directly to ids
#   2. normalize the remaining text (NFC for Qwen, string replacements for Apertus 1.5,
#      nothing for Llama 3), then split out the added tokens marked `normalized`
#   3. pre-tokenize: split into "words" with the model's regex
#   4. map each word's UTF-8 bytes to printable characters (GPT-2 byte-level alphabet)
#   5. apply BPE merges inside each word, lowest merge rank first
#   6. look up the merged symbols in the vocabulary
#
# Token ids are 0-based, as in the vocabulary file and in the model's embedding rows.

using JSON: JSON
using Unicode: Unicode

# GPT-2's reversible byte -> printable character table. Printable Latin-1 bytes map to
# themselves; the other 68 bytes (control characters, space, …) map to U+0100 onwards.
const BYTE_TO_CHAR, CHAR_TO_BYTE = let
    printable = [0x21:0x7e; 0xa1:0xac; 0xae:0xff]
    to_char = Vector{Char}(undef, 256)
    n = 0
    for b in 0x00:0xff
        if b in printable
            to_char[b+1] = Char(b)
        else
            to_char[b+1] = Char(256 + n)
            n += 1
        end
    end
    to_char, Dict(c => UInt8(b - 1) for (b, c) in enumerate(to_char))
end

byte_level(word::AbstractString) = String(map(b -> BYTE_TO_CHAR[b+1], codeunits(word)))

# A normalizer step: NFC, or replacing every match of a string or regex.
const NormalizerStep = Union{Symbol,Pair{String,String},Pair{Regex,String}}

apply_normalizer(text::AbstractString, steps::Vector{NormalizerStep}) =
    foldl((t, step) -> step === :nfc ? Unicode.normalize(t, :NFC) : replace(t, step), steps; init = String(text))

"""
Added tokens to split out of a text, matched leftmost-longest as Hugging Face `tokenizers`
does. Indexed by first character and length rather than compiled into one regex: Apertus
1.5 has 136k of them (image and audio codes).
"""
struct AddedTokens
    ids::Dict{String,Int}
    lengths::Dict{Char,Vector{Int}}          # code-unit lengths of the tokens starting with a char, longest first
end

function AddedTokens(ids::Dict{String,Int})
    lengths = Dict{Char,Vector{Int}}()
    for s in keys(ids)
        push!(get!(lengths, first(s), Int[]), ncodeunits(s))
    end
    foreach(v -> sort!(unique!(v); rev = true), values(lengths))
    return AddedTokens(ids, lengths)
end

# Calls f(piece, nothing) for each run of text between added tokens and f(token, id) for
# each added token, in order.
function foreach_piece(f, added::AddedTokens, text::AbstractString)
    start, i, n = 1, 1, ncodeunits(text)
    while i <= n
        len = 0
        for l in get(added.lengths, text[i], ())
            last = i + l - 1
            # `last` must end a character; SubString takes the index of that character.
            if last <= n && thisind(text, last + 1) == last + 1 &&
               haskey(added.ids, SubString(text, i, prevind(text, last + 1)))
                len = l
                break
            end
        end
        if len == 0
            i = nextind(text, i)
            continue
        end
        i > start && f(SubString(text, start, prevind(text, i)), nothing)
        token = SubString(text, i, prevind(text, i + len))
        f(token, added.ids[token])
        start = i = i + len
    end
    start <= n && f(SubString(text, start), nothing)
    return nothing
end

"""
    BPETokenizer

Byte-level BPE tokenizer. Load with [`load_tokenizer`](@ref); use [`encode`](@ref) and
[`decode`](@ref).
"""
struct BPETokenizer
    vocab::Dict{String,Int}                  # byte-level token string -> id
    tokens::Vector{String}                   # id + 1 -> token string (byte-level, or added content)
    merge_ranks::Dict{Tuple{String,String},Int}
    pattern::Regex                           # pre-tokenizer word split
    normalizer::Vector{NormalizerStep}       # applied before splitting into words
    ignore_merges::Bool                      # a word already in the vocab is one token (Llama 3)
    added::Dict{String,Int}                  # added token content -> id
    raw_added::AddedTokens                   # matched in the text as given
    normalized_added::AddedTokens            # matched after normalization (`"normalized": true`)
    special::Set{Int}                        # ids skipped by `decode(...; skip_special = true)`
    prefix::Vector{Int}                      # ids the post-processor prepends (e.g. BOS)
    cache::Dict{String,Vector{Int}}          # word -> ids
end

"Load `tokenizer.json` from a model directory."
load_tokenizer(dir::AbstractString) = BPETokenizer(JSON.parsefile(joinpath(dir, "tokenizer.json")))

function BPETokenizer(json::AbstractDict)
    model = json["model"]
    model["type"] == "BPE" || error("unsupported tokenizer model \"$(model["type"])\"")
    get(model, "byte_fallback", false) && error("byte-fallback BPE (sentencepiece-style) is not supported")

    vocab = Dict{String,Int}(String(k) => Int(v) for (k, v) in model["vocab"])
    merge_ranks = Dict{Tuple{String,String},Int}()
    for (rank, m) in enumerate(model["merges"])
        # Older files store "a b", newer ones ["a", "b"].
        a, b = m isa AbstractString ? split(m, ' '; limit = 2) : m
        merge_ranks[(String(a), String(b))] = rank
    end

    normalizer = parse_normalizer(json["normalizer"])
    added, raw, normalized = Dict{String,Int}(), Dict{String,Int}(), Dict{String,Int}()
    special = Set{Int}()
    for t in get(json, "added_tokens", [])
        any(k -> get(t, k, false), ("lstrip", "rstrip", "single_word")) &&
            error("added token $(t["content"]) strips or matches single words, which is not supported")
        added[t["content"]] = t["id"]
        # A normalized token is looked up in normalized text, so by its normalized content.
        get(t, "normalized", false) ? (normalized[apply_normalizer(t["content"], normalizer)] = t["id"]) :
                                      (raw[t["content"]] = t["id"])
        t["special"] && push!(special, t["id"])
    end

    ntokens = max(maximum(values(vocab)), isempty(added) ? 0 : maximum(values(added))) + 1
    tokens = fill("", ntokens)
    for (s, id) in vocab
        tokens[id+1] = s
    end
    for (s, id) in added
        tokens[id+1] = s
    end

    return BPETokenizer(vocab, tokens, merge_ranks, parse_pre_tokenizer(json["pre_tokenizer"]),
                        normalizer, get(model, "ignore_merges", false), added, AddedTokens(raw),
                        AddedTokens(normalized), special, parse_post_processor(json["post_processor"]),
                        Dict{String,Vector{Int}}())
end

function parse_normalizer(n)
    n === nothing && return NormalizerStep[]
    steps = n["type"] == "Sequence" ? n["normalizers"] : [n]
    return map(steps) do s
        s["type"] == "NFC" && return :nfc
        s["type"] == "Replace" || error("unsupported normalizer $(s["type"])")
        p = s["pattern"]
        return (haskey(p, "String") ? String(p["String"]) : Regex(p["Regex"])) => String(s["content"])
    end |> Vector{NormalizerStep}
end

# Accepts the pre-tokenizers used by byte-level BPE models: a regex Split followed by
# ByteLevel without its own regex, or a bare ByteLevel with the GPT-2 regex.
const GPT2_PATTERN = raw"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"

function parse_pre_tokenizer(p)
    steps = p["type"] == "Sequence" ? p["pretokenizers"] : [p]
    pattern = nothing
    for s in steps
        if s["type"] == "Split"
            pattern === nothing || error("more than one Split pre-tokenizer")
            s["behavior"] == "Isolated" && !s["invert"] ||
                error("unsupported Split behavior $(s["behavior"]) (invert = $(s["invert"]))")
            pattern = Regex(s["pattern"]["Regex"])
        elseif s["type"] == "ByteLevel"
            get(s, "add_prefix_space", false) && error("ByteLevel add_prefix_space is not supported")
            if get(s, "use_regex", true)
                pattern === nothing || error("ByteLevel regex after a Split pre-tokenizer")
                pattern = Regex(GPT2_PATTERN)
            end
        else
            error("unsupported pre-tokenizer $(s["type"])")
        end
    end
    pattern === nothing && error("pre-tokenizer has no word-splitting regex")
    return pattern
end

# Only a fixed prefix of special tokens is supported (e.g. Llama 3's BOS); Qwen's
# ByteLevel post-processor adds nothing.
function parse_post_processor(p)
    p === nothing && return Int[]
    steps = p["type"] == "Sequence" ? p["processors"] : [p]
    prefix = Int[]
    for s in steps
        s["type"] == "ByteLevel" && continue
        s["type"] == "TemplateProcessing" || error("unsupported post-processor $(s["type"])")
        for piece in s["single"]
            if haskey(piece, "SpecialToken")
                push!(prefix, s["special_tokens"][piece["SpecialToken"]["id"]]["ids"][1])
            elseif haskey(piece, "Sequence")
                break
            end
        end
    end
    return prefix
end

"""
    encode(tok, text; add_special_tokens = true) -> Vector{Int}

Token ids (0-based) of `text`. Added tokens written in the text, such as `<|im_end|>`,
become their ids. `add_special_tokens` controls the post-processor prefix (BOS).
"""
function encode(tok::BPETokenizer, text::AbstractString; add_special_tokens::Bool = true)
    ids = add_special_tokens ? copy(tok.prefix) : Int[]
    # As Hugging Face `tokenizers`: tokens split out of the raw text, normalization of the
    # pieces between them, then the normalized tokens split out of those.
    foreach_piece(tok.raw_added, text) do piece, id
        id === nothing || return push!(ids, id)
        foreach_piece(tok.normalized_added, apply_normalizer(piece, tok.normalizer)) do p, i
            i === nothing ? encode_text!(ids, tok, p) : push!(ids, i)
        end
    end
    return ids
end

function encode_text!(ids, tok::BPETokenizer, text::AbstractString)
    isempty(text) && return ids
    # Split "Isolated": matches are words, and so is any text between matches.
    start = 1
    for m in eachmatch(tok.pattern, text)
        m.offset > start && append!(ids, encode_word(tok, SubString(text, start, prevind(text, m.offset))))
        append!(ids, encode_word(tok, m.match))
        start = m.offset + ncodeunits(m.match)
    end
    start <= ncodeunits(text) && append!(ids, encode_word(tok, SubString(text, start)))
    return ids
end

function encode_word(tok::BPETokenizer, word::AbstractString)
    get!(tok.cache, word) do
        symbols = byte_level(word)
        tok.ignore_merges && haskey(tok.vocab, symbols) && return [tok.vocab[symbols]]
        return [tok.vocab[s] for s in bpe(tok.merge_ranks, symbols)]
    end
end

# Repeatedly merges the adjacent pair with the lowest rank, all of its occurrences at once.
function bpe(ranks::Dict{Tuple{String,String},Int}, word::String)
    parts = [string(c) for c in word]
    while length(parts) > 1
        best, best_rank = nothing, typemax(Int)
        for i in 1:length(parts)-1
            r = get(ranks, (parts[i], parts[i+1]), typemax(Int))
            r < best_rank && ((best, best_rank) = ((parts[i], parts[i+1]), r))
        end
        best === nothing && break
        merged = String[]
        i = 1
        while i <= length(parts)
            if i < length(parts) && (parts[i], parts[i+1]) == best
                push!(merged, parts[i] * parts[i+1])
                i += 2
            else
                push!(merged, parts[i])
                i += 1
            end
        end
        parts = merged
    end
    return parts
end

"""
    decode(tok, ids; skip_special_tokens = false) -> String

Inverse of [`encode`](@ref). Bytes that do not form valid UTF-8 on their own (a token
boundary inside a multi-byte character) are kept as-is.
"""
function decode(tok::BPETokenizer, ids; skip_special_tokens::Bool = false)
    bytes = UInt8[]
    for id in ids
        skip_special_tokens && id in tok.special && continue
        s = tok.tokens[id+1]
        if haskey(tok.added, s)
            append!(bytes, codeunits(s))
        else
            for c in s
                push!(bytes, CHAR_TO_BYTE[c])
            end
        end
    end
    return String(bytes)
end

Base.length(tok::BPETokenizer) = length(tok.tokens)
