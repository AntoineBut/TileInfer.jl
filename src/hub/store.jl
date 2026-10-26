# Model store on top of the Hugging Face hub cache (~/.cache/huggingface/hub, shared with
# Python's huggingface_hub). A model is identified by a repo id such as "Qwen/Qwen3-0.6B",
# or by a local directory with the same files.

using HuggingFaceApi: HuggingFaceApi, hf_hub_download

# Top-level files needed for inference: weights, configs, tokenizer. Everything else
# (pytorch_model.bin, onnx/, gguf, README, …) is skipped.
function wanted_file(name::AbstractString)
    occursin('/', name) && return false
    return endswith(name, ".safetensors") || endswith(name, ".json") || name == "merges.txt"
end

"""
    pull(repo_id; revision = "main") -> snapshot directory

Download the inference files of `repo_id` into the hub cache and return the directory of
the snapshot. Files already in the cache are not downloaded again.
"""
function pull(repo_id::AbstractString; revision::AbstractString = "main")
    files = filter(wanted_file, HuggingFaceApi.list_model_files(repo_id; revision))
    any(endswith(".safetensors"), files) ||
        error("$repo_id has no safetensors weights; only safetensors checkpoints are supported")
    "config.json" in files || error("$repo_id has no config.json")
    # config.json last: `model_dir` resolves a model through it, so an interrupted pull is
    # never mistaken for a complete one.
    sort!(files; by = ==("config.json"))
    snapshots = Set{String}()
    for file in files
        push!(snapshots, dirname(hf_hub_download(repo_id, file; revision)))
    end
    length(snapshots) == 1 ||
        error("$repo_id changed on the hub while downloading (revision $revision); pull again")
    return only(snapshots)
end

"""
    model_dir(model; revision = "main") -> directory

Resolve `model` (a local directory or a pulled repo id) to the directory holding its files,
without network access. A repo counts as pulled once its config.json, which `pull`
downloads last, is in the cache.
"""
function model_dir(model::AbstractString; revision::AbstractString = "main")
    isdir(model) && return abspath(model)
    config = try
        hf_hub_download(model, "config.json"; revision, local_files_only = true)
    catch
        error("$model is not pulled (revision $revision); run `tileinfer pull $model`")
    end
    return dirname(config)
end

"""
    CachedModel

A model in the hub cache: repo id, revision (a ref such as "main", or a commit hash),
snapshot files and their total size in bytes.
"""
struct CachedModel
    repo_id::String
    revision::String
    files::Vector{String}
    bytes::Int
end

"List the models in the hub cache that have safetensors weights."
function list_models()
    models = CachedModel[]
    for (repo_id, revisions) in get(HuggingFaceApi.list_cache(), "model", Dict())
        for (revision, files) in revisions
            any(endswith(".safetensors"), files) || continue
            dir = model_dir(repo_id; revision)
            bytes = sum(f -> filesize(realpath(joinpath(dir, f))), files)
            push!(models, CachedModel(repo_id, revision, sort(files), bytes))
        end
    end
    return sort!(models; by = m -> (m.repo_id, m.revision))
end

"Remove a pulled model (all revisions, or only `revision`) from the hub cache."
remove_model(repo_id::AbstractString; revision = nothing) =
    HuggingFaceApi.remove_cache(repo_id; repo_type = "model", revision)
