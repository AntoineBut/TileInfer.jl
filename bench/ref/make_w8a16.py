"""Write a W8A16 compressed-tensors copy of a Hugging Face checkpoint for vLLM, quantized
exactly as TileInfer's `quantize_int8`: every linear weight (and the lm_head, untied from
the embedding) as Int8 with one symmetric scale per output channel, max |W[j, :]| / 127,
rounded to nearest and clamped to [-127, 127]. The embedding stays in BFloat16. vLLM runs
it with its Marlin W8A16 kernels (`pack-quantized`, 8 bits, channel strategy).

    bench/ref/.venv/bin/python bench/ref/make_w8a16.py Qwen/Qwen3-0.6B [--out DIR]

Default output: bench/ref/models/<name>-w8a16.
"""

import argparse
import glob
import json
import os
import shutil

import torch
from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32
from safetensors import safe_open
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
LINEAR = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def quantize(w):
    """(out, in) BFloat16 -> packed Int8 (out, in / 4) as int32 and (out, 1) scales."""
    w = w.float()
    absmax = w.abs().amax(dim=1, keepdim=True)
    scale = torch.where(absmax > 0, absmax / 127, torch.ones_like(absmax))
    q = torch.clamp(torch.round(w / scale), -127, 127).to(torch.int8)
    return pack_to_int32(q, 8), scale.to(torch.bfloat16)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    org, name = a.model.split("/")
    snaps = glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--{org}--{name}/snapshots/*"))
    src = max(snaps, key=os.path.getmtime)
    out = a.out or os.path.join(HERE, "models", a.model.split("/")[-1] + "-w8a16")
    os.makedirs(out, exist_ok=True)

    tensors = {}
    for f in sorted(glob.glob(os.path.join(src, "*.safetensors"))):
        with safe_open(f, "pt") as st:
            for k in st.keys():
                tensors[k] = st.get_tensor(k)
    if "lm_head.weight" not in tensors:
        tensors["lm_head.weight"] = tensors["model.embed_tokens.weight"].clone()

    result = {}
    for k, w in tensors.items():
        name = k.removesuffix(".weight")
        if k.endswith(".weight") and (name.split(".")[-1] in LINEAR or name == "lm_head"):
            packed, scale = quantize(w)
            result[name + ".weight_packed"] = packed
            result[name + ".weight_scale"] = scale
            result[name + ".weight_shape"] = torch.tensor(list(w.shape), dtype=torch.int32)
        else:
            result[k] = w.contiguous()
    save_file(result, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})

    with open(os.path.join(src, "config.json")) as f:
        cfg = json.load(f)
    cfg["tie_word_embeddings"] = False
    cfg["quantization_config"] = {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "quantization_status": "compressed",
        "ignore": [],
        "config_groups": {"group_0": {
            "targets": ["Linear", "lm_head"],
            "format": "pack-quantized",
            "weights": {"num_bits": 8, "type": "int", "symmetric": True, "strategy": "channel",
                        "dynamic": False, "group_size": None, "observer": "minmax"},
            "input_activations": None, "output_activations": None,
        }},
    }
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    for f in os.listdir(src):
        if f.startswith("tokenizer") or f in ("generation_config.json", "vocab.json", "merges.txt"):
            shutil.copy(os.path.join(src, f), out)
    print(out)


if __name__ == "__main__":
    main()
