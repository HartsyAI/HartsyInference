"""Summarizes a safetensors checkpoint from its headers alone: it never downloads tensor data and never unpickles anything.

Usage:
    python -I tools/inspect_checkpoint_header.py <folder | file.safetensors | org/repo> [--revision main] [--json out.json]
        [--tokens "<|semantic:0|>" "<|im_end|>" ...]

For a folder or file it reads each .safetensors header from disk. For a hub repo it lists the files and reads each shard's header
with HTTP range requests (huggingface_hub.get_safetensors_metadata), plus config.json and tokenizer.json, which are small.

It prints, and with --json writes:
- the files and their sizes, and whether codec.pth is present;
- config.json;
- the tensor count per dtype and per two-part key prefix;
- the quantization markers it finds: GPTQ qweight / scales / qzeros / g_idx, FP8 weight_scale_inv, comfy_quant;
- the shapes of every tensor in the first layer of each prefix;
- the ids of the --tokens special tokens from tokenizer.json.
"""
import argparse
import collections
import json
import os
import struct
import sys

MARKERS = ("qweight", "scales", "qzeros", "g_idx", "weight_scale_inv", "comfy_quant", "input_scale")


def read_local_header(path):
    with open(path, "rb") as f:
        (length,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(length))
    header.pop("__metadata__", None)
    return {name: (info["dtype"], info["shape"]) for name, info in header.items()}


def local_source(root):
    files = [root] if os.path.isfile(root) else [os.path.join(root, n) for n in sorted(os.listdir(root))]
    sizes = {os.path.basename(p): os.path.getsize(p) for p in files if os.path.isfile(p)}
    tensors = {}
    for path in files:
        if path.endswith(".safetensors"):
            tensors.update(read_local_header(path))
    folder = root if os.path.isdir(root) else os.path.dirname(root)

    def read_json(name):
        path = os.path.join(folder, name)
        return json.load(open(path, encoding="utf-8")) if os.path.isfile(path) else None

    return sizes, tensors, read_json


def hub_source(repo, revision):
    from huggingface_hub import HfApi, get_safetensors_metadata, hf_hub_download

    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    sizes = {s.rfilename: s.size for s in info.siblings}
    tensors = {}
    metadata = get_safetensors_metadata(repo, revision=revision)
    for shard in metadata.files_metadata.values():
        for name, tensor in shard.tensors.items():
            tensors[name] = (tensor.dtype, tensor.shape)

    def read_json(name):
        if name not in sizes:
            return None
        return json.load(open(hf_hub_download(repo, name, revision=revision), encoding="utf-8"))

    return sizes, tensors, read_json


def first_layer_names(names):
    """Every tensor of layer 0 under each prefix that has numbered layers, plus every tensor outside numbered layers."""
    picked = []
    for name in names:
        parts = name.split(".")
        numbered = [i for i, p in enumerate(parts) if p.isdigit()]
        if not numbered or parts[numbered[0]] == "0" and all(parts[i] == "0" for i in numbered[1:]):
            picked.append(name)
    return picked


def special_ids(tokenizer, tokens):
    if tokenizer is None:
        return {}
    added = {t["content"]: t["id"] for t in tokenizer.get("added_tokens", [])}
    vocab = tokenizer.get("model", {}).get("vocab", {})
    return {t: added.get(t, vocab.get(t)) for t in tokens}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--json")
    parser.add_argument("--tokens", nargs="*", default=["<|semantic:0|>", "<|im_end|>", "<|voice|>", "<|speaker:0|>"])
    args = parser.parse_args()

    if os.path.exists(args.source):
        sizes, tensors, read_json = local_source(args.source)
    else:
        sizes, tensors, read_json = hub_source(args.source, args.revision)

    config = read_json("config.json")
    if config is not None and "semantic_end_id" in config and "semantic_start_id" in config:
        args.tokens.append(f"<|semantic:{config['semantic_end_id'] - config['semantic_start_id']}|>")
    by_dtype = collections.Counter(dtype for dtype, _ in tensors.values())
    by_prefix = collections.Counter(".".join(n.split(".")[:2]) for n in tensors)
    markers = {m: sum(1 for n in tensors if n.endswith("." + m) or n.endswith(m)) for m in MARKERS}
    report = {
        "source": args.source,
        "files": sizes,
        "has_codec_pth": "codec.pth" in sizes,
        "config": config,
        "tensor_count": len(tensors),
        "by_dtype": dict(by_dtype),
        "by_prefix": dict(sorted(by_prefix.items())),
        "quant_markers": {m: c for m, c in markers.items() if c},
        "first_layer": {n: {"dtype": tensors[n][0], "shape": tensors[n][1]} for n in sorted(first_layer_names(tensors))},
        "special_tokens": special_ids(read_json("tokenizer.json"), args.tokens),
    }
    text = json.dumps(report, indent=2, default=str)
    print(text)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    sys.exit(main())
