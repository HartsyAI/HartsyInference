#!/usr/bin/env python3
"""Repack orchestrator CLI: turn an official audio model checkpoint into one verified
single-file .safetensors, and (for license-permissive models) host it on HuggingFace.

Subcommands
-----------
  list                          List known recipes.
  repack  <name> [-o DIR]       Download source(s), merge to one .safetensors + manifest.
  verify  <file.safetensors>    Recompute sha256 and compare against the sidecar manifest.
  upload  <name> [-o DIR]       Push the repacked file + manifest + LICENSE + README to HF
                                (refused unless the recipe's redistribute flag is True).

Examples
--------
  python tools/repack/repack.py repack kokoro-82m
  python tools/repack/repack.py verify out/kokoro-82m.safetensors
  HF_TOKEN=... python tools/repack/repack.py upload kokoro-82m

Requires: pip install torch safetensors huggingface_hub
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import recipe as rcp  # noqa: E402
import recipes  # noqa: E402


def _fetch_factory(recipe: rcp.Recipe, source_hashes: dict[str, str]):
    """Returns a fetch(filename)->local_path that downloads from the source HF repo and
    records each source file's sha256 for the manifest."""
    from huggingface_hub import hf_hub_download

    def fetch(filename: str) -> str:
        path = hf_hub_download(recipe.source_repo, filename, revision=recipe.revision)
        source_hashes[filename] = rcp.sha256_file(path)
        return path

    return fetch


def cmd_list(_args) -> int:
    for name in recipes.names():
        r = recipes.get(name)
        host = "host-ok" if r.redistribute else "local-only"
        print(f"  {name:24s} {r.license:14s} {host:11s} <- {r.source_repo}")
    return 0


def cmd_repack(args) -> int:
    recipe = recipes.get(args.name)
    out_dir = args.output or "out"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, recipe.name + ".safetensors")

    source_hashes: dict[str, str] = {}
    fetch = _fetch_factory(recipe, source_hashes)

    print(f"repacking {recipe.name} from {recipe.source_repo} ...")
    tensors = recipe.build(fetch)
    print(f"merged {len(tensors)} tensors")
    rcp.save_safetensors(tensors, out_path, recipe)
    mpath = rcp.write_manifest(recipe, out_path, source_hashes, len(tensors))
    print(f"wrote {out_path}")
    print(f"wrote {mpath}")
    print(f"sha256 {rcp.sha256_file(out_path)}")
    return 0


def cmd_verify(args) -> int:
    out_path = args.file
    mpath = rcp.manifest_path(out_path)
    if not os.path.exists(mpath):
        print(f"no manifest found: {mpath}", file=sys.stderr)
        return 1
    with open(mpath, encoding="utf-8") as f:
        manifest = json.load(f)
    actual = rcp.sha256_file(out_path)
    expected = manifest.get("output_sha256")
    if actual == expected:
        print(f"OK  sha256 matches manifest: {actual}")
        return 0
    print(f"MISMATCH\n  expected {expected}\n  actual   {actual}", file=sys.stderr)
    return 1


def cmd_upload(args) -> int:
    import hf_upload
    recipe = recipes.get(args.name)
    out_dir = args.output or "out"
    out_path = os.path.join(out_dir, recipe.name + ".safetensors")
    if not os.path.exists(out_path):
        print(f"repacked file not found: {out_path}. Run `repack {recipe.name}` first.",
              file=sys.stderr)
        return 1
    hf_upload.upload(recipe, out_path)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="List known recipes.").set_defaults(func=cmd_list)

    p_repack = sub.add_parser("repack", help="Download + merge to one .safetensors.")
    p_repack.add_argument("name")
    p_repack.add_argument("-o", "--output", help="Output directory (default: out/).")
    p_repack.set_defaults(func=cmd_repack)

    p_verify = sub.add_parser("verify", help="Check sha256 against the sidecar manifest.")
    p_verify.add_argument("file")
    p_verify.set_defaults(func=cmd_verify)

    p_upload = sub.add_parser("upload", help="Push to HuggingFace (permissive recipes only).")
    p_upload.add_argument("name")
    p_upload.add_argument("-o", "--output", help="Output directory (default: out/).")
    p_upload.set_defaults(func=cmd_upload)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
