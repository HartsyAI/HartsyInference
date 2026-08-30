#!/usr/bin/env python3
"""Stamp SAI ModelSpec + hartsy.* identity into safetensors headers, or emit .swarm.json sidecars.

Why this exists
---------------
SwarmUI classifies a scanned model from `modelspec.architecture` in the safetensors
`__metadata__` block (checked BEFORE any tensor-shape predicate), and reads title/author/
license/description from the same block. Our repacked audio artifacts carry no metadata at
all, so a file-scanned audio model would land with a null class — which in AudioLab means
every audio parameter silently disappears from the UI. Stamping is what makes the file
self-describing.

Pure stdlib: a safetensors header is an 8-byte little-endian length followed by that many
bytes of JSON. Inserting `__metadata__` changes the header length, so the payload shifts and
the file must be rewritten. No torch, no safetensors package.

Hard links
----------
Most staged artifacts are hard-linked to the engine's own model cache (30 of 48 in the
2026-08-25 staging tree). Two consequences this tool refuses to get wrong:

  * Never rewrite in place. An in-place header write through a shared inode corrupts the
    engine's live cache copy, silently and irreversibly.
  * Write-new-then-rename breaks the link, so the stamped copy stops sharing storage with
    the cache and disk usage grows by the size of every stamped file. `--dry-run` reports
    that growth up front; `--allow-unlink` is required before any linked file is touched.

Usage
-----
    stamp_modelspec.py --root DIR [--only PATH_PREFIX] --dry-run
    stamp_modelspec.py --root DIR --only stt/whisper/base --allow-unlink
    stamp_modelspec.py --root DIR --sidecar-only        # write .swarm.json, touch no weights
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import shutil
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
IDENTITY_PATH = os.path.join(HERE, "audiolab_identity.json")
SPEC_VERSION = "1.0.1"
IMPLEMENTATION = "https://github.com/HartsyAI/HartsyInference"
ARTIFACT_SCHEMA = "1"
COPY_CHUNK = 1 << 22


def read_header(path):
    """Returns (header_dict, header_length, payload_offset)."""
    with open(path, "rb") as f:
        raw_len = f.read(8)
        if len(raw_len) != 8:
            raise ValueError(f"{path}: too small to be safetensors")
        (header_len,) = struct.unpack("<Q", raw_len)
        if header_len <= 0 or header_len > (1 << 30):
            raise ValueError(f"{path}: implausible header length {header_len}")
        header = json.loads(f.read(header_len))
    return header, header_len, 8 + header_len


def payload_sha256(path, payload_offset):
    """SHA-256 of the tensor bytes only, matching SwarmUI's modelspec.hash_sha256 convention.

    Excluding the header is what lets the value survive a restamp — hash the whole file and
    every metadata edit invalidates it.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        f.seek(payload_offset)
        while True:
            chunk = f.read(COPY_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_identity():
    with open(IDENTITY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def classify(rel_path, families):
    """Maps a staged relative path to its family entry, variant and component role.

    Returns None when the path is outside the identity contract (REVIEW_REQUIRED, unknown
    family) — those are skipped rather than guessed at.
    """
    parts = rel_path.split("/")
    # The marker appears both as its own directory and as a suffix (voices_REVIEW_REQUIRED).
    if len(parts) < 3 or any("REVIEW_REQUIRED" in p for p in parts):
        return None
    family_key = "/".join(parts[:2])
    family = families.get(family_key)
    if family is None:
        return None
    variant_dir = parts[2]
    tail = "/".join(parts[3:]) if len(parts) > 3 else parts[-1]
    filename = parts[-1]

    # _shared trees hold cross-variant components; they have no variant of their own.
    if variant_dir.startswith("_shared"):
        role = None
        for pattern, name in (family.get("shared_components") or {}).items():
            if fnmatch.fnmatch("/".join(parts[2:]), pattern):
                role = name
                break
        return {"family_key": family_key, "family": family, "variant": None,
                "component": role or "shared", "filename": filename}

    role = None
    for pattern, name in (family.get("components") or {}).items():
        if fnmatch.fnmatch(tail, pattern) or fnmatch.fnmatch(filename, pattern):
            role = name
            break
    if role is None:
        # Some families keep their shared components inside a variant dir (rvc/v2/shared/...).
        for pattern, name in (family.get("shared_components") or {}).items():
            if fnmatch.fnmatch(tail, pattern):
                role = name
                break
    if role is None:
        primary = family.get("primary", "")
        if fnmatch.fnmatch(tail, primary) or fnmatch.fnmatch(filename, primary):
            role = "main"
        elif fnmatch.fnmatch(filename, "model-*-of-*.safetensors"):
            # Later shards of a sharded set: the first shard is the entrypoint, the rest ride along.
            role = "shard"
        else:
            role = "aux"
    return {"family_key": family_key, "family": family, "variant": variant_dir,
            "component": role, "filename": filename}


def variant_id(family, variant):
    mapped = (family.get("variant_id") or {})
    if variant in mapped:
        return mapped[variant]
    if "*" in mapped:
        return mapped["*"]
    return variant


def architecture_for(family, variant):
    per_variant = (family.get("variant_architecture") or {})
    return per_variant.get(variant, family["architecture"])


def title_for(family, variant, model_id):
    return f"{family['prefix']} {model_id}" if variant else f"{family['prefix']} ({family['architecture']})"


def build_metadata(info, existing, tensor_hash):
    family = info["family"]
    variant = info["variant"]
    model_id = variant_id(family, variant) if variant else None
    arch = architecture_for(family, variant)
    meta = dict(existing or {})
    # Upstream files often carry {"format": "pt"} — HF tooling keys off it, so preserve it.
    meta.update({
        "modelspec.sai_model_spec": SPEC_VERSION,
        "modelspec.architecture": arch,
        "modelspec.implementation": IMPLEMENTATION,
        "modelspec.title": title_for(family, variant, model_id),
        "modelspec.author": family.get("author", ""),
        "modelspec.license": family.get("license", ""),
        "modelspec.hash_sha256": "0x" + tensor_hash,
        "hartsy.artifact_schema": ARTIFACT_SCHEMA,
        "hartsy.provider_id": family["provider_id"],
        "hartsy.engine_id": family["engine_id"],
        "hartsy.component": info["component"],
    })
    if model_id:
        meta["hartsy.model_id"] = model_id
    if family.get("alias_of"):
        meta["hartsy.alias_of"] = family["alias_of"]
    # modelspec.resolution is deliberately absent: a value that disagrees with a class's
    # declared standard makes SwarmUI clone the class with its matcher disabled.
    return meta


def rewrite_with_metadata(path, metadata, keep_backup=False):
    """Writes path.stamped with the new header + original payload, then renames over path."""
    header, _, payload_offset = read_header(path)
    header = {k: v for k, v in header.items() if k != "__metadata__"}
    new_header = {"__metadata__": {k: str(v) for k, v in metadata.items()}}
    new_header.update(header)
    encoded = json.dumps(new_header, separators=(",", ":")).encode("utf-8")
    tmp = path + ".stamped"
    with open(path, "rb") as src, open(tmp, "wb") as dst:
        dst.write(struct.pack("<Q", len(encoded)))
        dst.write(encoded)
        src.seek(payload_offset)
        shutil.copyfileobj(src, dst, COPY_CHUNK)
    if keep_backup:
        os.replace(path, path + ".orig")
    os.replace(tmp, path)


def sidecar_path(path):
    return os.path.splitext(path)[0] + ".swarm.json"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="staged artifact tree root")
    ap.add_argument("--only", default=None, help="restrict to paths starting with this relative prefix")
    ap.add_argument("--dry-run", action="store_true", help="report what would change, touch nothing")
    ap.add_argument("--allow-unlink", action="store_true",
                    help="permit stamping files with a link count > 1 (breaks sharing with the engine cache)")
    ap.add_argument("--sidecar-only", action="store_true",
                    help="write .swarm.json next to each artifact instead of rewriting headers")
    ap.add_argument("--keep-backup", action="store_true", help="leave the pre-stamp file as *.orig")
    args = ap.parse_args()

    identity = load_identity()
    families = identity["families"]
    root = os.path.abspath(args.root)

    planned, skipped, linked_bytes = [], [], 0
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in sorted(filenames):
            # .onnx carries no header we can stamp, but SwarmUI scans it now, so it still needs a sidecar.
            if not name.endswith((".safetensors", ".sft", ".onnx", ".gguf")):
                continue
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, root)
            if args.only and not rel.startswith(args.only):
                continue
            info = classify(rel, families)
            if info is None:
                skipped.append((rel, "outside the identity contract"))
                continue
            st = os.stat(full)
            if st.st_nlink > 1:
                linked_bytes += st.st_size
            planned.append((full, rel, info, st))

    if not planned:
        print("nothing matched", file=sys.stderr)
        return 1

    print(f"{len(planned)} artifact(s) in scope, {len(skipped)} skipped")
    if linked_bytes:
        print(f"WARNING: {sum(1 for _, _, _, s in planned if s.st_nlink > 1)} file(s) are hard-linked "
              f"({linked_bytes / 1e9:.1f} GB). Stamping unshares them from the engine cache and grows disk by that much.")
    free = shutil.disk_usage(root).free
    print(f"free space: {free / 1e9:.1f} GB")

    exit_code = 0
    for full, rel, info, st in planned:
        arch = architecture_for(info["family"], info["variant"])
        label = f"{rel}\n    arch={arch} component={info['component']}"
        if args.dry_run:
            print(f"[dry-run] {label}")
            continue
        if st.st_nlink > 1 and not args.sidecar_only and not args.allow_unlink:
            print(f"[REFUSED] {rel}: link count {st.st_nlink}; pass --allow-unlink to break sharing")
            exit_code = 2
            continue
        try:
            if rel.endswith((".safetensors", ".sft")):
                _header, _len, payload_offset = read_header(full)
                tensor_hash = payload_sha256(full, payload_offset)
                existing = _header.get("__metadata__") if isinstance(_header.get("__metadata__"), dict) else None
            else:
                # No stampable header (ONNX, GGUF): hash the whole file and describe it from the sidecar only.
                tensor_hash = payload_sha256(full, 0)
                existing = None
                if not args.sidecar_only:
                    print(f"[skipped] {rel}: no safetensors header to stamp; re-run with --sidecar-only")
                    continue
            meta = build_metadata(info, existing, tensor_hash)
            if args.sidecar_only:
                with open(sidecar_path(full), "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2, sort_keys=True)
                    f.write("\n")
                print(f"[sidecar] {label}")
            else:
                rewrite_with_metadata(full, meta, keep_backup=args.keep_backup)
                print(f"[stamped] {label}")
        except Exception as exc:  # noqa: BLE001 - one bad artifact must not abort the run
            print(f"[ERROR] {rel}: {exc}", file=sys.stderr)
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
