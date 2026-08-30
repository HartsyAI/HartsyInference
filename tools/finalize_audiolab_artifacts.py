#!/usr/bin/env python3
"""Validate a staged AudioLab artifact tree and emit an upload manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from safetensors import safe_open

DEFAULT_ROOT = Path.home() / "Desktop" / "AudioLab-Model-Artifacts"
GENERATED = {"staged-files.json", "upload-manifest.json"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_safetensors(path: Path) -> dict[str, Any]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        dtypes: dict[str, int] = {}
        for key in keys:
            dtype = str(handle.get_slice(key).get_dtype())
            dtypes[dtype] = dtypes.get(dtype, 0) + 1
        return {"tensor_count": len(keys), "tensor_dtypes": dtypes, "valid_header": True}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    entries: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in GENERATED or path.name.endswith(".manifest.json"):
            continue
        relative = path.relative_to(root)
        item: dict[str, Any] = {
            "path": relative.as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "extension": "".join(path.suffixes),
        }
        if path.suffix == ".safetensors":
            try:
                item.update(inspect_safetensors(path))
            except Exception as exc:  # report every bad artifact; never silently omit it
                item["valid_header"] = False
                errors.append({"path": relative.as_posix(), "error": str(exc)})
        entries.append(item)

    result = {
        "schema": 1,
        "root": str(root),
        "file_count": len(entries),
        "logical_bytes": sum(x["bytes"] for x in entries),
        "validation_errors": errors,
        "files": entries,
    }
    (root / "upload-manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Validated {len(entries)} files ({result['logical_bytes']} logical bytes)")
    print(f"SafeTensors/header errors: {len(errors)}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
