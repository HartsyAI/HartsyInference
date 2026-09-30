"""Download the non-weight files of deepseek-ai/DeepSeek-V4.1-Flash at the pinned revision and verify SHA-256.

The 48 weight shards are never fetched here. Usage: python fetch_upstream.py [--dest ~/dsv41-ref/upstream]
"""
import argparse
import hashlib
import os
import sys

REPO: str = "deepseek-ai/DeepSeek-V4.1-Flash"
REVISION: str = "dba1be0a40aa45a94ad051997016db3960a90277"
HERE: str = os.path.dirname(os.path.abspath(__file__))


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_manifest() -> dict:
    out = {}
    with open(os.path.join(HERE, "upstream.sha256")) as f:
        for line in f:
            digest, name = line.rstrip("\n").split("  ", 1)
            out[name] = digest
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=os.path.expanduser("~/dsv41-ref/upstream"))
    ap.add_argument("--verify-only", action="store_true")
    a = ap.parse_args()
    if not a.verify_only:
        from huggingface_hub import snapshot_download
        snapshot_download(REPO, revision=REVISION, local_dir=a.dest,
                          ignore_patterns=["*.safetensors", "*.bin", "*.gguf"])
    if not os.path.isdir(a.dest):
        print(f"destination not found: {a.dest}")
        return 1
    manifest = read_manifest()
    bad = 0
    for name, digest in manifest.items():
        p = os.path.join(a.dest, name)
        if not os.path.isfile(p):
            print(f"MISSING {name}")
            bad += 1
        elif sha256_file(p) != digest:
            print(f"MISMATCH {name}")
            bad += 1
    print(f"{len(manifest) - bad} verified, {bad} bad ({REPO}@{REVISION[:8]})")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
