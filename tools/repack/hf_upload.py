"""HuggingFace upload for repacked single-file models.

Gated on Recipe.redistribute: refuses to push unless the recipe explicitly opts in, so a
non-commercial model (FishSpeech, F5-TTS, MusicGen) can be repacked locally for convenience
but never accidentally re-hosted. The original LICENSE is copied into the target repo and an
auto README documents that this is a format-only repack (no weight changes).

Requires: pip install huggingface_hub ; auth via the HF_TOKEN environment variable.
"""

from __future__ import annotations

import json
import os
import sys

from recipe import Recipe, manifest_path

_LICENSE_CANDIDATES = ("LICENSE", "LICENSE.txt", "LICENSE.md", "license", "license.txt")


def _readme(recipe: Recipe, manifest: dict) -> str:
    return f"""---
license: {recipe.license}
tags:
  - hartsyinference
  - repack
---

# {recipe.name} (single-file safetensors repack)

Format-only repack of [`{recipe.source_repo}`](https://huggingface.co/{recipe.source_repo})
for use with [HartsyInference](https://github.com/kalebbroo/HartsyInference). **No weights
were changed** — the original checkpoint's tensors were merged into one `.safetensors` with a
flat key layout so the engine can load a single verified file instead of parsing pickles or
stitching multiple component files.

- **Source:** `{recipe.source_repo}` @ `{recipe.revision}`
- **License:** `{recipe.license}` (inherited from source; see `LICENSE`)
- **Output:** `{manifest['output_file']}` — {manifest['tensor_count']} tensors
- **sha256:** `{manifest['output_sha256']}`

Provenance and the exact merge steps are recorded in `{recipe.name}.manifest.json`.
"""


def upload(recipe: Recipe, output_path: str) -> None:
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as e:  # pragma: no cover
        print(f"missing dependency: {e}. Install with: pip install huggingface_hub", file=sys.stderr)
        raise

    if not recipe.redistribute:
        print(
            f"REFUSING to upload '{recipe.name}': recipe.redistribute is False.\n"
            f"  Source license: {recipe.license}\n"
            f"  This is the manual per-model sign-off. If '{recipe.license}' permits "
            f"redistribution, set redistribute=True in recipes/{recipe.name.split('-')[0]}.py "
            f"and re-run.",
            file=sys.stderr,
        )
        sys.exit(2)

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN not set. Create one at https://huggingface.co/settings/tokens", file=sys.stderr)
        sys.exit(1)

    mpath = manifest_path(output_path)
    if not os.path.exists(mpath):
        print(f"manifest not found next to output: {mpath}. Run `repack` first.", file=sys.stderr)
        sys.exit(1)
    with open(mpath, encoding="utf-8") as f:
        manifest = json.load(f)

    api = HfApi(token=token)
    print(f"creating repo {recipe.target_repo} (if needed) ...")
    api.create_repo(recipe.target_repo, repo_type="model", exist_ok=True)

    # Pull the source LICENSE so the redistribution carries its terms.
    license_local = None
    for cand in _LICENSE_CANDIDATES:
        try:
            license_local = hf_hub_download(recipe.source_repo, cand, revision=recipe.revision)
            break
        except Exception:
            continue
    if license_local is None:
        print(f"WARNING: no LICENSE file found in {recipe.source_repo}; uploading without one.",
              file=sys.stderr)

    # Generated README documenting the repack.
    readme_local = output_path + ".README.md"
    with open(readme_local, "w", encoding="utf-8") as f:
        f.write(_readme(recipe, manifest))

    print(f"uploading {os.path.basename(output_path)} ...")
    api.upload_file(path_or_fileobj=output_path,
                    path_in_repo=manifest["output_file"],
                    repo_id=recipe.target_repo)
    api.upload_file(path_or_fileobj=mpath,
                    path_in_repo=os.path.basename(mpath),
                    repo_id=recipe.target_repo)
    api.upload_file(path_or_fileobj=readme_local, path_in_repo="README.md", repo_id=recipe.target_repo)
    if license_local:
        api.upload_file(path_or_fileobj=license_local, path_in_repo="LICENSE", repo_id=recipe.target_repo)

    # Extra files (config.json, vocab.txt, ...) so the repo is self-sufficient for the engine.
    for extra in recipe.extra_files:
        try:
            local = hf_hub_download(recipe.source_repo, extra, revision=recipe.revision)
            api.upload_file(path_or_fileobj=local, path_in_repo=extra, repo_id=recipe.target_repo)
        except Exception as ex:
            print(f"WARNING: could not copy extra file '{extra}': {ex}", file=sys.stderr)

    os.remove(readme_local)
    print(f"done: https://huggingface.co/{recipe.target_repo}")
