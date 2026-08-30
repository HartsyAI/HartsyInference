#!/usr/bin/env python3
"""Build an upload-oriented AudioLab model tree from the local engine cache.

The tree uses hard links for existing artifacts on the same filesystem, so staging
multi-gigabyte SafeTensors/ONNX/GGUF files does not duplicate their disk blocks.
Missing variants still receive an artifact-plan.json, making absence explicit.
Pickle conversion is intentionally performed separately with CheckpointRepacker;
this script never relabels a pickle file as SafeTensors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

DEFAULT_CACHE = Path(__file__).resolve().parents[1] / "Models" / "audio"
DEFAULT_OUTPUT = Path.home() / "Desktop" / "AudioLab-Model-Artifacts"

CATALOG: dict[str, dict[str, Any]] = {
    "stt/whisper": {"variants": ["tiny", "base", "small", "medium", "large-v2", "large-v3", "turbo"], "format": "safetensors+sidecars", "license": "apache-2.0"},
    "stt/distil-whisper": {"variants": ["large-v3", "large-v3.5"], "format": "safetensors+sidecars", "license": "mit"},
    "stt/moonshine": {"variants": ["base", "tiny"], "format": "safetensors+sidecars", "license": "mit"},
    "stt/moonshine-streaming": {"variants": ["tiny", "small", "medium"], "format": "safetensors+sidecars", "license": "mit"},
    "stt/kyutai-stt": {"variants": ["1b-en-fr", "2.6b-en"], "format": "safetensors+sentencepiece", "license": "cc-by-4.0"},
    "stt/whisper-streaming": {"variants": ["base"], "format": "alias:stt/whisper/base", "license": "apache-2.0"},
    "tts/vibevoice": {"variants": ["1.5b"], "format": "sharded-safetensors+sidecars", "license": "mit"},
    "tts/kokoro": {"variants": ["default"], "format": "safetensors+voice-assets+sidecars", "license": "apache-2.0"},
    "tts/bark": {"variants": ["default"], "format": "safetensors+sidecars", "license": "mit"},
    "tts/dia": {"variants": ["1.6b"], "format": "multi-component-safetensors", "license": "apache-2.0"},
    "tts/orpheus": {"variants": ["3b"], "format": "sharded-safetensors+snac-safetensors", "license": "apache-2.0"},
    "tts/csm": {"variants": ["1b"], "format": "safetensors", "license": "apache-2.0"},
    "tts/neutts": {"variants": ["air"], "format": "safetensors+sidecars", "license": "gated-review-required", "upload_allowed": False},
    "tts/fish-speech": {"variants": ["fish-speech-1.5"], "format": "multi-component-safetensors+sidecars", "license": "cc-by-nc-sa-4.0", "upload_allowed": False},
    "tts/cosyvoice2": {"variants": ["2-0.5b"], "format": "multi-component-safetensors", "license": "multi-source-notices"},
    "tts/f5-tts": {"variants": ["v1-base"], "format": "multi-component-safetensors+sidecars", "license": "cc-by-nc-4.0", "upload_allowed": False},
    "tts/zipvoice": {"variants": ["base"], "format": "multi-component-safetensors+sidecars", "license": "unknown", "upload_allowed": False},
    "tts/qwen3-tts": {"variants": ["1.7b-base", "0.6b-base", "1.7b-customvoice", "0.6b-customvoice", "1.7b-voicedesign"], "format": "multi-component-safetensors+sidecars", "license": "apache-2.0"},
    "tts/chatterbox": {"variants": ["default"], "format": "multi-component-safetensors+sidecars", "license": "mit"},
    "tts/kyutai-tts": {"variants": ["1.6b-en-fr"], "format": "safetensors+sentencepiece+voice-assets", "license": "cc-by-4.0; voices-review-required"},
    "tts/piper": {"variants": ["en_US-amy-medium", "en_US-danny-low", "en_GB-alba-medium"], "format": "onnx+json", "license": "mit"},
    "tts/melotts": {"variants": ["english-v3"], "format": "multi-component-safetensors+sidecars", "license": "mit"},
    "tts/spark-tts": {"variants": ["0.5b"], "format": "multi-component-safetensors+sidecars", "license": "cc-by-nc-sa-4.0", "upload_allowed": False},
    "tts/pocket-tts": {"variants": ["default"], "format": "safetensors+voice-assets+sidecars", "license": "cc-by-4.0"},
    "tts/styletts2": {"variants": ["libritts"], "format": "safetensors", "license": "unknown", "upload_allowed": False},
    "tts/zonos": {"variants": ["transformer", "hybrid"], "format": "multi-component-safetensors", "license": "multi-source-notices", "notes": "hybrid is advertised but not implemented by the engine"},
    "tts/gpt-sovits": {"variants": ["default"], "format": "multi-component-safetensors", "license": "mit"},
    "music/musicgen": {"variants": ["small", "medium", "large"], "format": "multi-component-safetensors", "license": "cc-by-nc-4.0", "upload_allowed": False},
    "music/audiogen": {"variants": ["medium"], "format": "multi-component-safetensors", "license": "cc-by-nc-4.0", "upload_allowed": False},
    "music/ace-step-1.5": {"variants": ["turbo", "turbo-shift1", "turbo-shift3", "turbo-continuous", "sft", "base", "xl-turbo", "xl-sft", "xl-base"], "format": "safetensors+shared-components+sidecars", "license": "mit"},
    "music/yue": {"variants": ["en-cot", "en-icl", "zh-cot", "zh-icl"], "format": "sharded-safetensors+shared-components+sidecars", "license": "apache-2.0"},
    "music/heartmula": {"variants": [f"{m}-{q}" for m in ["3b-hny", "3b-base", "3b-rl"] for q in ["bf16", "q8", "q4"]], "format": "bf16=safetensors; q8/q4=gguf", "license": "apache-2.0"},
    "music/stable-audio-open-small": {"variants": ["open-small"], "format": "multi-component-safetensors", "license": "custom/gated-review-required", "upload_allowed": False},
    "vc/rvc": {"variants": ["v2"], "format": "shared-safetensors+user-supplied-voice", "license": "shared-assets-mit; user-model-varies"},
    "vc/openvoice": {"variants": ["v2"], "format": "safetensors", "license": "mit"},
    "fx/demucs": {"variants": ["htdemucs", "htdemucs_6s"], "format": "safetensors", "license": "mit"},
    "fx/resemble-enhance": {"variants": ["denoise", "enhance"], "format": "alias:shared-safetensors", "license": "mit"},
}

# source relative to Models/audio, family, variant, destination subdirectory
CACHE_TREES = [
    ("stt/openai--whisper-base", "stt/whisper", "base", ""),
    ("stt/openai--whisper-base", "stt/whisper-streaming", "base", ""),
    ("stt/openai--whisper-large-v3", "stt/whisper", "large-v3", ""),
    ("stt/UsefulSensors--moonshine-base", "stt/moonshine", "base", ""),
    ("stt/UsefulSensors--moonshine-tiny", "stt/moonshine", "tiny", ""),
    ("stt/kyutai--stt-1b-en_fr-trfs", "stt/kyutai-stt", "1b-en-fr", ""),
    ("stt/kyutai--stt-1b-en_fr", "stt/kyutai-stt", "1b-en-fr", "tokenizer"),
    ("tts/hexgrad--Kokoro-82M", "tts/kokoro", "default", ""),
    ("tts/suno--bark", "tts/bark", "default", ""),
    ("tts/unsloth--orpheus-3b-0.1-ft", "tts/orpheus", "3b", "model"),
    ("tts/unsloth--csm-1b", "tts/csm", "1b", ""),
    ("tts/ResembleAI--chatterbox", "tts/chatterbox", "default", ""),
    ("tts/fishaudio--fish-speech-1.5", "tts/fish-speech", "fish-speech-1.5", "REVIEW_REQUIRED"),
    ("tts/google-bert--bert-base-multilingual-cased", "tts/bark", "default", "tokenizer"),
    ("tts/kyutai--tts-1.6b-en_fr", "tts/kyutai-tts", "1.6b-en-fr", ""),
    ("tts/kyutai--tts-voices", "tts/kyutai-tts", "1.6b-en-fr", "voices_REVIEW_REQUIRED"),
    ("tts/rhasspy--piper-voices/en/en_GB/alba/medium", "tts/piper", "en_GB-alba-medium", ""),
    ("tts/rhasspy--piper-voices/en/en_US/amy/medium", "tts/piper", "en_US-amy-medium", ""),
    ("tts/rhasspy--piper-voices/en/en_US/danny/low", "tts/piper", "en_US-danny-low", ""),
    ("music/FastVideo--stable-audio-open-small-Diffusers", "music/stable-audio-open-small", "open-small", "REVIEW_REQUIRED"),
    ("music/HeartMuLa--HeartCodec-oss-20260123", "music/heartmula", "_shared-heartcodec", ""),
    ("music/Qwen--Qwen3-Embedding-0.6B", "music/ace-step-1.5", "_shared", "qwen-embedding"),
    ("music/Comfy-Org--ace_step_1.5_ComfyUI_files/split_files/vae", "music/ace-step-1.5", "_shared", "vae"),
    ("music/YuE/en-cot", "music/yue", "en-cot", ""),
    ("music/YuE", "music/yue", "_shared", ""),
]

# Individual cache files whose source layout does not correspond one-to-one with
# the public family/variant layout.
CACHE_FILES = [
    ("acestep_v1_lyric_tokenizer.json", "music/ace-step-1.5/_shared/lyric_tokenizer.json"),
    ("cmudict.dict", "tts/kokoro/default/cmudict.dict"),
    ("music/AceStep/acestep-v15-turbo.safetensors", "music/ace-step-1.5/turbo/model.safetensors"),
    ("music/AceStep/acestep-v15-turbo.config.json", "music/ace-step-1.5/turbo/config.json"),
    ("music/AceStep/acestep-v15-turbo-shift1.config.json", "music/ace-step-1.5/turbo-shift1/config.json"),
    ("music/AceStep/acestep-v15-turbo-shift3.config.json", "music/ace-step-1.5/turbo-shift3/config.json"),
    ("music/AceStep/acestep-v15-turbo-continuous.config.json", "music/ace-step-1.5/turbo-continuous/config.json"),
    ("music/acestep/acestep-v15-sft.safetensors", "music/ace-step-1.5/sft/model.safetensors"),
    ("music/acestep/acestep-v15-sft.config.json", "music/ace-step-1.5/sft/config.json"),
    ("music/AceStep/acestep-v15-base.config.json", "music/ace-step-1.5/base/config.json"),
    ("tts/ResembleAI--chatterbox/s3gen.safetensors", "tts/cosyvoice2/2-0.5b/s3gen.safetensors"),
]

SKIP_SUFFIXES = {".pt", ".pth", ".bin", ".th", ".ckpt"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hardlink(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if os.path.samefile(source, destination):
            return "existing-hardlink"
        raise FileExistsError(f"Refusing to replace {destination}")
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        destination.symlink_to(source)
        return "symlink"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--hash", action="store_true", help="Hash staged files now (slow for large catalogs).")
    args = parser.parse_args()
    cache = args.cache.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    inventory: list[dict[str, Any]] = []

    for family, spec in CATALOG.items():
        for variant in spec["variants"]:
            folder = output / family / variant
            folder.mkdir(parents=True, exist_ok=True)
            plan = {
                "schema": 1,
                "id": f"{family}/{variant}",
                "target_format": spec["format"],
                "license": spec["license"],
                "upload_allowed": spec.get("upload_allowed", True),
                "notes": spec.get("notes"),
                "status": "awaiting-artifacts",
            }
            (folder / "artifact-plan.json").write_text(json.dumps(plan, indent=2) + "\n")

    for source_rel, family, variant, destination_rel in CACHE_TREES:
        source_root = cache / source_rel
        if not source_root.exists():
            continue
        destination_root = output / family / variant / destination_rel
        for source in sorted(source_root.rglob("*")):
            if not source.is_file() or ".cache" in source.parts or source.suffix.lower() in SKIP_SUFFIXES:
                continue
            relative = source.relative_to(source_root)
            destination = destination_root / relative
            mode = hardlink(source, destination)
            item = {
                "source": str(source), "destination": str(destination.relative_to(output)),
                "bytes": source.stat().st_size, "staging": mode,
            }
            if args.hash:
                item["sha256"] = sha256(source)
            inventory.append(item)

    for source_rel, destination_rel in CACHE_FILES:
        source = cache / source_rel
        if not source.is_file():
            continue
        destination = output / destination_rel
        mode = hardlink(source, destination)
        item = {
            "source": str(source), "destination": str(destination.relative_to(output)),
            "bytes": source.stat().st_size, "staging": mode,
        }
        if args.hash:
            item["sha256"] = sha256(source)
        inventory.append(item)

    (output / "catalog.json").write_text(json.dumps(CATALOG, indent=2) + "\n")
    (output / "staged-files.json").write_text(json.dumps(inventory, indent=2) + "\n")
    print(f"Prepared {len(CATALOG)} model families and {sum(len(x['variants']) for x in CATALOG.values())} variants")
    print(f"Staged {len(inventory)} cached files under {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
