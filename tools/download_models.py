#!/usr/bin/env python3
"""
SharpInference model downloader for RunPod / SwarmUI testing.

Downloads all models needed to validate SharpInference's 🔧 (implementation-done,
checkpoint-pending) pipelines. Drives SwarmUI's download WebSocket API so models
land in the correct sub-folders automatically.

Setup:
    pip install websockets

Usage:
    python download_models.py                         # download everything
    python download_models.py --dry-run               # print plan, download nothing
    python download_models.py --host http://myhost:7801
    python download_models.py --group main            # only main DiT checkpoints
    python download_models.py --group side            # only text-encoders + VAEs
    python download_models.py --filter chroma         # name contains "chroma" (case-insensitive)

NOTE: Lines marked  *** VERIFY URL ***  use best-guess filenames derived from
Comfy-Org naming conventions. Check the HuggingFace repo if a download 404s and
update the URL before re-running.

How it works:
  1. POST /API/GetNewSession  → session_id
  2. For each model: WS /API/DoModelDownloadWS  { session_id, url, type, name }
     - Progress messages stream back until { "success": true }
     - "Model at that save path already exists." → skipped (not an error)
     - Any other error → printed, script continues to next model
"""

import argparse
import asyncio
import json
import sys
import urllib.request
import urllib.error

try:
    import websockets
except ImportError:
    print("ERROR: 'websockets' not found. Run:  pip install websockets")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Model catalog
# ---------------------------------------------------------------------------

# Groups:
#   "main"   – primary DiT / UNet checkpoint the loader reads directly
#   "side"   – text encoders and VAEs auto-downloaded by the extension on first
#               load, but pre-fetching them avoids a long stall mid-generation

MODELS = [

    # ── Main checkpoints (type = "Stable-Diffusion") ──────────────────────

    {
        "group": "main",
        "display": "Chroma v1  (Flux-lineage T2I)",
        "url":  "https://huggingface.co/lodestones/Chroma/resolve/main/chroma_v1.safetensors",
        "type": "Stable-Diffusion",
        "name": "Chroma/chroma_v1",
        # Side models auto-downloaded: T5-XXL enconly, Flux.1 AE
    },
    {
        "group": "main",
        "display": "AuraFlow v0.3  (bundled Pile-T5-XL)",
        "url":  "https://huggingface.co/fal/AuraFlow-v0.3/resolve/main/aura_flow_0.3.safetensors",
        "type": "Stable-Diffusion",
        "name": "AuraFlow/aura_flow_0.3",
        # Side models auto-downloaded: SDXL VAE, Pile-T5-XL SentencePiece
    },
    {
        "group": "main",
        "display": "Anima 2B BF16  (Cosmos-Predict2-2B T2I)",
        "url":  "https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/diffusion_models/anima_2b_bf16.safetensors",
        "type": "Stable-Diffusion",
        "name": "Anima/anima_2b_bf16",
        "verify": True,  # *** VERIFY URL *** — check circlestone-labs/Anima on HF
        # Side models auto-downloaded: Qwen3-0.6B base, Qwen-Image VAE
    },
    {
        "group": "main",
        "display": "Lens BF16  (Microsoft Lens standard, 20-step)",
        "url":  "https://huggingface.co/Comfy-Org/Lens/resolve/main/lens_bf16.safetensors",
        "type": "Stable-Diffusion",
        "name": "Lens/lens_bf16",
        "verify": True,  # *** VERIFY URL *** — check Comfy-Org/Lens on HF
        # Side models auto-downloaded: GPT-OSS-20B NVFP4, Flux.2 VAE
    },
    {
        "group": "main",
        "display": "Lens-Turbo BF16  (Microsoft Lens turbo, 4-step)",
        "url":  "https://huggingface.co/Comfy-Org/Lens/resolve/main/lens_turbo_bf16.safetensors",
        "type": "Stable-Diffusion",
        "name": "Lens/lens_turbo_bf16",
        "verify": True,  # *** VERIFY URL *** — check Comfy-Org/Lens on HF
        # Side models auto-downloaded: GPT-OSS-20B NVFP4, Flux.2 VAE (shared with Lens)
    },
    {
        "group": "main",
        "display": "Wan 2.2 TI2V-5B FP16  (text-to-video)",
        "url":  "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors",
        "type": "Stable-Diffusion",
        "name": "Wan/wan2.2_ti2v_5B_fp16",
        "verify": True,  # *** VERIFY URL *** — check Comfy-Org/Wan_2.2_ComfyUI_Repackaged on HF
        # Side models auto-downloaded: umT5-XXL fp8, Wan 2.2 VAE
    },
    {
        "group": "main",
        "display": "LTX-Video 2B v0.9  (single-file DiT + VAE)",
        "url":  "https://huggingface.co/Lightricks/LTX-Video/resolve/main/ltx-video-2b-v0.9.safetensors",
        "type": "Stable-Diffusion",
        "name": "LTXV/ltx-video-2b-v0.9",
        # Single-file: no separate side-model downloads needed
    },
    {
        "group": "main",
        "display": "ACE-Step v1 3.5B DiT  (music generation)",
        "url":  "https://huggingface.co/ACE-Step/ACE-Step-v1-3.5B/resolve/main/ace_step_transformer/diffusion_pytorch_model.safetensors",
        "type": "Stable-Diffusion",
        "name": "AceStep/ace_step_v1",
        # Side models auto-downloaded: umT5-base, Music-DCAE, Music Vocoder
    },

    # ── Text encoders (type = "Clip") ──────────────────────────────────────
    # These are auto-downloaded by each loader on first use. Pre-fetching them
    # here avoids long stalls during the first generation run.

    {
        "group": "side",
        "display": "T5-XXL encoder-only FP8  (Chroma / SD3 / Flux)",
        "url":  "https://huggingface.co/mcmonkey/google_t5-v1_1-xxl_encoderonly/resolve/main/t5xxl_fp8_e4m3fn.safetensors",
        "type": "Clip",
        "name": "t5xxl_enconly",
    },
    {
        "group": "side",
        "display": "umT5-XXL FP8 scaled  (Wan video)",
        "url":  "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        "type": "Clip",
        "name": "umt5_xxl_fp8_e4m3fn_scaled",
    },
    {
        "group": "side",
        "display": "umT5-base  (ACE-Step v1 style encoder)",
        "url":  "https://huggingface.co/ACE-Step/ACE-Step-v1-3.5B/resolve/main/umt5-base/model.safetensors",
        "type": "Clip",
        "name": "umt5_base",
    },
    {
        "group": "side",
        "display": "Qwen3-0.6B Base  (Anima text encoder)",
        "url":  "https://huggingface.co/circlestone-labs/Anima/resolve/main/split_files/text_encoders/qwen_3_06b_base.safetensors",
        "type": "Clip",
        "name": "qwen_3_600m",
    },
    {
        "group": "side",
        "display": "GPT-OSS-20B NVFP4  (Lens text encoder, ~13 GB)",
        "url":  "https://huggingface.co/Comfy-Org/Lens/resolve/main/text_encoders/gpt_oss_20b_nvfp4.safetensors",
        "type": "Clip",
        "name": "gpt_oss_20b_nvfp4",
    },
    {
        "group": "side",
        "display": "CLIP-L FP16  (SDXL / SD3 / Flux / AuraFlow)",
        "url":  "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/text_encoder/model.fp16.safetensors",
        "type": "Clip",
        "name": "clip_l",
    },
    {
        "group": "side",
        "display": "CLIP-G FP16  (SDXL / SD3 / AuraFlow)",
        "url":  "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/text_encoder_2/model.fp16.safetensors",
        "type": "Clip",
        "name": "clip_g",
    },

    # ── VAEs (type = "VAE") ────────────────────────────────────────────────

    {
        "group": "side",
        "display": "Flux.1 AE  (Chroma / Z-Image / Flux)",
        "url":  "https://huggingface.co/mcmonkey/swarm-vaes/resolve/main/flux_ae.safetensors",
        "type": "VAE",
        "name": "Flux/ae",
    },
    {
        "group": "side",
        "display": "Flux.2 VAE  (Lens / Ideogram 4)",
        "url":  "https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors",
        "type": "VAE",
        "name": "Flux2/flux2-vae",
    },
    {
        "group": "side",
        "display": "SDXL VAE fp16-fix  (AuraFlow / SDXL refinement)",
        "url":  "https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl_vae.safetensors",
        "type": "VAE",
        "name": "OfficialStableDiffusion/sdxl_vae",
    },
    {
        "group": "side",
        "display": "Wan 2.2 VAE  (Wan video / Matrix-Game)",
        "url":  "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan2.2_vae.safetensors",
        "type": "VAE",
        "name": "Wan/wan2.2_vae",
    },
    {
        "group": "side",
        "display": "ACE-Step Music-DCAE  (mel autoencoder)",
        "url":  "https://huggingface.co/ACE-Step/ACE-Step-v1-3.5B/resolve/main/music_dcae_f8c8/diffusion_pytorch_model.safetensors",
        "type": "VAE",
        "name": "AceStep/music_dcae_f8c8",
    },
    {
        "group": "side",
        "display": "ACE-Step Music Vocoder  (HiFi-GAN mel → 44.1 kHz)",
        "url":  "https://huggingface.co/ACE-Step/ACE-Step-v1-3.5B/resolve/main/music_vocoder/diffusion_pytorch_model.safetensors",
        "type": "VAE",
        "name": "AceStep/music_vocoder",
    },
    {
        "group": "side",
        "display": "Qwen-Image VAE  (Anima / Qwen-Image)",
        "url":  "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors",
        "type": "VAE",
        "name": "QwenImage/qwen_image_vae",
    },

]


# ---------------------------------------------------------------------------
# Session acquisition
# ---------------------------------------------------------------------------

def get_session(host: str) -> str:
    url = f"{host}/API/GetNewSession"
    req = urllib.request.Request(
        url,
        data=b"{}",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
    except urllib.error.URLError as e:
        print(f"ERROR: Cannot reach SwarmUI at {host}: {e}")
        print("Is Swarm running? Check the --host argument.")
        sys.exit(1)
    if "error" in data:
        print(f"ERROR from GetNewSession: {data['error']}")
        sys.exit(1)
    return data["session_id"]


# ---------------------------------------------------------------------------
# WebSocket download
# ---------------------------------------------------------------------------

async def download_one(ws_host: str, session_id: str, model: dict, dry_run: bool) -> str:
    """Return 'ok', 'skipped', or an error message string."""
    display = model["display"]
    url     = model["url"]
    mtype   = model["type"]
    name    = model["name"]

    verify_note = "  *** VERIFY URL ***" if model.get("verify") else ""
    print(f"\n[{mtype}] {display}{verify_note}")
    print(f"   → {url}")
    print(f"   → save as: {name}.safetensors")

    if dry_run:
        return "dry-run"

    ws_url = f"{ws_host}/API/DoModelDownloadWS"
    payload = json.dumps({
        "session_id": session_id,
        "url":        url,
        "type":       mtype,
        "name":       name,
    })

    try:
        async with websockets.connect(ws_url, open_timeout=30) as ws:
            await ws.send(payload)
            last_pct = -1
            async for raw in ws:
                msg = json.loads(raw)
                if msg.get("error"):
                    err = msg["error"]
                    if "already exists" in err:
                        print(f"   ✓ already downloaded — skipped")
                        return "skipped"
                    print(f"   ✗ error: {err}")
                    return f"error: {err}"
                if "current_percent" in msg:
                    pct = int(msg["current_percent"] * 100)
                    if pct != last_pct and pct % 5 == 0:
                        mbps = msg.get("per_second", 0) / (1024 * 1024)
                        print(f"   {pct:3d}%  ({mbps:.1f} MB/s)", end="\r", flush=True)
                        last_pct = pct
                if msg.get("success"):
                    print(f"   100%  done                    ")
                    return "ok"
    except Exception as e:
        return f"error: {e}"

    return "error: connection closed without success"


async def run(args):
    host    = args.host.rstrip("/")
    ws_host = host.replace("http://", "ws://").replace("https://", "wss://")
    dry_run = args.dry_run

    # Filter models
    selected = [m for m in MODELS
                if (not args.group  or m["group"] == args.group)
                and (not args.filter or args.filter.lower() in m["display"].lower())]

    if not selected:
        print("No models matched the filter — nothing to do.")
        return

    verify_count = sum(1 for m in selected if m.get("verify"))
    if verify_count and not dry_run:
        print(f"\nWARNING: {verify_count} model(s) have unverified URLs (marked *** VERIFY URL ***).")
        print("Check those HuggingFace repos before running — or use --dry-run first.")
        answer = input("Continue anyway? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted.")
            return

    if not dry_run:
        print(f"\nConnecting to {host} ...")
        session_id = get_session(host)
        print(f"Session: {session_id[:8]}...")
    else:
        session_id = "dry-run"
        print(f"\nDRY RUN — would connect to {host}")

    print(f"\n{len(selected)} model(s) to download\n{'─' * 60}")

    results = {"ok": [], "skipped": [], "dry-run": [], "error": []}

    for model in selected:
        status = await download_one(ws_host, session_id, model, dry_run)
        key = status if status in results else "error"
        results[key].append(model["display"])

    print(f"\n{'─' * 60}")
    print(f"Done.")
    if results["ok"]:
        print(f"  Downloaded : {len(results['ok'])}")
    if results["skipped"]:
        print(f"  Skipped    : {len(results['skipped'])}  (already on disk)")
    if results["dry-run"]:
        print(f"  Dry-run    : {len(results['dry-run'])}")
    if results["error"]:
        print(f"  Errors     : {len(results['error'])}")
        for name in results["error"]:
            print(f"    • {name}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host",    default="http://localhost:7801",
                        help="SwarmUI base URL (default: http://localhost:7801)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the download plan without actually downloading")
    parser.add_argument("--group",   choices=["main", "side"],
                        help="Only download 'main' checkpoints or 'side' models")
    parser.add_argument("--filter",  metavar="TEXT",
                        help="Only download models whose display name contains TEXT")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
