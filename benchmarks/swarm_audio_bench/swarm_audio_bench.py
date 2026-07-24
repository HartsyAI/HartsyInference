#!/usr/bin/env python3
"""Swarm-path audio benchmark: exercises every local (non-cloud) AudioLab provider through the real
SwarmUI API (ProcessTTS / ProcessSTT / ProcessAudio), same spirit as benchmarks/swarm_llm_bench but for
audio. Each model is tried independently (a crash on one must not abort the rest). Records wall time,
returned audio duration (for RTF), and any error verbatim so failures are diagnosable, not just "failed".

Usage: python3 swarm_audio_bench.py [--host HOST] [--port PORT] [--out results.json]
"""
import argparse
import base64
import json
import time
import wave
import io
from pathlib import Path

import requests

JFK_PATH = Path(__file__).resolve().parents[2] / "tests/python-reference/silerovad_reference/jfk.wav"
JFK_TRANSCRIPT = ("And so, my fellow Americans, ask not what your country can do for you, "
                   "ask what you can do for your country.")
TTS_TEXT = "Hello, this is a test of the text to speech system."
MUSIC_PROMPT = "An upbeat electronic dance track with synths and a steady beat"


def b64_of(path):
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def wav_duration_from_b64(b64_data):
    try:
        raw = base64.b64decode(b64_data)
        with wave.open(io.BytesIO(raw), "rb") as w:
            return w.getnframes() / w.getframerate()
    except Exception:
        return None


# provider_id -> (category, args_builder)
TTS_IDS = [
    "piper_tts", "kokoro_tts", "bark_tts", "styletts2_tts", "sparktts_tts", "cosyvoice_tts",
    "vibevoice_tts", "fishspeech_tts", "f5_tts", "dia_tts", "orpheus_tts", "csm_tts", "neutts_tts",
    "qwen3_tts", "chatterbox_tts", "kyutaitts_tts", "melotts_tts", "pockettts_tts", "zonos_tts",
    "gptsovits_clone", "zipvoice_tts",
]
STT_IDS = [
    "whisper_stt", "moonshine_stt", "distilwhisper_stt", "moonshinestreaming_stt",
    "kyutaistt_stt", "whisperstreaming_stt",
]
MUSIC_IDS = ["musicgen_music", "audiogen_sfx", "acestep_music", "yue_music", "stableaudio_music", "heartlib_music"]
VC_IDS = ["openvoice_clone", "rvc_clone"]
FX_IDS = ["demucs_fx", "resemble_enhance_fx"]


def build_tts_args(jfk_b64):
    return {
        "text": TTS_TEXT,
        "voice": "default",
        "language": "en",
        "volume": 1.0,
        "reference_audio": jfk_b64,
        "ref_text": JFK_TRANSCRIPT,
    }


def build_stt_args(jfk_b64):
    return {"audio_data": jfk_b64, "language": "en"}


def build_music_args():
    return {"prompt": MUSIC_PROMPT, "duration": 10, "task_type": "text2music"}


def build_vc_args(jfk_b64):
    return {"source_audio": jfk_b64, "target_voice": jfk_b64, "pitch_shift": 0}


def build_fx_args(jfk_b64, provider_id):
    args = {"audio_data": jfk_b64}
    if provider_id == "demucs_fx":
        args["model_name"] = "htdemucs"
    return args


def call(session_id, base_url, endpoint, provider_id, args, timeout):
    payload = {"session_id": session_id, "provider_id": provider_id, "args": args}
    t0 = time.time()
    resp = requests.post(f"{base_url}/API/{endpoint}", json=payload, timeout=timeout)
    elapsed = time.time() - t0
    resp.raise_for_status()
    data = resp.json()
    return data, elapsed


def call_stt(session_id, base_url, provider_id, args, timeout):
    payload = {"session_id": session_id, "provider_id": provider_id, **args}
    t0 = time.time()
    resp = requests.post(f"{base_url}/API/ProcessSTT", json=payload, timeout=timeout)
    elapsed = time.time() - t0
    resp.raise_for_status()
    return resp.json(), elapsed


def call_tts(session_id, base_url, provider_id, args, timeout):
    payload = {"session_id": session_id, "provider_id": provider_id, **args}
    t0 = time.time()
    resp = requests.post(f"{base_url}/API/ProcessTTS", json=payload, timeout=timeout)
    elapsed = time.time() - t0
    resp.raise_for_status()
    return resp.json(), elapsed


def try_install(session_id, base_url, provider_id):
    try:
        requests.post(f"{base_url}/API/AudioLabInstallEngine",
                      json={"session_id": session_id, "provider_id": provider_id}, timeout=120)
    except Exception:
        pass


def run_one(session_id, base_url, provider_id, category, jfk_b64, timeout):
    result = {"provider_id": provider_id, "category": category}
    try:
        if category == "TTS":
            args = build_tts_args(jfk_b64)
            data, elapsed = call_tts(session_id, base_url, provider_id, args, timeout)
        elif category == "STT":
            args = build_stt_args(jfk_b64)
            data, elapsed = call_stt(session_id, base_url, provider_id, args, timeout)
        elif category == "Music":
            args = build_music_args()
            data, elapsed = call(session_id, base_url, "ProcessAudio", provider_id, args, timeout)
        elif category == "VoiceConversion":
            args = build_vc_args(jfk_b64)
            data, elapsed = call(session_id, base_url, "ProcessAudio", provider_id, args, timeout)
        elif category == "Fx":
            args = build_fx_args(jfk_b64, provider_id)
            data, elapsed = call(session_id, base_url, "ProcessAudio", provider_id, args, timeout)
        else:
            result["status"] = "SKIP"
            result["error"] = f"unknown category {category}"
            return result

        result["wall_time_s"] = round(elapsed, 2)
        success = data.get("success", False)
        if success:
            result["status"] = "OK"
            audio_dur = data.get("duration") or wav_duration_from_b64(data.get("audio_data", ""))
            if audio_dur:
                result["audio_duration_s"] = round(audio_dur, 2)
                result["rtf"] = round(elapsed / audio_dur, 3) if audio_dur > 0 else None
            if category == "STT":
                result["transcription"] = data.get("transcription") or data.get("text")
        else:
            result["status"] = "FAIL"
            result["error"] = str(data.get("error") or data.get("error_id") or data)[:300]
    except requests.exceptions.Timeout:
        result["status"] = "TIMEOUT"
        result["error"] = f"exceeded {timeout}s"
    except Exception as e:
        result["status"] = "ERROR"
        result["error"] = str(e)[:300]
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="192.168.10.188")
    ap.add_argument("--port", default="7801")
    ap.add_argument("--out", default=str(Path(__file__).parent / "swarm_audio_results.json"))
    ap.add_argument("--timeout", type=float, default=240.0)
    ap.add_argument("--categories", default="TTS,STT,Music,VoiceConversion,Fx")
    args = ap.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    sess = requests.post(f"{base_url}/API/GetNewSession", json={}, timeout=30).json()
    session_id = sess["session_id"]
    print(f"Session: {session_id[:16]}...")

    jfk_b64 = b64_of(JFK_PATH)

    plan = []
    cats = args.categories.split(",")
    if "TTS" in cats:
        plan += [(pid, "TTS") for pid in TTS_IDS]
    if "STT" in cats:
        plan += [(pid, "STT") for pid in STT_IDS]
    if "Music" in cats:
        plan += [(pid, "Music") for pid in MUSIC_IDS]
    if "VoiceConversion" in cats:
        plan += [(pid, "VoiceConversion") for pid in VC_IDS]
    if "Fx" in cats:
        plan += [(pid, "Fx") for pid in FX_IDS]

    results = []
    for provider_id, category in plan:
        print(f"[{category}] {provider_id} ...", end=" ", flush=True)
        r = run_one(session_id, base_url, provider_id, category, jfk_b64, args.timeout)
        results.append(r)
        if r["status"] == "OK":
            rtf = r.get("rtf")
            print(f"OK  {r.get('wall_time_s')}s  rtf={rtf}")
        else:
            print(f"{r['status']}: {r.get('error', '')[:120]}")

    Path(args.out).write_text(json.dumps({"results": results}, indent=2))
    ok = sum(1 for r in results if r["status"] == "OK")
    print(f"\n{ok}/{len(results)} succeeded. Results -> {args.out}")


if __name__ == "__main__":
    main()
