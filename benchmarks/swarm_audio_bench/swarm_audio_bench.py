#!/usr/bin/env python3
"""Swarm-path audio benchmark: exercises every local (non-cloud) AudioLab provider through the real
SwarmUI API (ProcessTTS / ProcessSTT / ProcessAudio), same spirit as benchmarks/swarm_llm_bench but for
audio. Each model is tried independently (a crash on one must not abort the rest). Records wall time,
returned audio duration (for RTF), and any error verbatim so failures are diagnosable, not just "failed".

Usage: python3 swarm_audio_bench.py [--host HOST] [--port PORT] [--out results.json]

--voice mode (voice-turn latency, Phase 0 of the phone-call plan — bounding context, NOT a gate input):
  STT  whisper_stt on JFK slices of 2 / 5 / 10 s at 16 kHz, plus an 8 kHz -> 16 kHz narrowband round-trip
       variant of each slice; TTS kokoro_tts on 5 / 15 / 30-word sentences. 2 warm-up + 5 timed calls each,
       median / p95 / min of the HTTP wall clock. That wall clock INCLUDES the HTTP round trip and the
       JSON/base64 payload parse on both sides (request base64 is built before the timer starts); the STT
       response's own `processing_time` (server-side, inside AudioLab) is reported beside it so the transport
       share is visible. The Whisper variant AudioLab resolved is taken from Swarm's log lines emitted during the
       run (ListRecentLogMessages + the swarmui.service journal); when neither names a checkpoint it is reported
       as "unknown" — never guessed. The card each request used is attributed by sampling
       `nvidia-smi --query-compute-apps` by GPU UUID during the call (used_memory delta / new SwarmUI row).

  Narrowband method (pure Python — this host has no numpy): 63-tap Hamming-windowed sinc low-pass, cutoff
  4 kHz (0.25 cycles/sample at 16 kHz), decimate x2 to 8 kHz, then zero-stuff x2 and apply the same low-pass
  with gain 2 back to 16 kHz. This is NOT the engine's polyphase Resampler (which Phase 1 uses), so narrowband
  numbers are not comparable across phases.

  Usage: python3 swarm_audio_bench.py --voice [--voice-whisper-model small] [--voice-tts-voice af_heart]
                                      [--voice-out swarm_voice_results.json]

  The between-call idle check reads GetGlobalStatus (every session's queue). Afterwards,
  `tests/swarm-quiet-window.sh --verify-since <start>` lists this run's own 21 `ProcessTTS: requested` lines
  (STT calls write none) and exits 4 on them; any T2I `requested` line or extra TTS line is someone else's.
"""
import argparse
import base64
import datetime
import json
import math
import re
import struct
import subprocess
import statistics
import threading
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

# ---- --voice protocol (frozen; separate from the catalog sweep above) ---------------------------------------
VOICE_WARM = 2
VOICE_TIMED = 5
VOICE_STT_SECONDS = [2, 5, 10]
VOICE_TTS_SENTENCES = {
    5: "Please hold while I check.",
    15: "Thanks for calling, I can see your appointment is booked for Tuesday afternoon at three.",
    30: ("I have updated the delivery address on your order, the driver will call you when they are ten minutes "
         "away, and you will receive a message with the tracking link."),
}
VOICE_STT_PROVIDER = "whisper_stt"
VOICE_TTS_PROVIDER = "kokoro_tts"
NARROWBAND_TAPS = 63
GPU_SAMPLE_INTERVAL_S = 0.15


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


# =============================================================================================================
# --voice mode helpers
# =============================================================================================================

def read_wav_mono_float(path):
    """16-bit PCM WAV -> (rate, list of floats in [-1, 1]); channels are averaged."""
    with wave.open(str(path), "rb") as w:
        rate, ch, width, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    if width != 2:
        raise ValueError(f"{path}: expected 16-bit PCM, got {8 * width}-bit")
    ints = struct.unpack(f"<{n * ch}h", raw)
    if ch == 1:
        return rate, [v / 32768.0 for v in ints]
    return rate, [sum(ints[i * ch:(i + 1) * ch]) / (ch * 32768.0) for i in range(n)]


def wav_b64_from_float(samples, rate):
    clipped = [max(-32768, min(32767, int(round(v * 32767.0)))) for v in samples]
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(struct.pack(f"<{len(clipped)}h", *clipped))
    return base64.b64encode(buf.getvalue()).decode("ascii")


def lowpass_taps(cutoff_cycles_per_sample, n_taps):
    """Hamming-windowed sinc low-pass, unity DC gain."""
    m = n_taps - 1
    taps = []
    for i in range(n_taps):
        x = i - m / 2.0
        s = 2 * cutoff_cycles_per_sample if x == 0 else math.sin(2 * math.pi * cutoff_cycles_per_sample * x) / (math.pi * x)
        w = 0.54 - 0.46 * math.cos(2 * math.pi * i / m)
        taps.append(s * w)
    g = sum(taps)
    return [t / g for t in taps]


def narrowband_roundtrip(samples, rate):
    """16 kHz -> 8 kHz -> 16 kHz telephone-band simulation; see the module doc for the filter."""
    if rate != 16000:
        raise ValueError("narrowband_roundtrip expects 16 kHz input")
    taps = lowpass_taps(0.25, NARROWBAND_TAPS)
    half = (NARROWBAND_TAPS - 1) // 2
    n = len(samples)
    # Anti-alias + decimate: evaluate the centred convolution only at even output positions.
    down = []
    for m in range(0, n, 2):
        acc = 0.0
        for k, t in enumerate(taps):
            j = m + half - k
            if 0 <= j < n:
                acc += t * samples[j]
        down.append(acc)
    # Zero-stuff x2 + interpolation low-pass with gain 2; only even stuffed indices are non-zero.
    up_len = len(down) * 2
    up = []
    for i in range(up_len):
        acc = 0.0
        for k, t in enumerate(taps):
            j = i + half - k
            if 0 <= j < up_len and (j & 1) == 0:
                acc += t * down[j >> 1]
        up.append(2.0 * acc)
    return up[:n]


def slice_seconds(samples, rate, seconds):
    return samples[:int(rate * seconds)]


def percentile(values, p):
    """Linear interpolation between order statistics (numpy 'linear' method)."""
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    pos = p * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def summarize(values):
    return {"median": statistics.median(values), "p95": percentile(values, 0.95), "min": min(values), "n": len(values)}


def nvidia_gpus():
    out = subprocess.run(["nvidia-smi", "--query-gpu=uuid,name,index,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True, check=True).stdout
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        gpus.append({"uuid": parts[0], "name": parts[1], "index": int(parts[2]), "driver": parts[3]})
    return gpus


def nvidia_compute_apps():
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    rows = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4 and parts[0]:
            try:
                mem = int(parts[3])
            except ValueError:
                mem = 0
            rows.append({"uuid": parts[0], "pid": parts[1], "name": parts[2], "mem": mem})
    return rows


def swarm_pid():
    try:
        pid = subprocess.run(["systemctl", "--user", "show", "-p", "MainPID", "--value", "swarmui.service"],
                             capture_output=True, text=True).stdout.strip()
        return pid if pid and pid != "0" else None
    except Exception:
        return None


def is_swarm_row(row, spid):
    return row["pid"] == spid or "swarmui" in row["name"].lower()


class GpuSampler:
    """Samples compute-apps by UUID while one request is in flight. Attribution: the card whose SwarmUI row
    grew the most (>= 32 MiB over the pre-call baseline) or gained a new SwarmUI row. Foreign rows (any pid
    that is not SwarmUI) are collected so the caller can abort — a foreign process mid-run breaks the
    "no timed batch beside another tenant" rule."""

    def __init__(self, spid):
        self.spid = spid
        self.baseline = {}
        self.peak = {}
        self.foreign = set()
        self._stop = threading.Event()
        self._thread = None

    def _snapshot(self, into):
        for row in nvidia_compute_apps():
            if is_swarm_row(row, self.spid):
                into[row["uuid"]] = max(into.get(row["uuid"], 0), row["mem"])
            else:
                self.foreign.add((row["uuid"], row["pid"], row["name"]))

    def __enter__(self):
        self.baseline = {}
        self.peak = {}
        self._snapshot(self.baseline)
        self.peak = dict(self.baseline)
        self._stop.clear()

        def loop():
            while not self._stop.is_set():
                self._snapshot(self.peak)
                self._stop.wait(GPU_SAMPLE_INTERVAL_S)

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=5)
        self._snapshot(self.peak)

    def attribution(self):
        best_uuid, best_delta = None, 0
        for uuid, peak in self.peak.items():
            delta = peak - self.baseline.get(uuid, 0)
            if uuid not in self.baseline:
                delta = max(delta, 32)   # a new SwarmUI row on a card is attribution by itself
            if delta > best_delta:
                best_uuid, best_delta = uuid, delta
        if best_uuid is None or best_delta < 32:
            return None, 0
        return best_uuid, best_delta


def swarm_status(session_id, base_url):
    # GetGlobalStatus sums every session; GetCurrentStatus counts only this session's own gens.
    r = requests.post(f"{base_url}/API/GetGlobalStatus", json={"session_id": session_id}, timeout=20)
    r.raise_for_status()
    return r.json().get("status", {})


def assert_swarm_idle(session_id, base_url, when):
    s = swarm_status(session_id, base_url)
    busy = {k: s.get(k) for k in ("live_gens", "waiting_gens", "loading_models") if s.get(k)}
    if busy:
        raise RuntimeError(f"foreign Swarm activity {when}: {busy} — aborting the timed batch")
    return s


def log_cursor(session_id, base_url, types):
    """Per-type last sequence id so a later fetch returns only lines emitted after this point.
    ListRecentLogMessages throws NRE without `last_sequence_ids`, so an empty map is always sent."""
    try:
        r = requests.post(f"{base_url}/API/ListRecentLogMessages",
                          json={"session_id": session_id, "types": types, "last_sequence_ids": {}}, timeout=20)
        d = r.json()
        cursor = {}
        for t in types:
            msgs = (d.get("data") or {}).get(t) or []
            cursor[t] = max((m.get("sequence_id", 0) for m in msgs), default=d.get("last_sequence_id") or 0)
        return cursor
    except Exception as e:
        print(f"  (log cursor unavailable: {e})")
        return None


def log_lines_since(session_id, base_url, types, cursor):
    if cursor is None:
        return []
    try:
        r = requests.post(f"{base_url}/API/ListRecentLogMessages",
                          json={"session_id": session_id, "types": types, "last_sequence_ids": cursor}, timeout=20)
        d = r.json()
        lines = []
        for t, msgs in (d.get("data") or {}).items():
            for m in msgs:
                lines.append(f"[{t}] {m.get('time', '')} {m.get('message', '')}")
        return lines
    except Exception as e:
        print(f"  (log fetch unavailable: {e})")
        return []


def journal_lines_since(start_utc):
    try:
        out = subprocess.run(["journalctl", "--user", "-u", "swarmui.service", "--no-pager", "-o", "cat",
                              "--since", start_utc.strftime("%Y-%m-%d %H:%M:%S UTC")],
                             capture_output=True, text=True, timeout=60).stdout
        return out.splitlines()
    except Exception as e:
        print(f"  (journal unavailable: {e})")
        return []


WHISPER_PATTERNS = [
    re.compile(r"openai/whisper-[A-Za-z0-9.\-]+"),
    re.compile(r"distil-whisper/[A-Za-z0-9.\-]+"),
    re.compile(r"whisper-[a-z0-9.\-]+_fp\d+\.safetensors", re.I),
    re.compile(r"whisper:[a-z0-9.\-/]+", re.I),
]


def resolve_whisper_variant(lines):
    hits = []
    for line in lines:
        if "whisper" not in line.lower():
            continue
        for pat in WHISPER_PATTERNS:
            for m in pat.findall(line):
                if m not in hits:
                    hits.append(m)
    return hits


def timed_calls(label, make_call, session_id, base_url, sampler_factory):
    """VOICE_WARM warm-ups then VOICE_TIMED timed calls; per call: HTTP wall, server processing_time (if any),
    card attribution by UUID. Aborts on foreign Swarm activity or a foreign GPU process."""
    walls, server_times, cards, payloads = [], [], [], []
    for i in range(VOICE_WARM + VOICE_TIMED):
        phase = "warm" if i < VOICE_WARM else "timed"
        assert_swarm_idle(session_id, base_url, f"before {label} call {i + 1}")
        with sampler_factory() as sampler:
            data, wall = make_call()
        if sampler.foreign:
            raise RuntimeError(f"foreign GPU process during {label}: {sorted(sampler.foreign)} — aborting")
        if not data.get("success", False):
            raise RuntimeError(f"{label} call {i + 1} failed: {str(data.get('error') or data)[:300]}")
        uuid, delta = sampler.attribution()
        print(f"    {label} {phase} {i + 1}: wall {wall * 1000:7.1f} ms"
              + (f"  server {float(data['processing_time']) * 1000:7.1f} ms" if data.get("processing_time") is not None else "")
              + f"  card={uuid or 'unattributed'}(+{delta} MiB)")
        if phase == "timed":
            walls.append(wall)
            if data.get("processing_time") is not None:
                server_times.append(float(data["processing_time"]))
            cards.append(uuid or "unattributed")
            payloads.append(data)
    return walls, server_times, cards, payloads


def run_voice(args):
    base_url = f"http://{args.host}:{args.port}"
    sess = requests.post(f"{base_url}/API/GetNewSession", json={}, timeout=30).json()
    session_id = sess["session_id"]
    start_utc = datetime.datetime.now(datetime.timezone.utc)
    gpus = nvidia_gpus()
    spid = swarm_pid()
    print(f"Session: {session_id[:16]}...  start {start_utc.isoformat()}")
    print(f"GPUs: " + "; ".join(f"[{g['index']}] {g['name']} {g['uuid']}" for g in gpus) + f"  driver {gpus[0]['driver']}")
    print(f"swarmui.service MainPID: {spid}")
    status0 = assert_swarm_idle(session_id, base_url, "at start")
    print(f"Swarm status at start: {json.dumps(status0)}")
    log_types = ["Info", "Debug", "Warning", "Error"]
    cursor = log_cursor(session_id, base_url, log_types)

    rate, jfk = read_wav_mono_float(JFK_PATH)
    if rate != 16000:
        raise RuntimeError(f"{JFK_PATH} is {rate} Hz, expected 16 kHz")
    t = time.time()
    jfk_nb = narrowband_roundtrip(jfk, rate)
    print(f"narrowband round-trip of the {len(jfk) / rate:.1f}s clip built in {time.time() - t:.1f}s (pure Python FIR)")

    def sampler_factory():
        return GpuSampler(spid)

    results = {
        "mode": "voice",
        "protocol": {"warm": VOICE_WARM, "timed": VOICE_TIMED, "stt_seconds": VOICE_STT_SECONDS,
                     "tts_words": sorted(VOICE_TTS_SENTENCES), "wall_includes": "HTTP round trip + JSON/base64 parse",
                     "narrowband": f"pure-Python {NARROWBAND_TAPS}-tap Hamming sinc, 4 kHz cutoff, x2 decimate/zero-stuff"},
        "host": base_url, "started_utc": start_utc.isoformat(), "gpus": gpus, "swarm_pid": spid,
        "swarm_status_start": status0, "stt": [], "tts": [],
    }

    # ---- STT ----------------------------------------------------------------------------------------------
    stt_model = args.voice_whisper_model
    for seconds in VOICE_STT_SECONDS:
        for variant, src in (("16k", jfk), ("narrowband 8k->16k", jfk_nb)):
            clip = slice_seconds(src, rate, seconds)
            b64 = wav_b64_from_float(clip, rate)
            label = f"STT {seconds}s {variant}"
            print(f"[{label}] {len(clip) / rate:.2f}s audio, {len(b64) // 1024} KiB base64, model={stt_model}")
            stt_args = {"audio_data": b64, "language": "en"}
            if stt_model:
                stt_args["model"] = stt_model
            walls, server, cards, payloads = timed_calls(
                label, lambda: call_stt(session_id, base_url, VOICE_STT_PROVIDER, stt_args, args.timeout),
                session_id, base_url, sampler_factory)
            row = {"seconds": seconds, "variant": variant, "requested_model": stt_model,
                   "wall_s": summarize(walls), "server_processing_s": summarize(server) if server else None,
                   "cards": cards, "transcription": payloads[-1].get("transcription")}
            results["stt"].append(row)
            print(f"  -> wall median {row['wall_s']['median'] * 1000:.0f} ms  p95 {row['wall_s']['p95'] * 1000:.0f}  "
                  f"min {row['wall_s']['min'] * 1000:.0f} | text: {row['transcription']!r}")

    # ---- TTS ----------------------------------------------------------------------------------------------
    tts_voice = args.voice_tts_voice
    for words in sorted(VOICE_TTS_SENTENCES):
        text = VOICE_TTS_SENTENCES[words]
        assert len(text.split()) == words, f"sentence for {words} words has {len(text.split())}"
        label = f"TTS {words}w"
        print(f"[{label}] voice={tts_voice}: {text!r}")
        tts_args = {"text": text, "voice": tts_voice, "language": "en"}
        try:
            walls, server, cards, payloads = timed_calls(
                label, lambda: call_tts(session_id, base_url, VOICE_TTS_PROVIDER, tts_args, args.timeout),
                session_id, base_url, sampler_factory)
        except RuntimeError as e:
            if tts_voice != "default" and "failed" in str(e):
                print(f"  voice {tts_voice!r} rejected ({e}); retrying with 'default'")
                tts_voice = "default"
                tts_args["voice"] = tts_voice
                walls, server, cards, payloads = timed_calls(
                    label, lambda: call_tts(session_id, base_url, VOICE_TTS_PROVIDER, tts_args, args.timeout),
                    session_id, base_url, sampler_factory)
            else:
                raise
        dur = payloads[-1].get("duration") or wav_duration_from_b64(payloads[-1].get("audio_data", ""))
        row = {"words": words, "text": text, "voice": tts_voice, "wall_s": summarize(walls),
               "audio_seconds": dur, "rtf_median": (statistics.median(walls) / dur) if dur else None, "cards": cards}
        results["tts"].append(row)
        print(f"  -> wall median {row['wall_s']['median'] * 1000:.0f} ms  p95 {row['wall_s']['p95'] * 1000:.0f}  "
              f"min {row['wall_s']['min'] * 1000:.0f} | audio {dur}s  RTF {row['rtf_median']}")

    # ---- what Whisper actually loaded --------------------------------------------------------------------
    api_lines = log_lines_since(session_id, base_url, log_types, cursor)
    journal = journal_lines_since(start_utc)
    hits = resolve_whisper_variant(api_lines + journal)
    whisper_lines = [l for l in api_lines + journal if "whisper" in l.lower()][:20]
    results["whisper_resolved"] = hits if hits else "unknown"
    results["whisper_log_lines"] = whisper_lines
    results["swarm_status_end"] = swarm_status(session_id, base_url)
    print("\nWhisper variant per Swarm logs during the run: "
          + (", ".join(hits) if hits else "unknown (no log line named a checkpoint; the STT API response carries no model field)"))
    for l in whisper_lines[:8]:
        print("   ", l[:200])

    # ---- markdown --------------------------------------------------------------------------------------------
    def card_name(u):
        for g in gpus:
            if g["uuid"] == u:
                return f"{g['name']} ({u})"
        return u
    print("\n| STT case | wall median ms | p95 | min | server processing median ms | card(s) | transcript |")
    print("|---|---:|---:|---:|---:|---|---|")
    for r in results["stt"]:
        w = r["wall_s"]
        sv = r["server_processing_s"]
        print(f"| whisper_stt model={r['requested_model']} {r['seconds']}s {r['variant']} | {w['median'] * 1000:.0f} | "
              f"{w['p95'] * 1000:.0f} | {w['min'] * 1000:.0f} | {sv['median'] * 1000:.0f} | "
              f"{', '.join(sorted(set(card_name(c) for c in r['cards'])))} | {r['transcription']} |")
    print("\n| TTS case | wall median ms | p95 | min | audio s | RTF (median wall / audio) | card(s) |")
    print("|---|---:|---:|---:|---:|---:|---|")
    for r in results["tts"]:
        w = r["wall_s"]
        print(f"| kokoro_tts voice={r['voice']} {r['words']} words | {w['median'] * 1000:.0f} | {w['p95'] * 1000:.0f} | "
              f"{w['min'] * 1000:.0f} | {r['audio_seconds']} | {r['rtf_median']:.3f} | "
              f"{', '.join(sorted(set(card_name(c) for c in r['cards'])))} |")

    Path(args.voice_out).write_text(json.dumps(results, indent=2))
    print(f"\nVoice results -> {args.voice_out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="192.168.10.188")
    ap.add_argument("--port", default="7801")
    ap.add_argument("--out", default=str(Path(__file__).parent / "swarm_audio_results.json"))
    ap.add_argument("--timeout", type=float, default=240.0)
    ap.add_argument("--categories", default="TTS,STT,Music,VoiceConversion,Fx")
    ap.add_argument("--voice", action="store_true", help="voice-turn latency mode (see module doc)")
    ap.add_argument("--voice-whisper-model", default="small",
                    help="AudioLab whisper_stt model id (tiny|base|small|medium|large-v2|large-v3|turbo); empty = provider default")
    ap.add_argument("--voice-tts-voice", default="af_heart")
    ap.add_argument("--voice-out", default=str(Path(__file__).parent / "swarm_voice_results.json"))
    args = ap.parse_args()

    if args.voice:
        run_voice(args)
        return

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
