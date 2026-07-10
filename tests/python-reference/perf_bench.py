#!/usr/bin/env python3
"""Python GPU baseline timings for TTS/STT models whose libs are importable in this env.
Each model: warm once, then time. Robust — skips models that fail to import/run.
NOTE: this env has transformers 5.13 (neucodec install) which breaks kokoro/melo/torchvision."""
import warnings, time, os; warnings.filterwarnings("ignore")
import numpy as np, soundfile as sf
os.makedirs("/tmp/hartsyinference_tts_to_stt", exist_ok=True)
TEXT="The speech synthesizer is now working correctly."
JFK="/home/kalebbroo/.cache/hartsyinference/models/test-clips/jfk.wav"

def sync():
    try:
        import torch; torch.cuda.synchronize() if torch.cuda.is_available() else None
    except: pass

def bench(name, fn):
    try:
        fn(); sync()  # warm
        t0=time.perf_counter(); dur=fn(); sync(); dt=time.perf_counter()-t0
        print(f"PY {name}: gen {dt:.2f}s | audio {dur:.2f}s | RTF {dt/dur if dur else 0:.3f}", flush=True)
    except Exception as e:
        print(f"PY {name}: FAIL {str(e)[:90]}", flush=True)

# --- F5 ---
def f5():
    import torch
    from f5_tts.api import F5TTS
    global _f5
    if '_f5' not in globals(): _f5=F5TTS(model="F5TTS_v1_Base", device="cuda")
    wav,sr,_=_f5.infer(ref_file=JFK, ref_text="And so my fellow Americans ask not what your country can do for you", gen_text=TEXT, nfe_step=32, cfg_strength=2.0, sway_sampling_coef=-1, seed=7)
    return len(np.asarray(wav).flatten())/sr
bench("F5-TTS", f5)

# --- Dia ---
def dia():
    import torch
    from dia.model import Dia
    global _dia
    if '_dia' not in globals(): _dia=Dia.from_pretrained("nari-labs/Dia-1.6B-0626", compute_dtype="float32")
    out=_dia.generate("[S1] "+TEXT+" [S2] That is wonderful news.", use_torch_compile=False, verbose=False)
    return len(np.asarray(out).flatten())/44100
bench("Dia-1.6B", dia)

# --- Bark ---
def bark():
    import torch
    from transformers import BarkModel, AutoProcessor
    global _bark,_bp
    if '_bark' not in globals():
        _bp=AutoProcessor.from_pretrained("suno/bark"); _bark=BarkModel.from_pretrained("suno/bark").to("cuda")
    inp=_bp(TEXT, voice_preset="v2/en_speaker_6")
    inp={k:v.to("cuda") for k,v in inp.items()}
    with torch.no_grad(): out=_bark.generate(**inp)
    return out.shape[-1]/_bark.generation_config.sample_rate
bench("Bark", bark)

# --- Whisper STT (transcribe jfk) ---
def whisper_stt():
    import whisper, torch
    global _wsp
    if '_wsp' not in globals(): _wsp=whisper.load_model("base")
    import wave; w=wave.open(JFK); d=w.getnframes()/w.getframerate(); w.close()
    _wsp.transcribe(JFK, language="en", fp16=True)
    return d
bench("Whisper-base(STT)", whisper_stt)
print("=== PY BENCH DONE ===")
