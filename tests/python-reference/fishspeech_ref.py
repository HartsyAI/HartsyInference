"""Fish-Speech 1.5 REFERENCE ORACLE (text -> semantic tokens -> firefly-gan-vq -> 44.1 kHz wav).

Purpose: produce a ground-truth waveform + level (RMS) for the sentence below, so the pure-C#
HartsyInference FishSpeech pipeline can be compared for OUTPUT SCALE (the "too quiet" symptom).

RUN ON GPU ONLY (do NOT run inside the C#/dotnet IDE session -- host-RAM OOM risk). Requires the
official `fish-speech` package (pip install fish-speech, or the fishaudio/fish-speech repo on PYTHONPATH)
and torch with CUDA.

Weights already on disk (no download needed):
    ~/.cache/hartsyinference/models/fishaudio--fish-speech-1.5/
        model.pth                                   # DualAR text2semantic
        firefly-gan-vq-fsq-8x1024-21hz-generator.pth  # firefly codec (decode)
        tokenizer.tiktoken, special_tokens.json     # tokenizer

Output: /tmp/hartsyinference_tts_to_stt/fishspeech_REF_python.wav
Prints: wall-clock gen time, audio duration (s), and RMS (the number to compare against C#).

ASSUMPTIONS / NOTES (verify against your installed fish-speech version; the internal module paths
have drifted across releases -- if an import fails, the equivalent official CLI is given at the bottom):
  * fish-speech 1.5 checkpoint dir == the HF snapshot above (it is the canonical layout).
  * `generate_long` yields segments whose `.codes` is a LongTensor [num_codebooks=8, T]; concatenated
    along T across segments. These are the SEMANTIC codebook indices fed to the firefly decoder.
  * The firefly decoder (`FireflyArchitecture.decode`) takes indices [B, 8, T] and returns audio in
    [-1, 1] at 44100 Hz. NO post-gain / loudness-normalization is applied anywhere -- that is the whole
    point of this oracle: whatever RMS it prints is the TRUE reference level (the C# pipeline must match
    it WITHOUT the LoudnessNormalizer band-aid).
  * temperature=0.7, top_p=0.7, repetition_penalty=1.2 (fish-speech generate_long/webui defaults, same
    as FishSpeechConfig.V1_5). Set a fixed seed for reproducibility.
"""

import os
import time
import numpy as np
import torch
import soundfile as sf

MODEL_DIR = os.path.expanduser("~/.cache/hartsyinference/models/fishaudio--fish-speech-1.5")
LLAMA_CKPT = MODEL_DIR                                                   # dir holding model.pth + tokenizer
FIREFLY_CKPT = os.path.join(MODEL_DIR, "firefly-gan-vq-fsq-8x1024-21hz-generator.pth")
OUT_DIR = "/tmp/hartsyinference_tts_to_stt"
OUT_WAV = os.path.join(OUT_DIR, "fishspeech_REF_python.wav")
TEXT = "The speech synthesizer is now working correctly."
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(SEED)

    # --- Stage 1: text -> semantic codebook indices (DualAR) -------------------------------------
    # Official module (fish-speech >=1.5). If your version renames these, see the CLI fallback below.
    from fish_speech.models.text2semantic.inference import (
        launch_thread_safe_queue,
        generate_long,
    )

    # Load the DualAR model onto the queue-based generator used by the official server.
    engine = launch_thread_safe_queue(
        checkpoint_path=LLAMA_CKPT,
        device=DEVICE,
        precision=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
        compile=False,
    )

    t0 = time.time()
    codes_segments = []
    for resp in generate_long(
        model=engine,
        text=TEXT,
        num_samples=1,
        max_new_tokens=1500,
        top_p=0.7,
        repetition_penalty=1.2,
        temperature=0.7,
        # No reference audio -> pure TTS (this path does NOT hit any resampler).
        prompt_text=None,
        prompt_tokens=None,
    ):
        if getattr(resp, "action", None) == "sample" and getattr(resp, "codes", None) is not None:
            codes_segments.append(resp.codes)          # [8, T_seg] LongTensor
    if not codes_segments:
        raise RuntimeError("generate_long produced no codes -- check the fish-speech version/API.")
    codes = torch.cat(codes_segments, dim=1)           # [8, T]
    gen_secs = time.time() - t0
    print(f"semantic codes shape={tuple(codes.shape)}  (num_codebooks, T)")

    # --- Stage 2: firefly-gan-vq decode (indices -> 44.1 kHz audio) ------------------------------
    from fish_speech.models.vqgan.inference import load_model as load_decoder

    decoder = load_decoder(
        config_name="firefly_gan_vq",
        checkpoint_path=FIREFLY_CKPT,
        device=DEVICE,
    )
    with torch.no_grad():
        indices = codes.to(DEVICE).long()[None]        # [1, 8, T]
        # FireflyArchitecture.decode returns (audio [1,1,S], feature_lengths) in most 1.5 builds.
        out = decoder.decode(indices=indices, feature_lengths=torch.tensor([indices.shape[-1]], device=DEVICE))
        audio = out[0] if isinstance(out, (tuple, list)) else out
    audio = audio.squeeze().float().cpu().numpy()      # [S], already in [-1, 1], NO gain applied

    sr = 44100
    dur = len(audio) / sr
    rms = float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else 0.0
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    sf.write(OUT_WAV, audio, sr)
    print(f"wrote {OUT_WAV}")
    print(f"gen_time={gen_secs:.2f}s  duration={dur:.2f}s  sample_rate={sr}")
    print(f"RMS={rms:.6f}  peak={peak:.6f}   <-- compare RMS/peak against the C# output (no LoudnessNormalizer)")


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------------------------------
# CLI FALLBACK (if the internal import paths differ in your fish-speech version) -- two official steps:
#
#   # 1) text -> codes (writes codes_0.npy)
#   python fish_speech/models/text2semantic/inference.py \
#       --text "The speech synthesizer is now working correctly." \
#       --checkpoint-path ~/.cache/hartsyinference/models/fishaudio--fish-speech-1.5 \
#       --num-samples 1 --top-p 0.7 --repetition-penalty 1.2 --temperature 0.7
#
#   # 2) codes -> wav
#   python fish_speech/models/vqgan/inference.py \
#       --input-path codes_0.npy \
#       --checkpoint-path ~/.cache/hartsyinference/models/fishaudio--fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth \
#       --output-path /tmp/hartsyinference_tts_to_stt/fishspeech_REF_python.wav
#
# Then read the wav and print RMS:
#   python -c "import soundfile as sf,numpy as np; a,sr=sf.read('/tmp/hartsyinference_tts_to_stt/fishspeech_REF_python.wav'); print('RMS',np.sqrt((a**2).mean()),'peak',np.abs(a).max(),'dur',len(a)/sr)"
# ---------------------------------------------------------------------------------------------------
