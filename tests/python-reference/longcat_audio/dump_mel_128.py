"""128-bin log-mel reference for the C# MelSpectrogramExtractor.WhisperConfig(128), from HF WhisperFeatureExtractor.

Splits the clip the way get_audio_embedding_whisper does (480000-sample chunks, each padded to 30 s by the extractor) and
writes every chunk's input_features. Needs numpy, transformers and soundfile; no model weights, only the extractor's
preprocessor_config.json (feature_size 128) from openai/whisper-large-v3, the repo LongCat bundles.

Usage: dump_mel_128.py --wav clip.wav --whisper-dir DIR_WITH_preprocessor_config.json [--out DIR]
"""
import argparse

import numpy as np
from transformers import AutoFeatureExtractor

from common import OUT_DEFAULT, load_speech_16k, set_meta, write_f32

MEL_CHUNK = 750 * 640


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--whisper-dir", required=True)
    ap.add_argument("--out", default=OUT_DEFAULT)
    args = ap.parse_args()

    speech = load_speech_16k(args.wav)
    extractor = AutoFeatureExtractor.from_pretrained(args.whisper_dir)
    assert extractor.feature_size == 128, extractor.feature_size
    chunks = [
        extractor(speech[i : i + MEL_CHUNK], sampling_rate=16000, return_tensors="np").input_features[0]
        for i in range(0, len(speech), MEL_CHUNK)
    ]
    write_f32(args.out, "speech16k", speech)
    write_f32(args.out, "mel128_chunks", np.stack(chunks))
    try:
        import torch

        set_meta(args.out, "torch", torch.__version__)  # with torch present HF computes the STFT in torch (F32)
    except ImportError:
        set_meta(args.out, "torch", None)  # numpy path: F64 STFT
    import transformers

    set_meta(args.out, "transformers", transformers.__version__)
    print("chunks", len(chunks), "->", args.out)


if __name__ == "__main__":
    main()
