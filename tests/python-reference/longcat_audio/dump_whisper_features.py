"""Avatar 1.5/1.6 audio features from the real Whisper-large-v3 encoder, the reference's own path, on CPU in F32.

Re-states get_audio_embedding_whisper (pipeline_longcat_video_avatar.py L663-732) with the pinned code's constants, then the
demo's window cut and the DiT's first-frame/4-frame grouping (via dump_host_math). Loudness normalisation is off, as for 1.6
(the 1.5 path would run pyloudnorm first; pass --loudness-norm to dump it). Input: the saved speech16k.f32 that
dump_mel_128.py wrote (pre-separated 16 kHz mono vocals), so the separator is out of the comparison.

Usage: dump_whisper_features.py --whisper-dir DIR_WITH_model.safetensors [--out DIR] [--fps 25] [--loudness-norm]
Needs torch >= 2.6, transformers (4.41 pinned upstream), numpy. fp16 file is loaded as fp32, as WhisperModel.from_pretrained does.
"""
import argparse
import os

import numpy as np
import torch
from transformers import AutoFeatureExtractor, WhisperModel

from common import OUT_DEFAULT, set_meta, write_f32
from dump_host_math import dump_layout, whisper_pool

MEL_CHUNK = 750 * 640
ENC_CHUNK = 3000


@torch.no_grad()
def embed(speech, model, extractor, fps, loudness_norm):
    video_length = int(len(speech) / 16000 * fps)
    if loudness_norm:
        import pyloudnorm as pyln

        meter = pyln.Meter(16000)
        loudness = meter.integrated_loudness(speech)
        if abs(loudness) <= 100:
            speech = pyln.normalize.loudness(speech, loudness, -23)
    mels = [
        extractor(speech[i : i + MEL_CHUNK], sampling_rate=16000, return_tensors="pt").input_features
        for i in range(0, len(speech), MEL_CHUNK)
    ]
    audio_features = torch.cat(mels, dim=-1).to(model.dtype)
    enc = []
    for i in range(0, audio_features.shape[-1], ENC_CHUNK):
        hs = model.encoder(audio_features[:, :, i : i + ENC_CHUNK], output_hidden_states=True).hidden_states
        enc.append(torch.stack(hs, dim=2))
    prompts = torch.cat(enc, dim=1)
    return prompts, whisper_pool(prompts, video_length, fps), video_length


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--whisper-dir", required=True)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--loudness-norm", action="store_true")
    ap.add_argument("--segments", type=int, default=2)
    args = ap.parse_args()

    speech = np.fromfile(os.path.join(args.out, "speech16k.f32"), dtype="<f4")
    model = WhisperModel.from_pretrained(args.whisper_dir).eval()
    assert model.dtype == torch.float32, model.dtype
    extractor = AutoFeatureExtractor.from_pretrained(args.whisper_dir)
    prompts, emb, video_length = embed(speech, model, extractor, args.fps, args.loudness_norm)

    write_f32(args.out, "ws_states_last", prompts[0, :, 32])  # post-final-LN state, [T_enc, 1280]
    write_f32(args.out, "ws_states_first", prompts[0, :, 0])
    set_meta(args.out, "ws_video_length", video_length)
    set_meta(args.out, "ws_loudness_norm", bool(args.loudness_norm))
    dump_layout(args.out, "ws", emb, stride=1, segments=args.segments)
    print("video frames", video_length, "->", args.out)


if __name__ == "__main__":
    main()
