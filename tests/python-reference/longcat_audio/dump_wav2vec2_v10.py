"""Avatar 1.0 audio features from chinese-wav2vec2-base, through the pinned reference's own Wav2Vec2ModelWrapper.

get_audio_embedding_wav2vec2 (pipeline_longcat_video_avatar.py L735-757) preprocesses with loudness norm (-23 LUFS), an
UNSEEDED -45 dBFS noise floor and a 3 kHz Butterworth low-pass, then Wav2Vec2FeatureExtractor (default do_normalize=True,
because the reference builds it with the constructor, not from_pretrained). The noise is unseeded upstream, so this script
seeds numpy and SAVES the preprocessed waveform and the extractor's input_values: the C# side starts from those.

Also converts the bundled pickle (pytorch_model.bin, 'wav2vec2.' prefix) to a safetensors file next to the dumps, because the
TencentGameMate repo ships no safetensors and the engine reads safetensors.

Usage: dump_wav2vec2_v10.py --longcat-repo CHECKOUT_AT_58eb336 --w2v-dir DIR_WITH_pytorch_model.bin_and_config.json [--out DIR]
Needs torch >= 2.6, transformers (4.41 pinned upstream), scipy, pyloudnorm, safetensors. Input: speech16k.f32 from dump_mel_128.py.
"""
import argparse
import importlib.util
import os
import sys
import types

import numpy as np
import scipy.signal as ss
import torch
import torch.nn.functional as F

from common import OUT_DEFAULT, set_meta, write_f32
from dump_host_math import dump_layout


def load_reference_wav2vec2(repo):
    """Import audio_process/wav2vec2.py without the heavy longcat_video package (its torch_utils pulls librosa, imageio...)."""
    pkg = types.ModuleType("lc_audio")
    pkg.__path__ = [os.path.join(repo, "longcat_video", "audio_process")]
    sys.modules["lc_audio"] = pkg
    tu = types.ModuleType("lc_audio.torch_utils")

    def linear_interpolation(features, seq_len):  # verbatim from torch_utils.py L25-28
        features = features.transpose(1, 2)
        output_features = F.interpolate(features, size=seq_len, align_corners=True, mode="linear")
        return output_features.transpose(1, 2)

    tu.linear_interpolation = linear_interpolation
    sys.modules["lc_audio.torch_utils"] = tu
    spec = importlib.util.spec_from_file_location("lc_audio.wav2vec2", os.path.join(pkg.__path__[0], "wav2vec2.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["lc_audio.wav2vec2"] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--longcat-repo", required=True)
    ap.add_argument("--w2v-dir", required=True)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--segments", type=int, default=2)
    args = ap.parse_args()

    import pyloudnorm as pyln
    from safetensors.torch import save_file
    from transformers import Wav2Vec2FeatureExtractor

    mod = load_reference_wav2vec2(args.longcat_repo)
    speech = np.fromfile(os.path.join(args.out, "speech16k.f32"), dtype="<f4").astype("float64")
    fps = 32  # save_fps 16 * audio_stride 2
    video_length = len(speech) / 16000 * fps

    # preprocessing, as _loudness_norm / _add_noise_floor / _smooth_transients
    meter = pyln.Meter(16000)
    loudness = meter.integrated_loudness(speech)
    if abs(loudness) <= 100:
        speech = pyln.normalize.loudness(speech, loudness, -23)
    np.random.seed(0)
    speech = speech + np.random.randn(len(speech)) * 10 ** (-45 / 20)
    b, a = ss.butter(3, 3000 / (16000 / 2))
    speech = ss.lfilter(b, a, speech)
    write_f32(args.out, "w2v_waveform", speech)

    extractor = Wav2Vec2FeatureExtractor(args.w2v_dir, local_files_only=True)  # the reference's construction
    set_meta(args.out, "w2v_do_normalize", bool(extractor.do_normalize))
    values = np.squeeze(extractor(speech, sampling_rate=16000).input_values)
    write_f32(args.out, "w2v_input_values", values)

    encoder = mod.Wav2Vec2ModelWrapper(args.w2v_dir, device="cpu")
    with torch.no_grad():
        feats = torch.from_numpy(values).float().unsqueeze(0)
        out = encoder(feats, seq_len=int(video_length), output_hidden_states=True)
        emb = torch.stack(out.hidden_states[1:], dim=1).squeeze(0)  # [12, T, 768]
        emb = emb.permute(1, 0, 2).contiguous()  # [T, 12, 768]
    write_f32(args.out, "w2v_state0", out.hidden_states[0][0])
    set_meta(args.out, "w2v_video_length", int(video_length))
    dump_layout(args.out, "w2vr", emb, stride=2, segments=args.segments)

    # safetensors twin of the pickle, 'wav2vec2.' prefix stripped, the engine's key names
    state = torch.load(os.path.join(args.w2v_dir, "pytorch_model.bin"), map_location="cpu", weights_only=True)
    twin = {k[len("wav2vec2."):]: v.float().contiguous() for k, v in state.items() if k.startswith("wav2vec2.")}
    save_file(twin, os.path.join(args.out, "chinese-wav2vec2-base.safetensors"))
    missing = [k for k in encoder.model.state_dict() if k not in twin]
    set_meta(args.out, "w2v_state_dict_keys_missing_from_twin", missing[:20])
    print("video frames", int(video_length), "->", args.out)


if __name__ == "__main__":
    main()
