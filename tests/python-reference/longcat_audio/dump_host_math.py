"""Host-math reference for LongCat-Video-Avatar's audio windowing: no weights, saved random inputs.

Runs the reference's own operations (pipeline_longcat_video_avatar.py get_audio_embedding_whisper L663-732, the demo's
window cut L267-273 and L364-368, and the first-frame/4-frame grouping from longcat_video_dit_avatar.py L504-534) on saved
random "hidden states" and writes inputs and outputs. The C# LongCatAudioFeatures must match at relRMS ~1e-5 (CPU F32).

Usage: dump_host_math.py [--out DIR]
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F

from common import OUT_DEFAULT, set_meta, write_f32


def linear_interpolation_fps(features, input_fps, output_fps, output_len=None):
    features = features.transpose(1, 2)  # [B, D, T]
    if output_len is None:
        output_len = int(features.shape[2] / float(input_fps) * output_fps)
    out = F.interpolate(features, size=output_len, align_corners=True, mode="linear")
    return out.transpose(1, 2)


def whisper_pool(audio_prompts, video_length, fps=25, enc_fps=50):
    """audio_prompts: [1, T_enc, 33, D] -> [T, 5, D], as get_audio_embedding_whisper does after the encoder."""
    audio_prompts = audio_prompts[:, : video_length * 2]
    f0 = linear_interpolation_fps(audio_prompts[:, :, 0:8].mean(dim=2), enc_fps, fps, video_length)
    f1 = linear_interpolation_fps(audio_prompts[:, :, 8:16].mean(dim=2), enc_fps, fps, video_length)
    f2 = linear_interpolation_fps(audio_prompts[:, :, 16:24].mean(dim=2), enc_fps, fps, video_length)
    f3 = linear_interpolation_fps(audio_prompts[:, :, 24:32].mean(dim=2), enc_fps, fps, video_length)
    f4 = linear_interpolation_fps(audio_prompts[:, :, 32], enc_fps, fps, video_length)
    return torch.stack([f0, f1, f2, f3, f4], dim=2)[0]


def segment_windows(full_audio_emb, start, stride, num_frames=93, audio_window=5):
    indices = torch.arange(audio_window) - audio_window // 2
    end = start + stride * num_frames
    center = torch.arange(start, end, stride).unsqueeze(1) + indices.unsqueeze(0)
    center = torch.clamp(center, min=0, max=full_audio_emb.shape[0] - 1)
    return full_audio_emb[center][None, ...]  # [1, F, W, S, C]


def group_for_audio_proj(audio_cond, audio_window=5, vae_scale=4):
    """The slicing at the top of the DiT forward; returns the two AudioProjModel inputs (before proj1 / proj1_vf)."""
    first = audio_cond[:, :1, ...]
    latter = audio_cond[:, 1:, ...]
    b, nf, w, s, c = latter.shape
    latter = latter.reshape(b, nf // vae_scale, vae_scale, w, s, c)
    mid = audio_window // 2
    a = latter[:, :, :1, : mid + 1, ...].reshape(b, nf // vae_scale, -1, s, c)
    z = latter[:, :, -1:, mid:, ...].reshape(b, nf // vae_scale, -1, s, c)
    m = latter[:, :, 1:-1, mid : mid + 1, ...].reshape(b, nf // vae_scale, -1, s, c)
    return first, torch.cat([a, m, z], dim=2)


def dump_layout(out, tag, full, stride, segments):
    write_f32(out, f"{tag}_features", full.numpy())
    for k in range(segments):
        start = k * stride * (93 - 13)
        cond = segment_windows(full, start, stride)
        first, later = group_for_audio_proj(cond)
        write_f32(out, f"{tag}_windows_seg{k}", cond[0].numpy())
        write_f32(out, f"{tag}_first_seg{k}", first[0].numpy())
        write_f32(out, f"{tag}_later_seg{k}", later[0].numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DEFAULT)
    args = ap.parse_args()
    rng = np.random.default_rng(1234)

    # Whisper layout: 7.3 s of "speech" -> 182 video frames, 364 encoder frames, 33 states, 16 channels.
    video_length = 182
    states = rng.standard_normal((1, 2 * video_length + 21, 33, 16)).astype("float32")  # a few frames past the cut
    write_f32(args.out, "hm_whisper_states", states[0])
    pooled = whisper_pool(torch.from_numpy(states), video_length)
    set_meta(args.out, "hm_video_length", video_length)
    dump_layout(args.out, "hm_whisper", pooled, stride=1, segments=2)

    # wav2vec2 layout: 12 blocks x 8 channels at 32 Hz, stride 2 (Avatar 1.0).
    w2v = torch.from_numpy(rng.standard_normal((400, 12, 8)).astype("float32"))
    dump_layout(args.out, "hm_w2v", w2v, stride=2, segments=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
