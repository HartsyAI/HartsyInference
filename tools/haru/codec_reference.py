"""Upstream Haru codec (PyTorch, float32, CPU) on a real recording: the numbers the engine is compared with.

    python codec_reference.py <haru package dir> <codec-only safetensors> <wav> <out dir>
"""
import sys
import wave
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

package, weights, wav_path, out = sys.argv[1:5]
sys.path.insert(0, package)
from haru.config import CodecConfig  # noqa: E402
from haru.models.vibevoice_codec import CodecStream, VibeVoiceCodec  # noqa: E402

torch.set_num_threads(4)
cfg = CodecConfig(latent_dim=64, base_channels=32, ratios=(8, 5, 5, 4, 2, 2),
                  encoder_depths=(3, 3, 3, 3, 3, 3, 8), decoder_depths=(8, 3, 3, 3, 3, 3, 3), norm_eps=1e-5, latent_std=0.5)
codec = VibeVoiceCodec(cfg)
state = load_file(weights)
state = {k[len("codec."):]: v.float() for k, v in state.items()}
codec.load_state_dict(state, strict=True)
codec.eval().requires_grad_(False)

w = wave.open(wav_path)
rate, n = w.getframerate(), w.getnframes()
pcm = np.frombuffer(w.readframes(n), dtype="<i2").astype(np.float32) / 32768.0
# linear resample to 24 kHz: both implementations get exactly this array
t_out = np.arange(int(len(pcm) * 24000 / rate)) * rate / 24000
clip = np.interp(t_out, np.arange(len(pcm)), pcm).astype(np.float32)
clip = clip[: 24000 * 6 + 1234]  # 6.05 s, ends inside a frame
print("clip samples", len(clip), "rms", float(np.sqrt((clip ** 2).mean())))

with torch.no_grad():
    mean = codec.encode(torch.from_numpy(clip))                    # [1, T, 64]
    audio = codec.decode(mean)                                      # [1, T*3200]
    stream = CodecStream(codec)
    frames = clip[: 3 * 3200][None]
    stream_mean = stream.encode(torch.from_numpy(frames))
    stream_audio = torch.cat([stream.decode(mean[:, i : i + 1]) for i in range(3)], dim=1)

outp = Path(out)
outp.mkdir(parents=True, exist_ok=True)
save_file({
    "in.clip": torch.from_numpy(clip),
    "in.frames": torch.from_numpy(frames),
    "ref.mean": mean.float().contiguous(),
    "ref.audio": audio.float().contiguous(),
    "ref.stream_mean": stream_mean.float().contiguous(),
    "ref.stream_audio": stream_audio.float().contiguous(),
}, str(outp / "codec_real_expected.safetensors"), metadata={"format": "pt"})
print("mean", tuple(mean.shape), "audio", tuple(audio.shape), "mean max|x|", float(mean.abs().max()), "audio max|x|", float(audio.abs().max()))
