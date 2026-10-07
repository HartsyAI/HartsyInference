"""Reference activations of ControlFoley's video-conditioning path (Synchformer, CAV-MAE-ST visual branch, frame
preprocessing) for the C# port. The official methods are executed verbatim (extracted with `ast`, only `.cuda()` removed).

Usage:
  python video_reference.py tiny <controlfoley checkout> <output dir>
      tiny random official Synchformer / CAVMAEST (released key layout, CAV-MAE saved with its `module.` prefix) and a
      synthetic mp4 run through the official frame extraction + transforms at small sizes
      -> video_sync_tiny.safetensors, video_cav_tiny.safetensors, video_frames_tiny.safetensors/.json
  python video_reference.py real <controlfoley checkout> <output dir> <synchformer.pth> <cav_mae_st.pth> <video.mp4> [seconds]
      the real checkpoints on a real clip -> video_real_ref.safetensors (frames, preprocessed tensors, sync features,
      CAV-MAE pooled features)

Note: the official load of cav_mae_st.pth (strict=False, keys prefixed `module.`) matches nothing; the references here load
the real weights with the prefix stripped, which is what the port does.
Needs torch, torchvision, timm, einops, omegaconf, av, safetensors, numpy.
"""
import ast
import json
import sys
import types
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import torch
from einops import rearrange
from safetensors.torch import save_file
from torchvision.transforms import v2

repo = Path(sys.argv[2])
sys.path.insert(0, str(repo))

import lib.cav_mae_st.core.models as cav_models  # noqa: E402
import lib.synchformer.motionformer as motionformer  # noqa: E402
from controlfoley.media_utils import extract_video_segments  # noqa: E402
from lib.synchformer import Synchformer  # noqa: E402


def class_method(name: str, strip_cuda: bool = False):
    source = (repo / "controlfoley" / "feature_extractor.py").read_text(encoding="utf-8")
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "FeaturesUtils")
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    code = ast.unparse(node)
    if strip_cuda:
        code = code.replace(".cuda()", "")
    namespace = {"torch": torch, "rearrange": rearrange}
    exec(code, namespace)
    return namespace[name]


def load_video_function(sizes: dict):
    source = (repo / "controlfoley" / "inference_utils.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "load_video")
    from controlfoley.media_utils import MediaClipData
    import logging
    namespace = {"torch": torch, "v2": v2, "Path": Path, "MediaClipData": MediaClipData, "log": logging.getLogger(),
                 "extract_video_segments": extract_video_segments, "_CLIP_FPS": 8.0, "_VISUAL_FPS": 4.0, "_SYNC_FPS": 25.0, **sizes}
    exec(ast.unparse(node), namespace)
    return namespace["load_video"]


def randomize(model: torch.nn.Module, seed: int) -> None:
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            noise = torch.randn(p.shape, generator=gen)
            if name.endswith("norm.weight") or ".norm" in name and name.endswith("weight"):
                p.copy_(1.0 + 0.2 * noise)
            elif p.ndim == 1 or "cls_token" in name or "embed" in name and "proj" not in name:
                p.copy_(0.1 * noise)
            else:
                fan_in = int(np.prod(p.shape[1:]))
                p.copy_(noise * (0.9 / fan_in ** 0.5))


def tiny_sync(out: Path) -> None:
    class Shim:
        @staticmethod
        def load(path):
            from omegaconf import OmegaConf
            cfg = OmegaConf.load(path)
            cfg.VIT.EMBED_DIM, cfg.VIT.DEPTH, cfg.VIT.NUM_HEADS, cfg.VIT.PATCH_SIZE = 48, 2, 3, 56
            return cfg

    original = motionformer.OmegaConf
    motionformer.OmegaConf = Shim
    try:
        model = Synchformer().eval()
    finally:
        motionformer.OmegaConf = original
    randomize(model, 21)
    seed = torch.rand(32, 3, 14, 14, generator=torch.Generator().manual_seed(3)) * 2 - 1
    frames = seed.repeat_interleave(16, dim=-1).repeat_interleave(16, dim=-2).unsqueeze(0)
    stub = types.SimpleNamespace(synchformer=model)
    with torch.inference_mode():
        ref = class_method("encode_video_with_sync")(stub, frames)
    tensors = {k: v.contiguous().float() for k, v in model.state_dict().items()}
    tensors["in.seed"] = seed.contiguous()
    tensors["ref.sync"] = ref.contiguous()
    out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out / "video_sync_tiny.safetensors"))
    print("wrote sync", tuple(ref.shape))


def tiny_cav(out: Path) -> None:
    model = cav_models.CAVMAEST(img_size=64, patch_size=16, embed_dim=48, num_heads=3, modality_specific_depth=2,
                                decoder_embed_dim=32, decoder_depth=1, decoder_num_heads=2, norm_pix_loss=False,
                                tr_pos=False).eval()
    randomize(model, 33)
    frames = torch.randn(5, 3, 64, 64, generator=torch.Generator().manual_seed(4))
    stub = types.SimpleNamespace(cav_mae=model)
    with torch.inference_mode():
        tokens = class_method("encode_video_with_cav_mae", strip_cuda=True)(stub, frames)  # [1, T, N, D]
        pooled = torch.mean(tokens, dim=2)
    tensors = {"module." + k: v.contiguous().float() for k, v in model.state_dict().items()}
    tensors["in.frames"] = frames.contiguous()
    tensors["ref.tokens"] = tokens[0].contiguous()
    tensors["ref.pooled"] = pooled[0].contiguous()
    out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out / "video_cav_tiny.safetensors"))
    print("wrote cav", tuple(tokens.shape), tuple(pooled.shape))


def decode_all(path: Path, end_time: float):
    """Frames and presentation times exactly as `extract_video_segments(..., extract_all_frames=True)` sees them."""
    frames, times = [], []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for packet in container.demux(stream):
            for frame in packet.decode():
                if frame.time < 0:
                    continue
                if frame.time > end_time:
                    break
                frames.append(frame.to_ndarray(format="rgb24"))
                times.append(frame.time)
    return np.stack(frames), np.array(times, dtype=np.float64)


def synthetic_video(path: Path) -> None:
    gen = np.random.default_rng(5)
    h, w, n = 36, 52, 75
    yy, xx = np.mgrid[0:h, 0:w]
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=30)
        stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
        for i in range(n):
            base = np.stack([128 + 100 * np.sin(xx / 7.0 + i / 9.0), 128 + 100 * np.cos(yy / 5.0 - i / 6.0),
                             128 + 100 * np.sin((xx + yy) / 11.0 + i / 4.0)], axis=-1)
            img = np.clip(base + gen.normal(0, 8, base.shape), 0, 255).astype(np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def tiny_frames(out: Path, scratch: Path) -> None:
    sizes = {"_CLIP_SIZE": 40, "_VISUAL_SIZE": 24, "_SYNC_SIZE": 32}
    video = scratch / "synthetic.mp4"
    synthetic_video(video)
    load_video = load_video_function(sizes)
    cases = [2.0, 4.0]
    all_frames, all_times = decode_all(video, 1e9)
    tensors = {"in.frames": torch.from_numpy(all_frames), "in.times": torch.from_numpy(all_times)}
    meta = {"sizes": {"clip": 40, "visual": 24, "sync": 32}, "cases": []}
    for i, duration in enumerate(cases):
        info = load_video(video, duration, load_all_frames=True)
        tensors[f"ref.{i}.clip"] = info.clip_embeddings.contiguous()
        tensors[f"ref.{i}.visual"] = info.visual_features.contiguous()
        tensors[f"ref.{i}.sync"] = info.sync_embeddings.contiguous()
        meta["cases"].append({"duration": duration, "total_duration": info.total_duration})
        print("case", duration, info.total_duration, tuple(info.clip_embeddings.shape), tuple(info.visual_features.shape),
              tuple(info.sync_embeddings.shape))
    out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out / "video_frames_tiny.safetensors"))
    (out / "video_frames_tiny.json").write_text(json.dumps(meta))


def real(out: Path, sync_ckpt: Path, cav_ckpt: Path, video: Path, seconds: float) -> None:
    sizes = {"_CLIP_SIZE": 384, "_VISUAL_SIZE": 224, "_SYNC_SIZE": 224}
    info = load_video_function(sizes)(video, seconds, load_all_frames=False)
    frames, times = decode_all(video, info.total_duration)
    print("decoded", frames.shape, "total_duration", info.total_duration)

    synchformer = Synchformer().eval()
    synchformer.load_state_dict(torch.load(sync_ckpt, weights_only=True, map_location="cpu"))
    cav = cav_models.CAVMAEST(audio_length=208, norm_pix_loss=False, modality_specific_depth=11, tr_pos=False).eval()
    state = {k[len("module."):]: v for k, v in torch.load(cav_ckpt, map_location="cpu").items()}
    print(cav.load_state_dict(state, strict=False))
    with torch.inference_mode():
        sync = class_method("encode_video_with_sync")(types.SimpleNamespace(synchformer=synchformer), info.sync_embeddings.unsqueeze(0))
        tokens = class_method("encode_video_with_cav_mae", strip_cuda=True)(types.SimpleNamespace(cav_mae=cav),
                                                                              info.visual_features)
        pooled = torch.mean(tokens, dim=2)
    save_file({"in.frames": torch.from_numpy(frames), "in.times": torch.from_numpy(times),
               "ref.clip": info.clip_embeddings.contiguous(), "ref.visual": info.visual_features.contiguous(),
               "ref.sync_frames": info.sync_embeddings.contiguous(), "ref.sync": sync.contiguous(),
               "ref.cav_pooled": pooled[0].contiguous()}, str(out / "video_real_ref.safetensors"))
    (out / "video_real_ref.json").write_text(json.dumps({"duration": seconds, "total_duration": info.total_duration}))
    print("wrote real", tuple(sync.shape), tuple(pooled.shape))


if __name__ == "__main__":
    mode, outdir = sys.argv[1], Path(sys.argv[3])
    outdir.mkdir(parents=True, exist_ok=True)
    if mode == "tiny":
        tiny_sync(outdir)
        tiny_cav(outdir)
        scratch = Path(sys.argv[4]) if len(sys.argv) > 4 else outdir
        tiny_frames(outdir, scratch)
    elif mode == "real":
        real(outdir, Path(sys.argv[4]), Path(sys.argv[5]), Path(sys.argv[6]), float(sys.argv[7]) if len(sys.argv) > 7 else 2.0)
