"""Reference activations of the official ControlFoley `AudioGenerationNetwork` (+ `FlowMatching`) for the C# port.

Usage:
  python network_reference.py tiny <controlfoley checkout> <output dir>
      Builds tiny random V1 and V2 networks (released key layout, tiny dims) and writes
      controlfoley_net_v1.safetensors / controlfoley_net_v2.safetensors (weights + `cond.*` / `ref.*` tensors)
      plus controlfoley_net.json (config, timesteps, temporal-config reference values).
  python network_reference.py real <controlfoley checkout> <output dir> [checkpoint path or URL]
      Builds the released large_44k network truncated to joint blocks [0, 1, 17] + fused blocks [0, 1] with the REAL
      weights (read tensor-by-tensor from the 11 GB controlfoley.pth), writes controlfoley_trunc.safetensors (the weights,
      renamed to the truncated layout) and controlfoley_real_ref.safetensors (random fixed conditions, a velocity and a
      5-step euler trajectory), CPU float32.

Needs torch, einops, safetensors, numpy.
"""
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

repo = Path(sys.argv[2])
sys.path.insert(0, str(repo))
sys.modules.setdefault("torchdiffeq", types.SimpleNamespace(odeint=None))  # only the 'adaptive' mode needs it

from controlfoley.audio_model import AudioGenerationNetwork  # noqa: E402
from controlfoley.temporal_config import TemporalConfiguration, DEFAULT_44K_CONFIG  # noqa: E402
from lib.flow_matching import FlowMatching  # noqa: E402


def randomize(net: AudioGenerationNetwork, seed: int) -> None:
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in net.named_parameters():
            if name.endswith("latent_std"):
                p.copy_(torch.rand(p.shape, generator=g) * 0.5 + 0.7)
            elif name.endswith("latent_mean"):
                p.copy_(torch.randn(p.shape, generator=g) * 0.5)
            elif name.endswith("q_norm.weight") or name.endswith("k_norm.weight"):
                p.copy_(1.0 + 0.2 * torch.randn(p.shape, generator=g))
            elif name.startswith("empty_") or name == "sync_pos_emb":
                p.copy_(torch.randn(p.shape, generator=g) * 0.5)
            elif p.dim() >= 2:
                fan_in = int(np.prod(p.shape[1:]))
                p.copy_(torch.randn(p.shape, generator=g) * (0.8 / fan_in ** 0.5))
            else:
                p.copy_(torch.randn(p.shape, generator=g) * 0.1)


def make_conditions(net: AudioGenerationNetwork, bs: int, seed: int) -> dict:
    g = torch.Generator().manual_seed(seed)
    dims = dict(clip_f=(net.clip_seq_len, net.empty_clip_feat.shape[1]),
                visual_f=(net.visual_seq_len, net.empty_visual_feat.shape[1]),
                sync_f=(net.sync_seq_len, net.empty_sync_feat.shape[1]),
                text_f=(net._text_seq_len, net.empty_string_feat.shape[1]),
                audio_f=(net.audio_seq_len, net.empty_audio_feat.shape[1]),
                timbre_f=(net.timbre_seq_len, net.empty_timbre_feat.shape[1]))
    return {k: torch.randn(bs, *v, generator=g) for k, v in dims.items()}


def pre_dict(c, prefix: str) -> dict:
    return {f"{prefix}.{k}": getattr(c, k).detach().contiguous().clone() for k in
            ("clip_f", "sync_f", "text_f", "audio_f", "timbre_f", "clip_f_c", "text_f_c")}


def expanded(t: torch.Tensor) -> torch.Tensor:
    return t.detach().contiguous().clone()


def run_cases(net: AudioGenerationNetwork, cond: dict, ts: list, steps: int, cfg: float, full: bool, seed: int) -> dict:
    bs = cond["clip_f"].shape[0]
    g = torch.Generator().manual_seed(seed)
    out: dict = {}
    for k, v in cond.items():
        out[f"cond.{k}"] = v
    latent = torch.randn(bs, net.latent_seq_len, net.latent_dim, generator=g)
    noise = torch.randn(bs, net.latent_seq_len, net.latent_dim, generator=g)
    out["cond.latent"] = latent
    out["cond.noise"] = noise
    neg_text = torch.randn(bs, net._text_seq_len, net.empty_string_feat.shape[1], generator=g)
    out["cond.negative_text_f"] = neg_text

    with torch.no_grad():
        pre = net.preprocess_conditions(cond["clip_f"], cond["visual_f"], cond["sync_f"], cond["text_f"],
                                        cond["audio_f"], cond["timbre_f"])
        out.update(pre_dict(pre, "ref.pre"))
        empty = net.get_empty_conditions(bs)
        out.update(pre_dict(empty, "ref.empty"))
        empty_neg = net.get_empty_conditions(bs, negative_text_features=neg_text)
        out.update(pre_dict(empty_neg, "ref.emptyneg"))

        for i, tv in enumerate(ts):
            t = torch.full((bs,), tv)
            flow, multimodal, _ = net.predict_flow(latent, t, pre)
            out[f"ref.flow.t{i}"] = flow
            if i == 0:
                out["ref.multimodal"] = multimodal
            tt = torch.tensor(tv)
            out[f"ref.ode_cfg.t{i}"] = net.ode_wrapper(tt, latent, pre, empty, cfg)
            if full:
                out[f"ref.ode_cfgneg.t{i}"] = net.ode_wrapper(tt, latent, pre, empty_neg, cfg)
                out[f"ref.ode_nocfg.t{i}"] = net.ode_wrapper(tt, latent, pre, empty, 0.5)

        if full:
            getters = dict(clip_f=net.get_empty_clip_sequence, visual_f=net.get_empty_visual_sequence,
                           sync_f=net.get_empty_sync_sequence, text_f=net.get_empty_string_sequence,
                           audio_f=net.get_empty_audio_sequence, timbre_f=net.get_empty_timbre_sequence)
            for name, getter in getters.items():
                out[f"ref.getter.{name}"] = expanded(getter(bs))
                variant = dict(cond)
                variant[name] = getter(bs).contiguous()
                p = net.preprocess_conditions(variant["clip_f"], variant["visual_f"], variant["sync_f"],
                                              variant["text_f"], variant["audio_f"], variant["timbre_f"])
                flow, _, _ = net.predict_flow(latent, torch.full((bs,), ts[1]), p)
                out[f"ref.drop.{name}"] = flow

        states = []
        fm = FlowMatching(min_sigma=0, inference_mode="euler", num_steps=steps)

        def fn(t, x):
            states.append(x.clone())
            return net.ode_wrapper(t, x, pre, empty, cfg)

        x1 = fm.to_data(fn, noise.clone())
        for i, s in enumerate(states):
            out[f"ref.traj.state{i}"] = s
        out["ref.traj.final_normalized"] = x1.clone()
        out["ref.traj.final"] = net.unnormalize(x1.clone())
    return {k: v.detach().contiguous().clone() for k, v in out.items()}


def tiny_net(v2: bool, seed: int) -> AudioGenerationNetwork:
    torch.manual_seed(seed)
    net = AudioGenerationNetwork(
        mode=2, latent_dim=8, clip_dim=16, visual_dim=12, sync_dim=10, text_dim=14, audio_dim=6, timbre_dim=18,
        hidden_dim=32, depth=5, fused_depth=3, num_heads=4, mlp_ratio=4.0, latent_seq_len=12, clip_seq_len=8,
        visual_seq_len=4, sync_seq_len=16, text_seq_len=7, v2=v2).eval()
    randomize(net, seed + 1)
    return net


def weights_of(net: AudioGenerationNetwork) -> dict:
    return {k: v.detach().contiguous().clone() for k, v in net.state_dict().items()}


def temporal_reference() -> list:
    rows = []
    for secs in (8.0, 5.0, 10.5, 3.3, 6.0, 1.0, 16.0):
        try:
            c = TemporalConfiguration(total_time_seconds=secs, audio_sample_rate=44100, spec_frame_frequency=512)
            rows.append(dict(seconds=secs, latent=c.latent_sequence_length, clip=c.clip_sequence_length,
                             visual=c.visual_sequence_length, sync=c.sync_sequence_length,
                             samples=c.total_audio_sample_count))
        except RuntimeError as e:
            rows.append(dict(seconds=secs, error=str(e)))
    assert DEFAULT_44K_CONFIG.latent_sequence_length == 345
    return rows


def run_tiny(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = [0.0, 0.37, 1.0]
    meta: dict = {"timesteps": ts, "cfg": 4.5, "steps": 4, "batch": 2, "temporal": temporal_reference()}
    for v2, name in ((False, "v1"), (True, "v2")):
        net = tiny_net(v2, 100 + int(v2))
        cond = make_conditions(net, 2, 7)
        tensors = weights_of(net)
        refs = run_cases(net, cond, ts, 4, 4.5, full=not v2, seed=11)
        tensors.update(refs)
        tensors["ref.rot.latent"] = net.latent_rot.detach().contiguous().clone()
        tensors["ref.rot.clip"] = net.clip_rot.detach().contiguous().clone()
        save_file(tensors, str(out_dir / f"controlfoley_net_{name}.safetensors"))
        meta[name] = dict(v2=v2, latent_dim=8, clip_dim=16, visual_dim=12, sync_dim=10, text_dim=14, audio_dim=6,
                          timbre_dim=18, hidden_dim=32, depth=5, fused_depth=3, num_heads=4, mlp_ratio=4.0,
                          latent_seq_len=12, clip_seq_len=8, visual_seq_len=4, sync_seq_len=16, text_seq_len=7)
        flows = refs["ref.flow.t1"]
        print(name, "flow std", float(flows.std()), "max", float(flows.abs().max()), "final std",
              float(refs["ref.traj.final"].std()))
    # update_seq_lengths: same weights, different lengths
    net = tiny_net(False, 100)
    net.update_seq_lengths(10, 6, 3, 8)
    cond = make_conditions(net, 1, 21)
    tensors = {f"cond.{k}": v for k, v in cond.items()}
    g = torch.Generator().manual_seed(5)
    latent = torch.randn(1, 10, 8, generator=g)
    with torch.no_grad():
        pre = net.preprocess_conditions(cond["clip_f"], cond["visual_f"], cond["sync_f"], cond["text_f"],
                                        cond["audio_f"], cond["timbre_f"])
        flow, _, _ = net.predict_flow(latent, torch.full((1,), 0.6), pre)
    tensors["cond.latent"] = latent
    tensors["ref.flow"] = flow
    tensors["ref.rot.latent"] = net.latent_rot.detach().contiguous().clone()
    tensors["ref.rot.clip"] = net.clip_rot.detach().contiguous().clone()
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(out_dir / "controlfoley_net_resized.safetensors"))
    meta["resized"] = dict(latent_seq_len=10, clip_seq_len=6, visual_seq_len=3, sync_seq_len=8, t=0.6)
    (out_dir / "controlfoley_net.json").write_text(json.dumps(meta, indent=1))


def run_real(out_dir: Path, source: str) -> None:
    sys.path.insert(0, str(Path(__file__).parent))
    from remote_pth import RemotePth

    out_dir.mkdir(parents=True, exist_ok=True)
    joint_src, fused_src = [0, 1, 17], [0, 1]
    ckpt = RemotePth(source)
    heads = 14
    net = AudioGenerationNetwork(
        mode=2, latent_dim=40, clip_dim=1024, visual_dim=768, sync_dim=768, text_dim=1024, audio_dim=512,
        timbre_dim=1536, hidden_dim=64 * heads, depth=len(joint_src) + len(fused_src), fused_depth=len(fused_src),
        num_heads=heads, latent_seq_len=345, clip_seq_len=64, visual_seq_len=32, sync_seq_len=192).eval()

    sd = {}
    for key in ckpt.keys():
        if key.startswith("repa_mlp"):
            continue
        new = key
        if key.startswith("joint_blocks."):
            idx = int(key.split(".")[1])
            if idx not in joint_src:
                continue
            new = key.replace(f"joint_blocks.{idx}.", f"joint_blocks.{joint_src.index(idx)}.", 1)
        elif key.startswith("fused_blocks."):
            idx = int(key.split(".")[1])
            if idx not in fused_src:
                continue
            new = key.replace(f"fused_blocks.{idx}.", f"fused_blocks.{fused_src.index(idx)}.", 1)
        sd[new] = torch.from_numpy(ckpt.read(key))
    missing, unexpected = net.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    assert all(m.startswith("repa_mlp") for m in missing), missing
    save_file({k: v.contiguous() for k, v in sd.items()}, str(out_dir / "controlfoley_trunc.safetensors"))

    cond = make_conditions(net, 1, 1234)
    refs = run_cases(net, cond, [0.0, 0.5], 5, 4.5, full=False, seed=99)
    save_file(refs, str(out_dir / "controlfoley_real_ref.safetensors"))
    print("real flow std", float(refs["ref.flow.t1"].std()), "final std", float(refs["ref.traj.final"].std()))


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "tiny":
        run_tiny(Path(sys.argv[3]))
    else:
        default = "https://huggingface.co/YJX-Xiaomi/ControlFoley/resolve/main/weights/controlfoley.pth"
        run_real(Path(sys.argv[3]), sys.argv[4] if len(sys.argv) > 4 else default)
