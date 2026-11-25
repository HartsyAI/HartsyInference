"""L1 oracle for the Duo DiT forward on a tiny random geometry (HartsyInference PR 2j).

Runs upstream inclusionAI/Duo's DuoTransformer (duo/models/realization/modules/model.py, with its causal build_mask) on one fixed packed layout
with seeded random weights, and writes the weights already renamed to the H3 names MiniMaxH3Transformer loads, the layout, the inputs and the
expected video output. The C# test rebuilds the same forward and compares it with this file.

Run in the hartsy-ref:cpu image, never on the Windows host. --src is a read-only checkout of the pinned commit; the image has no diffusers, so
install the pinned version in the throwaway container first (diffusers==0.41.0, peft==0.20.0, PyYAML==6.0.3). Nothing is downloaded here.

  python -I duo_l1_oracle.py --src /duo --out duo_l1_expected.json --commit <sha>
"""

import argparse
import json
import sys

import torch


def h3_name(name):
    """Upstream diffusers DuoTransformer parameter name -> the H3 name the C# forward loads. Returns None for names it does not load."""
    simple = {
        "proj_in": "video_patch_proj",
        "audio_proj_in": "audio_patch_proj",
        "context_embedder": "condition_proj",
        "time_embedder.linear_1": "time_embedder.proj_in",
        "time_embedder.linear_2": "time_embedder.proj_out",
        "norm_out.norm": "final_layer.norm",
        "norm_out.linear": "final_layer.adaln_proj.linear",
        "proj_out": "final_layer.video_out",
        "audio_proj_out": "final_layer.audio_out",
    }
    for upstream, h3 in simple.items():
        if name == upstream or name.startswith(upstream + "."):
            return h3 + name[len(upstream):]
    if name.startswith("token_refiner.refiner_blocks."):
        rest = name[len("token_refiner.refiner_blocks."):]
        return "token_refiner.blocks." + rest
    if name.startswith("token_refiner.final_norm"):
        return name
    if name.startswith("transformer_blocks."):
        return "blocks." + name[len("transformer_blocks."):]
    return None


def fuse_block(state, prefix):
    """Returns the H3 tensors for one block: qkv rows [q; k; v], fc1 gate rows first (diffusers SwiGLU: value = first half, gate = second)."""
    out = {}
    qkv = torch.cat([state[f"{prefix}.attn.to_q.weight"], state[f"{prefix}.attn.to_k.weight"], state[f"{prefix}.attn.to_v.weight"]], dim=0)
    out[f"{prefix}.attn.qkv_proj.weight"] = qkv
    out[f"{prefix}.attn.q_norm.weight"] = state[f"{prefix}.attn.norm_q.weight"]
    out[f"{prefix}.attn.k_norm.weight"] = state[f"{prefix}.attn.norm_k.weight"]
    out[f"{prefix}.attn.out_proj.weight"] = state[f"{prefix}.attn.to_out.0.weight"]
    proj = state[f"{prefix}.ff.net.0.proj.weight"]
    half = proj.shape[0] // 2
    out[f"{prefix}.mlp.fc1.weight"] = torch.cat([proj[half:], proj[:half]], dim=0)
    out[f"{prefix}.mlp.fc2.weight"] = state[f"{prefix}.ff.net.2.weight"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--commit", required=True)
    args = ap.parse_args()
    sys.path.insert(0, args.src)

    from duo.models.realization.modules.model import DuoTransformer

    torch.manual_seed(7)
    model = DuoTransformer(
        num_attention_heads=2, attention_head_dim=96, hidden_size=8, num_layers=1, num_refiner_layers=1,
        ffn_dim=16, in_channels=24, audio_in_channels=32, patch_size=(1, 2, 2), text_dim=5, freq_dim=256,
        time_embed_hidden_dim=8, time_embed_dim=6, rope_freq_dim=16, rope_theta=10000.0,
        norm_eps=1e-5, qk_norm_eps=1e-5, final_norm_eps=1e-5,
    ).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name:
                p.copy_(1.0 + 0.05 * torch.randn_like(p))
            else:
                p.copy_(0.2 * torch.randn_like(p))

    # One packed layout, the same as the C# test: shared text rows 0-1, reference video rows 2-3, noisy video rows 4-7 (block 0, time 1).
    class Sequence:
        pass

    s = Sequence()
    s.device = torch.device("cpu")
    s.length = 8
    s.groups = [
        {"kind": "shared", "block": 0, "sample": 0, "start": 0, "end": 2},
        {"kind": "reference", "block": -1, "sample": 0, "start": 2, "end": 4},
        {"kind": "noisy", "block": 0, "sample": 0, "start": 4, "end": 8},
    ]
    s.refiner_groups = [{"kind": "text", "block": 0, "sample": 0, "start": 0, "end": 2}]
    s.refiner_text = torch.randn(1, 2, 5, generator=torch.Generator().manual_seed(11))
    s.text_indices = torch.tensor([0, 1])
    s.text_selection = torch.tensor([0, 1])
    s.video_indices = torch.tensor([2, 3, 4, 5, 6, 7])
    s.audio_indices = torch.tensor([], dtype=torch.long)
    s.video = torch.randn(1, 6, 96, generator=torch.Generator().manual_seed(12))
    s.audio = torch.zeros(1, 0, 32)
    s.positions = torch.tensor([[r * 0.5, float(r % 2), float(r // 2)] for r in range(8)])
    s.time_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    s.tags = torch.tensor([1, 1, 0, 0, 0, 0, 0, 0])
    s.roles = torch.full((8,), -1, dtype=torch.long)
    times = torch.tensor([0.0, 0.7])

    with torch.no_grad():
        (video, audio), _ = model(s, times)

    state = model.state_dict()
    h3 = {}
    for name, tensor in state.items():
        if name.startswith("transformer_blocks.") or name.startswith("token_refiner.refiner_blocks."):
            continue  # fused below
        mapped = h3_name(name)
        if mapped is not None:
            h3[mapped] = tensor
    for upstream_prefix, h3_prefix in (("transformer_blocks.0", "blocks.0"), ("token_refiner.refiner_blocks.0", "token_refiner.blocks.0")):
        fused = fuse_block(state, upstream_prefix)
        for key, value in fused.items():
            h3[key.replace(upstream_prefix, h3_prefix, 1)] = value
        for rest in ("norm1.weight", "norm2.weight", "adaln_proj.linear.weight", "adaln_proj.linear.bias"):
            if f"{upstream_prefix}.{rest}" in state:
                h3[f"{h3_prefix}.{rest}"] = state[f"{upstream_prefix}.{rest}"]

    def pack(t):
        return {"shape": list(t.shape), "data": [float(x) for x in t.detach().reshape(-1).tolist()]}

    out = {
        "source": "inclusionAI/Duo",
        "commit": args.commit,
        "config": {
            "hiddenSize": 8, "numLayers": 1, "tokenRefinerNumLayers": 1, "numAttentionHeads": 2, "attentionHeadDim": 96,
            "ffnHiddenSize": 16, "textDim": 5, "videoPatchDim": 96, "audioDim": 32, "timeEmbedHiddenSize": 8,
            "timeEmbedDim": 6, "normEps": 1e-5, "qkNormEps": 1e-5, "finalNormEps": 1e-5, "ropeInvFreqLen": 16,
        },
        "layout": {
            "groups": s.groups,
            "tags": s.tags.tolist(), "timeIds": s.time_ids.tolist(), "positions": s.positions.reshape(-1).tolist(),
            "videoRows": s.video_indices.tolist(), "audioRows": [], "textRows": s.text_indices.tolist(),
            "textSourceRows": s.text_selection.tolist(), "refinerGroups": s.refiner_groups,
        },
        "times": times.tolist(),
        "textStates": pack(s.refiner_text[0]),
        "video": pack(s.video[0]),
        "weights": {name: pack(t) for name, t in h3.items()},
        "expectedVideo": pack(video[0]),
        "expectedAudioRows": int(audio.shape[1]),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print(f"wrote {args.out}: {len(h3)} weights, video {tuple(video.shape)}")


if __name__ == "__main__":
    main()
