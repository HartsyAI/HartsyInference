"""Tiny-config reference for the HunyuanVideo-1.5 / MovieRoll DiT forward (one clip-wide timestep, valid text tokens only).

Run inside the CPU reference image, never the Windows venv:

  bash .claude/run-outputs/refenv/run.sh -v "<worktree>:/wt" -v "<MovieRoll checkout at 0b5f73e>:/src:ro" -- \
      python -I /wt/tests/python-reference/movieroll_reference/dump_hy15_tiny.py --out /wt/tests/HartsyInference.Diffusion.Tests/Fixtures/hy15_dit_tiny.safetensors

The building blocks (ModulateDiT, MLP, RMSNorm, TimestepEmbedder, TextProjection, ByT5Mapper, FinalLayer, modulate, apply_gate, the interleaved
RoPE helpers) are imported from the MovieRoll checkout; the attention calls, which need FlashAttention in the reference, are replaced by
torch SDPA over the same [img, txt] sequence with RoPE on the image rows only. Parameters are random (the reference zero-initialises
modulation and final layers, which would make every block the identity) and stored under the reference's own state-dict names.
"""
import argparse
import sys
from unittest.mock import MagicMock

for name in ["flash_attn", "flash_attn_interface", "flash_attn.bert_padding", "flash_attn.flash_attn_interface", "triton",
             "triton.language", "sageattention", "loguru"]:
    sys.modules[name] = MagicMock()
sys.path.insert(0, "/src")

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file

from movieroll.models.transformers.modules.activation_layers import get_activation_layer
from movieroll.models.transformers.modules.byt5_mapper import ByT5Mapper
from movieroll.models.transformers.modules.embed_layers import PatchEmbed, TimestepEmbedder, TextProjection
from movieroll.models.transformers.modules.mlp_layers import MLP, FinalLayer
from movieroll.models.transformers.modules.modulate_layers import ModulateDiT, modulate, apply_gate
from movieroll.models.transformers.modules.norm_layers import RMSNorm
from movieroll.models.transformers.modules.posemb_layers import apply_rotary_emb, get_nd_rotary_pos_embed

HIDDEN, HEADS, BLOCKS, LATENT = 32, 2, 2, 4
TEXT_DIM, BYT5_DIM, BYT5_HIDDEN, BYT5_MID = 24, 16, 40, 28
ROPE_DIMS = [4, 6, 6]
FRAMES, HEIGHT, WIDTH = 3, 2, 3
QWEN_LEN, BYT5_LEN = 5, 3
TIMESTEP = 537.0
HEAD_DIM = HIDDEN // HEADS


class RefinerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(HIDDEN, eps=1e-6)
        self.self_attn_qkv = nn.Linear(HIDDEN, 3 * HIDDEN)
        self.self_attn_proj = nn.Linear(HIDDEN, HIDDEN)
        self.norm2 = nn.LayerNorm(HIDDEN, eps=1e-6)
        self.mlp = MLP(in_channels=HIDDEN, hidden_channels=4 * HIDDEN, act_layer=get_activation_layer("silu"))
        self.adaLN_modulation = nn.Sequential(get_activation_layer("silu")(), nn.Linear(HIDDEN, 2 * HIDDEN))

    def forward(self, x, c):
        gate_msa, gate_mlp = self.adaLN_modulation(c).chunk(2, dim=1)
        q, k, v = self.self_attn_qkv(self.norm1(x)).unflatten(-1, (3, HEADS, -1)).unbind(2)
        attn = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2).flatten(2)
        x = x + apply_gate(self.self_attn_proj(attn), gate_msa)
        return x + apply_gate(self.mlp(self.norm2(x)), gate_mlp)


class Refiner(nn.Module):
    """SingleTokenRefiner with all text tokens valid (the mask is all ones, so the masked mean is the plain mean)."""

    def __init__(self):
        super().__init__()
        act = get_activation_layer("silu")
        self.input_embedder = nn.Linear(TEXT_DIM, HIDDEN)
        self.t_embedder = TimestepEmbedder(HIDDEN, act)
        self.c_embedder = TextProjection(TEXT_DIM, HIDDEN, act)
        self.individual_token_refiner = nn.Module()
        self.individual_token_refiner.blocks = nn.ModuleList([RefinerBlock() for _ in range(2)])

    def forward(self, x, t):
        c = self.t_embedder(t) + self.c_embedder(x.mean(dim=1))
        x = self.input_embedder(x)
        for block in self.individual_token_refiner.blocks:
            x = block(x, c)
        return x


class Block(nn.Module):
    """MMDoubleStreamBlock parameters (reference names) with the forward_inference path minus the KV cache."""

    def __init__(self):
        super().__init__()
        silu = get_activation_layer("silu")
        gelu_tanh = get_activation_layer("gelu_tanh")
        for stream in ("img", "txt"):
            setattr(self, f"{stream}_mod", ModulateDiT(HIDDEN, factor=6, act_layer=silu))
            setattr(self, f"{stream}_norm1", nn.LayerNorm(HIDDEN, elementwise_affine=False, eps=1e-6))
            setattr(self, f"{stream}_attn_qkv", nn.Linear(HIDDEN, 3 * HIDDEN, bias=True))
            setattr(self, f"{stream}_attn_q_norm", RMSNorm(HEAD_DIM, eps=1e-6))
            setattr(self, f"{stream}_attn_k_norm", RMSNorm(HEAD_DIM, eps=1e-6))
            setattr(self, f"{stream}_attn_proj", nn.Linear(HIDDEN, HIDDEN, bias=True))
            setattr(self, f"{stream}_norm2", nn.LayerNorm(HIDDEN, elementwise_affine=False, eps=1e-6))
            setattr(self, f"{stream}_mlp", MLP(HIDDEN, 4 * HIDDEN, act_layer=gelu_tanh, bias=True))

    def qkv(self, stream, x):
        q, k, v = getattr(self, f"{stream}_attn_qkv")(x).unflatten(-1, (3, HEADS, -1)).unbind(2)
        q = getattr(self, f"{stream}_attn_q_norm")(q).to(v)
        k = getattr(self, f"{stream}_attn_k_norm")(k).to(v)
        return q, k, v

    def forward(self, img, txt, vec, vec_txt, freqs, thw):
        i1s, i1c, i1g, i2s, i2c, i2g = self.img_mod(vec).chunk(6, dim=-1)
        t1s, t1c, t1g, t2s, t2c, t2g = self.txt_mod(vec_txt).chunk(6, dim=-1)
        img_q, img_k, img_v = self.qkv("img", modulate(self.img_norm1(img), shift=i1s, scale=i1c, thw=thw))
        txt_q, txt_k, txt_v = self.qkv("txt", modulate(self.txt_norm1(txt), shift=t1s, scale=t1c, num_chunks=1))
        img_q = apply_rotary_emb(img_q, freqs)
        img_k = apply_rotary_emb(img_k, freqs)
        q = torch.cat([img_q, txt_q], 1).transpose(1, 2)
        k = torch.cat([img_k, txt_k], 1).transpose(1, 2)
        v = torch.cat([img_v, txt_v], 1).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).flatten(2)
        img_len = img.shape[1]
        img_attn, txt_attn = attn[:, :img_len], attn[:, img_len:]
        img = img + apply_gate(self.img_attn_proj(img_attn), gate=i1g, thw=thw)
        img = img + apply_gate(self.img_mlp(modulate(self.img_norm2(img), shift=i2s, scale=i2c, thw=thw)), gate=i2g, thw=thw)
        txt = txt + apply_gate(self.txt_attn_proj(txt_attn), gate=t1g, num_chunks=1)
        txt = txt + apply_gate(self.txt_mlp(modulate(self.txt_norm2(txt), shift=t2s, scale=t2c, num_chunks=1)), gate=t2g, num_chunks=1)
        return img, txt


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        silu = get_activation_layer("silu")
        self.img_in = PatchEmbed([1, 1, 1], LATENT, HIDDEN)
        self.time_in = TimestepEmbedder(HIDDEN, silu)
        self.txt_in = Refiner()
        self.byt5_in = ByT5Mapper(BYT5_DIM, BYT5_MID, BYT5_HIDDEN, HIDDEN)
        self.cond_type_embedding = nn.Embedding(3, HIDDEN)
        self.double_blocks = nn.ModuleList([Block() for _ in range(BLOCKS)])
        self.final_layer = FinalLayer(HIDDEN, [1, 1, 1], LATENT, silu)

    def forward(self, x65, t, qwen, byt5, stages):
        thw = [FRAMES, HEIGHT, WIDTH]
        img = self.img_in(x65)
        stages["img_in"] = img
        vec = self.time_in(t)
        stages["time_in"] = vec
        txt = self.txt_in(qwen, t) + self.cond_type_embedding.weight[0]
        stages["txt_refined"] = txt
        byt = self.byt5_in(byt5) + self.cond_type_embedding.weight[1]
        stages["byt5_mapped"] = byt
        txt = torch.cat([byt, txt], 1)
        vec_frames = vec[:, None, :].expand(-1, FRAMES, -1)
        vec_txt = vec[:, None, :]
        cos, sin = get_nd_rotary_pos_embed(ROPE_DIMS, (FRAMES, HEIGHT, WIDTH), theta=256, device="cpu")
        for index, block in enumerate(self.double_blocks):
            img, txt = block(img, txt, vec_frames, vec_txt, (cos, sin), thw)
            stages[f"block{index}_img"] = img
            stages[f"block{index}_txt"] = txt
        out = self.final_layer(img, vec_frames, thw=thw)
        stages["final"] = out
        return out.reshape(1, FRAMES, HEIGHT, WIDTH, LATENT).permute(0, 4, 1, 2, 3)


def randomise(model, generator):
    for name, parameter in model.named_parameters():
        if name.endswith("_norm.weight") or ("norm" in name and name.endswith(".weight") and parameter.dim() == 1):
            parameter.data = 1.0 + 0.1 * torch.randn(parameter.shape, generator=generator)
        else:
            parameter.data = 0.1 * torch.randn(parameter.shape, generator=generator)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    generator = torch.Generator().manual_seed(1234)
    model = Model().float().eval()
    randomise(model, generator)

    latent = torch.randn(1, LATENT, FRAMES, HEIGHT, WIDTH, generator=generator)
    qwen = torch.randn(1, QWEN_LEN, TEXT_DIM, generator=generator)
    byt5 = torch.randn(1, BYT5_LEN, BYT5_DIM, generator=generator)
    cond = torch.randn(1, LATENT + 1, FRAMES, HEIGHT, WIDTH, generator=generator)
    t = torch.tensor([TIMESTEP])

    tensors = {}
    with torch.no_grad():
        stages = {}
        zero_cond = torch.zeros(1, LATENT + 1, FRAMES, HEIGHT, WIDTH)
        velocity_text = model(torch.cat([latent, zero_cond], 1), t, qwen, byt5, stages)
        for key, value in stages.items():
            tensors[f"ref.{key}"] = value.float().contiguous()
        tensors["ref.velocity"] = velocity_text.contiguous()
        velocity_cond = model(torch.cat([latent, cond], 1), t, qwen, byt5, {})
        tensors["ref.velocity_conditioned"] = velocity_cond.contiguous()
        # Without byT5 tokens the text stream is the refined Qwen rows alone.
        velocity_no_byt5 = _forward_without_byt5(model, torch.cat([latent, zero_cond], 1), t, qwen)
        tensors["ref.velocity_no_byt5"] = velocity_no_byt5.contiguous()
    tensors["in.latent"] = latent
    tensors["in.qwen"] = qwen
    tensors["in.byt5"] = byt5
    tensors["in.conditioning"] = cond
    tensors["in.timestep"] = t
    for name, value in model.state_dict().items():
        tensors[name] = value.float().contiguous()
    save_file(tensors, args.out, metadata={"source": "MovieRoll 0b5f73e building blocks + SDPA attention", "hidden": str(HIDDEN)})
    print("wrote", args.out, len(tensors), "tensors")


def _forward_without_byt5(model, x65, t, qwen):
    thw = [FRAMES, HEIGHT, WIDTH]
    img = model.img_in(x65)
    vec = model.time_in(t)
    txt = model.txt_in(qwen, t) + model.cond_type_embedding.weight[0]
    vec_frames = vec[:, None, :].expand(-1, FRAMES, -1)
    vec_txt = vec[:, None, :]
    cos, sin = get_nd_rotary_pos_embed(ROPE_DIMS, (FRAMES, HEIGHT, WIDTH), theta=256, device="cpu")
    for block in model.double_blocks:
        img, txt = block(img, txt, vec_frames, vec_txt, (cos, sin), thw)
    out = model.final_layer(img, vec_frames, thw=thw)
    return out.reshape(1, FRAMES, HEIGHT, WIDTH, LATENT).permute(0, 4, 1, 2, 3)


if __name__ == "__main__":
    main()
