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
    # Window latents come last in the draw order, so every tensor above keeps its value.
    window_a, window_b, window_c, window_d = (torch.randn(1, LATENT, FRAMES, HEIGHT, WIDTH, generator=generator) for _ in range(4))

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
    # Per-frame timesteps: deterministic values, drawn without the generator so the clip-wide keys above are unchanged.
    frame_t = torch.tensor([[700.0, 537.0, 300.0]])
    with torch.no_grad():
        velocity_per_frame = _forward_per_frame(model, torch.cat([latent, zero_cond], 1), frame_t, t, qwen, byt5)
    tensors["ref.velocity_per_frame"] = velocity_per_frame.contiguous()
    tensors["in.frame_timesteps"] = frame_t
    with torch.no_grad():
        tensors.update(_window_cases(model, window_a, window_b, window_c, frame_t, t, qwen, byt5))
        tensors.update(_stream_cases(model, [window_a, window_b, window_c, window_d], frame_t, t, qwen, byt5))
    tensors["in.latent"] = latent
    tensors["in.qwen"] = qwen
    tensors["in.byt5"] = byt5
    tensors["in.conditioning"] = cond
    tensors["in.timestep"] = t
    for name, value in model.state_dict().items():
        tensors[name] = value.float().contiguous()
    save_file(tensors, args.out, metadata={"source": "MovieRoll 0b5f73e building blocks + SDPA attention", "hidden": str(HIDDEN)})
    print("wrote", args.out, len(tensors), "tensors")


def _forward_per_frame(model, x65, frame_t, t, qwen, byt5):
    """The clip with one timestep per latent frame (image modulation and final layer per frame), text at the window timestep."""
    thw = [FRAMES, HEIGHT, WIDTH]
    img = model.img_in(x65)
    vec = model.time_in(frame_t.reshape(-1)).reshape(1, FRAMES, -1)
    vec_txt = model.time_in(t)[:, None, :]
    txt = model.txt_in(qwen, t) + model.cond_type_embedding.weight[0]
    byt = model.byt5_in(byt5) + model.cond_type_embedding.weight[1]
    txt = torch.cat([byt, txt], 1)
    cos, sin = get_nd_rotary_pos_embed(ROPE_DIMS, (FRAMES, HEIGHT, WIDTH), theta=256, device="cpu")
    for block in model.double_blocks:
        img, txt = block(img, txt, vec, vec_txt, (cos, sin), thw)
    out = model.final_layer(img, vec, thw=thw)
    return out.reshape(1, FRAMES, HEIGHT, WIDTH, LATENT).permute(0, 4, 1, 2, 3)


WINDOW_MAX_FRAMES = 9


def _window_block(block, img, txt, vec, vec_txt, thw, cos, sin, q_rows, k_rows, cache, return_kv):
    """MovieRoll's forward_inference for one double block. The cache holds the history's post-norm, pre-RoPE K and V; the window's
    K and V are appended to it, then RoPE runs on the concatenation at the packed rows ``k_rows`` (the window's Q at ``q_rows``).
    The text stream is skipped under ``return_kv``, which returns the window's own pre-RoPE K and V for the cache."""
    need_txt = not return_kv
    i1s, i1c, i1g, i2s, i2c, i2g = block.img_mod(vec).chunk(6, dim=-1)
    img_q, img_k, img_v = block.qkv("img", modulate(block.img_norm1(img), shift=i1s, scale=i1c, thw=thw))
    kv_out = (img_k.clone(), img_v.clone()) if return_kv else None
    k_full, v_full = img_k, img_v
    if cache is not None:
        k_full = torch.cat([cache[0], img_k], 1)
        v_full = torch.cat([cache[1], img_v], 1)
    img_q = apply_rotary_emb(img_q, (cos[q_rows], sin[q_rows]))
    k_full = apply_rotary_emb(k_full, (cos[k_rows], sin[k_rows]))
    if need_txt:
        t1s, t1c, t1g, t2s, t2c, t2g = block.txt_mod(vec_txt).chunk(6, dim=-1)
        txt_q, txt_k, txt_v = block.qkv("txt", modulate(block.txt_norm1(txt), shift=t1s, scale=t1c, num_chunks=1))
        q = torch.cat([img_q, txt_q], 1)
        k = torch.cat([k_full, txt_k], 1)
        v = torch.cat([v_full, txt_v], 1)
    else:
        q, k, v = img_q, k_full, v_full
    attn = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2).flatten(2)
    img_len = img.shape[1]
    img = img + apply_gate(block.img_attn_proj(attn[:, :img_len]), gate=i1g, thw=thw)
    img = img + apply_gate(block.img_mlp(modulate(block.img_norm2(img), shift=i2s, scale=i2c, thw=thw)), gate=i2g, thw=thw)
    if need_txt:
        txt = txt + apply_gate(block.txt_attn_proj(attn[:, img_len:]), gate=t1g, num_chunks=1)
        txt = txt + apply_gate(block.txt_mlp(modulate(block.txt_norm2(txt), shift=t2s, scale=t2c, num_chunks=1)), gate=t2g, num_chunks=1)
        return img, txt, None
    return img, None, kv_out


def _window_cases(model, window_a, window_b, window_c, frame_t, t, qwen, byt5):
    """Block 0 of the inference-regime window, against the MovieRoll cache semantics. Window A is encoded at packed frames 0..2 and
    kept as history. Window B is encoded at frames 6..8 and kept too. Window C is denoised at frames 3..5 over A's cache (packed
    where it was captured), and again over B's cache, whose absolute position (6..8) differs from its packed position (0..2)."""
    thw = [FRAMES, HEIGHT, WIDTH]
    cos, sin = get_nd_rotary_pos_embed(ROPE_DIMS, (WINDOW_MAX_FRAMES, HEIGHT, WIDTH), theta=256, device="cpu")
    block = model.double_blocks[0]
    per_frame = HEIGHT * WIDTH

    def rows(first_frame):
        return torch.arange(first_frame * per_frame, (first_frame + FRAMES) * per_frame)

    zero_cond = torch.zeros(1, LATENT + 1, FRAMES, HEIGHT, WIDTH)
    img_a = model.img_in(torch.cat([window_a, zero_cond], 1))
    img_b = model.img_in(torch.cat([window_b, zero_cond], 1))
    img_c = model.img_in(torch.cat([window_c, zero_cond], 1))
    vec = model.time_in(frame_t.reshape(-1)).reshape(1, FRAMES, -1)
    vec_txt = model.time_in(t)[:, None, :]
    txt = torch.cat([model.byt5_in(byt5) + model.cond_type_embedding.weight[1],
                     model.txt_in(qwen, t) + model.cond_type_embedding.weight[0]], 1)

    _, _, kv_a = _window_block(block, img_a, None, vec, vec_txt, thw, cos, sin, rows(0), rows(0), None, True)
    _, _, kv_b = _window_block(block, img_b, None, vec, vec_txt, thw, cos, sin, rows(6), rows(6), None, True)
    img_d, txt_d, _ = _window_block(block, img_c, txt, vec, vec_txt, thw, cos, sin, rows(3), torch.arange(0, 6 * per_frame),
                                    kv_a, False)
    img_r, txt_r, _ = _window_block(block, img_c, txt, vec, vec_txt, thw, cos, sin, rows(3), torch.arange(0, 6 * per_frame),
                                    kv_b, False)
    tensors = {
        "win.in.img_a": img_a, "win.in.img_b": img_b, "win.in.img_c": img_c, "win.in.txt": txt,
        "win.in.vec": vec, "win.in.vec_txt": vec_txt,
        "win.encode.k": kv_a[0], "win.encode.v": kv_a[1],
        "win.encode_b.k": kv_b[0], "win.encode_b.v": kv_b[1],
        "win.denoise.img": img_d, "win.denoise.txt": txt_d,
        "win.repos.img": img_r, "win.repos.txt": txt_r,
        # The RoPE tables alone: MovieRoll's rotation of all-ones Q and K at the window rows and at the cache-then-window rows.
        "win.table.rot_ones_q": apply_rotary_emb(torch.ones(1, FRAMES * per_frame, HEADS, HEAD_DIM),
                                                 (cos[rows(3)], sin[rows(3)])).reshape(1, FRAMES * per_frame, HIDDEN),
        "win.table.rot_ones_k": apply_rotary_emb(torch.ones(1, 6 * per_frame, HEADS, HEAD_DIM),
                                                 (cos[torch.arange(0, 6 * per_frame)], sin[torch.arange(0, 6 * per_frame)]))
                                    .reshape(1, 6 * per_frame, HIDDEN),
    }
    return {name: value.float().contiguous() for name, value in tensors.items()}


STREAM_MAX_FRAMES = 4 * FRAMES
STREAM_FULL_HISTORY = 1   # block 0 reads its whole history; block 1 is sink-recent


def _stream_cases(model, windows, frame_t, t, qwen, byt5):
    """A rollout of four windows through both blocks, each denoised and then encoded. Block 0 (full history) keeps every window;
    block 1 (sink-recent) keeps the sink, window 0, and, before each encode, drops the previous window, so a denoise reads the sink
    and the previous window and an encode reads the sink alone. Text is the window timestep's, as in MovieRoll."""
    thw = [FRAMES, HEIGHT, WIDTH]
    cos, sin = get_nd_rotary_pos_embed(ROPE_DIMS, (STREAM_MAX_FRAMES, HEIGHT, WIDTH), theta=256, device="cpu")
    per_frame = HEIGHT * WIDTH
    zero_cond = torch.zeros(1, LATENT + 1, FRAMES, HEIGHT, WIDTH)
    vec = model.time_in(frame_t.reshape(-1)).reshape(1, FRAMES, -1)
    vec_txt = model.time_in(t)[:, None, :]
    txt0 = torch.cat([model.byt5_in(byt5) + model.cond_type_embedding.weight[1],
                      model.txt_in(qwen, t) + model.cond_type_embedding.weight[0]], 1)
    cache = [[] for _ in model.double_blocks]   # per block: [(keys, values, frames), ...] in order

    def history(entries):
        cached = sum(frames for _, _, frames in entries)
        if not entries:
            return None, cached
        return (torch.cat([k for k, _, _ in entries], 1), torch.cat([v for _, v, _ in entries], 1)), cached

    tensors = {}
    with torch.no_grad():
        for w, window in enumerate(windows):
            tensors[f"win.stream.latent{w}"] = window
            x = model.img_in(torch.cat([window, zero_cond], 1))

            img, txt = x, txt0
            for b, block in enumerate(model.double_blocks):
                kv, cached = history(cache[b])
                n = cached + FRAMES
                img, txt, _ = _window_block(block, img, txt, vec, vec_txt, thw, cos, sin,
                                            torch.arange(cached * per_frame, n * per_frame),
                                            torch.arange(0, n * per_frame), kv, False)
            velocity = model.final_layer(img, vec, thw=thw).reshape(1, FRAMES, HEIGHT, WIDTH, LATENT).permute(0, 4, 1, 2, 3)
            tensors[f"win.stream.velocity{w}"] = velocity.contiguous()

            img = x
            for b, block in enumerate(model.double_blocks):
                if b >= STREAM_FULL_HISTORY:
                    cache[b] = cache[b][:1]   # trim_to_sink before the encode
                kv, cached = history(cache[b])
                n = cached + FRAMES
                img, _, kv_out = _window_block(block, img, None, vec, vec_txt, thw, cos, sin,
                                               torch.arange(cached * per_frame, n * per_frame),
                                               torch.arange(0, n * per_frame), kv, True)
                cache[b].append((kv_out[0], kv_out[1], FRAMES))
    return {name: value.float().contiguous() for name, value in tensors.items()}


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
