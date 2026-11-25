"""LongCat-Video base pipeline: per-step reference dump on the tiny B1 checkpoint.

Runs the pinned reference's own LongCatVideoPipeline loops (generate_t2v, generate_i2v, generate_vc, and a two-segment continuation chain)
with the reference's own FlowMatchEulerDiscreteScheduler, DiT and CFG combination. Only the environment is stubbed, none of the numerics:
diffusers and transformers are not installed (the base classes are stubbed; the scheduler's own code runs), ftfy and loguru are stubbed,
the Wan VAE and umT5 are replaced by seams that return or accept latents (the text encoder by fixed embeddings with a valid length, the VAE by
the queue of latents each scenario hands it), and flash-attn is replaced by torch SDPA exactly as in the B1 dump.

Stubbed on purpose and therefore NOT verified here: the VAE (encode and decode numerics, Diffusers vs Comfy layout), umT5 itself, the video
pixel round trip of the continuation chain (the chain is checked in latent space), and the shipped scheduler shift (the pinned repo does not
ship scheduler_config.json; the dump uses SHIFT and records it).

Outputs (under --out): `ckpt/` copied from the B1 dump (the tiny model), `meta.json`, and raw little-endian F32 files `<scenario>.<name>.f32`.
"""
import argparse
import functools
import inspect
import json
import logging
import os
import shutil
import sys
import types
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

LONGCAT_COMMIT = "58eb336edc5c8c63389838727e40b8c18434e54e"
CONFIG = dict(
    in_channels=4, out_channels=4, hidden_size=128, depth=2, num_heads=4, caption_channels=48, mlp_ratio=4,
    adaln_tembed_dim=64, frequency_embedding_size=64, patch_size=(1, 2, 2), enable_flashattn2=True, cp_split_hw=[1, 1],
    text_tokens_zero_pad=True,
)
TEXT_LEN = 7
VALID_POS, VALID_NEG = 5, 4
SHIFT = 3.0
GUIDANCE = 4.0
STEPS = 4
HEIGHT, WIDTH = 32, 48          # pixel size: latent 4 x 6, patch grid 2 x 3
LATENT_MEAN = [0.10, -0.20, 0.30, -0.40]   # stand-ins for the VAE's latents_mean/std (length = in_channels)
LATENT_STD = [1.50, 0.80, 1.20, 0.90]
COND_CHUNK_LATENTS = 2          # continuation: 5 pixel frames -> 2 context latents (the real 13 -> 4 at 93 frames)
PROMPT, NEG_PROMPT = "a tiny positive prompt", "a tiny negative prompt"


def install_stubs(repo):
    sys.path.insert(0, repo)

    def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, **_):  # [B, S, H, D]
        assert not causal, "the SDPA stub has no causal mask"
        out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale=softmax_scale)
        return out.transpose(1, 2)

    def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale=None, causal=False, **_):
        # packed [S, H, D] with one segment per batch item (CFG runs two): attend within each segment, as flash-attn does.
        assert not causal, "the SDPA stub has no causal mask"
        cu_q = [int(x) for x in cu_seqlens_q]
        cu_k = [int(x) for x in cu_seqlens_k]
        assert len(cu_q) == len(cu_k), "query and key segment counts differ"
        outs = []
        for i in range(len(cu_q) - 1):
            qi = q[cu_q[i]:cu_q[i + 1]].transpose(0, 1)[None]
            ki = k[cu_k[i]:cu_k[i + 1]].transpose(0, 1)[None]
            vi = v[cu_k[i]:cu_k[i + 1]].transpose(0, 1)[None]
            outs.append(F.scaled_dot_product_attention(qi, ki, vi, scale=softmax_scale)[0].transpose(0, 1))
        return torch.cat(outs, dim=0).contiguous()

    flash = types.ModuleType("flash_attn")
    flash.flash_attn_func = flash_attn_func
    flash.flash_attn_varlen_func = flash_attn_varlen_func
    sys.modules["flash_attn"] = flash

    def refuse(*_a, **_k):
        raise RuntimeError("block-sparse attention is not part of this dump")

    bsa = types.ModuleType("longcat_video.block_sparse_attention.bsa_interface")
    bsa.flash_attn_bsa_3d = refuse
    sys.modules["longcat_video.block_sparse_attention.bsa_interface"] = bsa
    torch_utils = types.ModuleType("longcat_video.audio_process.torch_utils")
    torch_utils.get_attn_map_with_target = refuse
    sys.modules["longcat_video.audio_process.torch_utils"] = torch_utils
    ulysses = types.ModuleType("longcat_video.context_parallel.ulysses_wrapper")
    ulysses.ulysses_wrapper = lambda func: func
    sys.modules["longcat_video.context_parallel.ulysses_wrapper"] = ulysses

    # diffusers base classes only. register_to_config records the __init__ arguments (with defaults) as self.config, the way
    # ConfigMixin does, so the reference scheduler's own __init__ and config reads run unchanged.
    def register_to_config(init):
        signature = inspect.signature(init)

        @functools.wraps(init)
        def wrapper(self, *args, **kwargs):
            bound = signature.bind(self, *args, **kwargs)
            bound.apply_defaults()
            self.config = SimpleNamespace(**{k: v for k, v in bound.arguments.items() if k != "self"})
            return init(self, *args, **kwargs)

        return wrapper

    config_utils = types.ModuleType("diffusers.configuration_utils")
    config_utils.ConfigMixin = type("ConfigMixin", (), {})
    config_utils.register_to_config = register_to_config
    modeling_utils = types.ModuleType("diffusers.models.modeling_utils")
    modeling_utils.ModelMixin = torch.nn.Module
    utils = types.ModuleType("diffusers.utils")
    utils.BaseOutput = type("BaseOutput", (), {})
    utils.is_scipy_available = lambda: False
    utils.logging = SimpleNamespace(get_logger=logging.getLogger)
    scheduling_utils = types.ModuleType("diffusers.schedulers.scheduling_utils")
    scheduling_utils.SchedulerMixin = type("SchedulerMixin", (), {})
    video_processor = types.ModuleType("diffusers.video_processor")

    class VideoProcessor:  # the inputs the dump passes are already the tensors the pipeline would have produced
        def __init__(self, vae_scale_factor):
            self.vae_scale_factor = vae_scale_factor

        def preprocess(self, image, height=None, width=None):
            return image

        def preprocess_video(self, video, height=None, width=None):
            return video

        def postprocess_video(self, *_a, **_k):
            raise RuntimeError("the dump returns latents, not pixels")

    video_processor.VideoProcessor = VideoProcessor
    image_processor = types.ModuleType("diffusers.image_processor")
    image_processor.PipelineImageInput = object
    diffusers = types.ModuleType("diffusers")
    models = types.ModuleType("diffusers.models")
    for name, module in (("diffusers", diffusers), ("diffusers.configuration_utils", config_utils), ("diffusers.models", models),
                         ("diffusers.models.modeling_utils", modeling_utils),
                         ("diffusers.utils", utils), ("diffusers.schedulers.scheduling_utils", scheduling_utils),
                         ("diffusers.video_processor", video_processor), ("diffusers.image_processor", image_processor)):
        sys.modules[name] = module
    diffusers.ConfigMixin = config_utils.ConfigMixin
    diffusers.ModelMixin = modeling_utils.ModelMixin

    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = type("AutoTokenizer", (), {})
    transformers.UMT5EncoderModel = type("UMT5EncoderModel", (), {})
    sys.modules["transformers"] = transformers
    ftfy = types.ModuleType("ftfy")
    ftfy.fix_text = lambda text: text
    sys.modules["ftfy"] = ftfy
    loguru = types.ModuleType("loguru")
    loguru.logger = SimpleNamespace(warning=lambda *a, **k: None)
    sys.modules["loguru"] = loguru
    vae_module = types.ModuleType("longcat_video.modules.autoencoder_kl_wan")
    vae_module.AutoencoderKLWan = type("AutoencoderKLWan", (), {})
    sys.modules["longcat_video.modules.autoencoder_kl_wan"] = vae_module


class TextEncoderStub:
    dtype = torch.float32
    config = SimpleNamespace(d_model=CONFIG["caption_channels"])


class VaeStub:
    """encode() returns the next queued latent (the caller queues what the real VAE would have produced); decode() is not used (latents out)."""

    def __init__(self):
        self.config = SimpleNamespace(scale_factor_temporal=4, scale_factor_spatial=8, latents_mean=LATENT_MEAN, latents_std=LATENT_STD,
                                      z_dim=CONFIG["in_channels"])
        self.queue = []

    def encode(self, _frames):
        return SimpleNamespace(latents=self.queue.pop(0))

    def decode(self, *_a, **_k):
        raise RuntimeError("the dump returns latents, not pixels")


class DitProxy:
    """The reference DiT, unchanged; records the input and output of each denoising call (not the cache-building call)."""

    def __init__(self, model, recorder):
        self.model = model
        self.recorder = recorder
        self.config = SimpleNamespace(in_channels=CONFIG["in_channels"])
        self.cp_split_hw = CONFIG["cp_split_hw"]
        self.dtype = torch.float32

    def __call__(self, **kwargs):
        out = self.model(**kwargs)
        if not kwargs.get("return_kv", False):
            self.recorder.dit_call(kwargs["hidden_states"], kwargs["timestep"], out)
        return out


class Recorder:
    def __init__(self, out_dir, meta):
        self.out_dir, self.meta = out_dir, meta
        self.name = None
        self.calls = 0
        self.steps = 0

    def begin(self, name, **info):
        self.name, self.calls, self.steps = name, 0, 0
        self.meta["scenarios"][name] = dict(info, steps=[])

    def tensor(self, label, value):
        array = value.detach().float().cpu().contiguous().numpy()
        file = f"{self.name}.{label}.f32"
        with open(os.path.join(self.out_dir, file), "wb") as f:
            f.write(array.astype("<f4").tobytes())
        self.meta["tensors"][f"{self.name}.{label}"] = list(array.shape)

    def dit_call(self, hidden, timestep, out):
        i = self.calls
        self.calls += 1
        self.tensor(f"step{i}.dit_in", hidden)
        self.tensor(f"step{i}.timestep", timestep)
        self.tensor(f"step{i}.dit_out", out)

    def step(self, i, t, model_output, sample_in, sample_out):
        self.tensor(f"step{i}.noise", model_output)
        self.tensor(f"step{i}.sample_in", sample_in)
        self.tensor(f"step{i}.sample_out", sample_out)
        self.meta["scenarios"][self.name]["steps"].append(float(t))


def build_pipeline(ckpt_dir, recorder):
    from longcat_video.modules.longcat_video_dit import LongCatVideoTransformer3DModel
    from longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
    from longcat_video.pipeline_longcat_video import LongCatVideoPipeline
    from safetensors.torch import load_file

    model = LongCatVideoTransformer3DModel(**CONFIG).float().eval()
    index = json.load(open(os.path.join(ckpt_dir, "diffusion_pytorch_model.safetensors.index.json")))
    state = {}
    for file in sorted(set(index["weight_map"].values())):
        state.update(load_file(os.path.join(ckpt_dir, file)))
    model.load_state_dict(state, strict=True)

    scheduler = FlowMatchEulerDiscreteScheduler(shift=SHIFT)
    pipe = LongCatVideoPipeline(tokenizer=None, text_encoder=TextEncoderStub(), vae=VaeStub(), scheduler=scheduler,
                                dit=DitProxy(model, recorder))
    pipe.device = "cpu"

    def fake_prompt(prompt=None, num_videos_per_prompt=1, max_sequence_length=512, device=None, dtype=None):
        embeds, mask = (pipe.neg if prompt[0] == NEG_PROMPT else pipe.pos)
        return embeds.repeat(1, num_videos_per_prompt, 1).view(len(prompt) * num_videos_per_prompt, 1, TEXT_LEN, -1), mask

    pipe._get_t5_prompt_embeds = fake_prompt
    pipe.get_condition_shape = lambda condition, resolution, scale_factor_spatial=32: (HEIGHT, WIDTH)

    orig_step = scheduler.step

    def recording_step(model_output, timestep, sample, return_dict=False, **kwargs):
        out = orig_step(model_output, timestep, sample, return_dict=return_dict, **kwargs)
        recorder.step(recorder.steps, float(timestep), model_output, sample, out[0])
        recorder.steps += 1
        return out

    scheduler.step = recording_step
    return pipe


def text_inputs(generator):
    pos = torch.randn(TEXT_LEN, CONFIG["caption_channels"], generator=generator) * 0.5
    neg = torch.randn(TEXT_LEN, CONFIG["caption_channels"], generator=generator) * 0.5
    pos_mask = torch.tensor([1] * VALID_POS + [0] * (TEXT_LEN - VALID_POS), dtype=torch.int64)[None]
    neg_mask = torch.tensor([1] * VALID_NEG + [0] * (TEXT_LEN - VALID_NEG), dtype=torch.int64)[None]
    return (pos[None], pos_mask), (neg[None], neg_mask)


def run(args):
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    install_stubs(os.path.abspath(args.repo))
    ckpt_dst = os.path.join(out_dir, "ckpt")
    if not os.path.isdir(ckpt_dst):
        shutil.copytree(os.path.abspath(args.ckpt), ckpt_dst)

    meta = dict(commit=LONGCAT_COMMIT, torch=torch.__version__, shift=SHIFT, guidance=GUIDANCE, steps=STEPS, height=HEIGHT, width=WIDTH,
                latent_mean=LATENT_MEAN, latent_std=LATENT_STD, text_len=TEXT_LEN, valid_pos=VALID_POS, valid_neg=VALID_NEG,
                tensors={}, scenarios={})
    recorder = Recorder(out_dir, meta)
    pipe = build_pipeline(ckpt_dst, recorder)
    gen = torch.Generator().manual_seed(args.seed)
    pipe.pos, pipe.neg = text_inputs(gen)
    (pos_embeds, pos_mask), (neg_embeds, neg_mask) = pipe.pos, pipe.neg
    for label, value in (("pos_embeds", pos_embeds[0]), ("neg_embeds", neg_embeds[0])):
        recorder.name = "text"
        recorder.tensor(label, value)

    def noise(frames):
        return torch.randn(1, CONFIG["in_channels"], frames, HEIGHT // 8, WIDTH // 8, generator=gen)

    def cond_norm(frames):
        return torch.randn(1, CONFIG["in_channels"], frames, HEIGHT // 8, WIDTH // 8, generator=gen)

    def denorm(x):
        return pipe.denormalize_latents(x)

    def scenario(name, **info):
        recorder.begin(name, **info)

    def record_inputs(name, init, cond=None):
        recorder.name = name
        recorder.tensor("init", init)
        if cond is not None:
            recorder.tensor("cond", cond)

    frames_t2v = 9
    # t2v and its distilled schedule
    for name, distill in (("t2v", False), ("t2v_distill", True)):
        init = noise(3)
        scenario(name, kind="t2v", frames=frames_t2v, use_distill=distill, enhance=False)
        record_inputs(name, init)
        final = pipe.generate_t2v(prompt=PROMPT, negative_prompt=NEG_PROMPT, height=HEIGHT, width=WIDTH, num_frames=frames_t2v,
                                  num_inference_steps=STEPS, use_distill=distill, guidance_scale=GUIDANCE, latents=init.clone(),
                                  output_type="latent", max_sequence_length=TEXT_LEN)
        recorder.name = name
        recorder.tensor("final", final)
        meta["scenarios"][name]["sigmas"] = pipe.scheduler.sigmas.tolist()

    # image-to-video: the first latent frame is the conditioning image
    cond = cond_norm(1)
    init = noise(3)   # raw noise: the pipeline writes the conditioning frame into it
    scenario("i2v", kind="i2v", frames=frames_t2v, use_distill=False, enhance=False, num_cond=1)
    record_inputs("i2v", init, cond)
    pipe.vae.queue.append(denorm(cond))
    image = torch.zeros(1, 3, HEIGHT, WIDTH)
    final = pipe.generate_i2v(image=image, prompt=PROMPT, negative_prompt=NEG_PROMPT, resolution="480p", num_frames=frames_t2v,
                              num_inference_steps=STEPS, guidance_scale=GUIDANCE, latents=init.clone(), output_type="latent",
                              max_sequence_length=TEXT_LEN)
    recorder.name = "i2v"
    recorder.tensor("final", final)
    meta["scenarios"]["i2v"]["sigmas"] = pipe.scheduler.sigmas.tolist()

    # continuation (generate_vc): the reference's own default path (KV cache, enhance_hf) and the same run without the cache.
    frames_vc = 17   # latent 5: 2 context + 3 new
    video_frames = 1 + (COND_CHUNK_LATENTS - 1) * 4   # 5 pixel frames of context
    for name, kv in (("vc", True), ("vc_nocache", False)):
        ctx = cond_norm(COND_CHUNK_LATENTS)
        init = noise(5)   # raw noise; the pipeline writes the context latents into the first frames
        scenario(name, kind="vc", frames=frames_vc, num_cond=COND_CHUNK_LATENTS, use_kv_cache=kv, enhance=True)
        record_inputs(name, init, ctx)
        pipe.vae.queue.append(denorm(ctx))
        video = torch.zeros(1, 3, video_frames, HEIGHT, WIDTH)
        final = pipe.generate_vc(video=video, prompt=PROMPT, negative_prompt=NEG_PROMPT, resolution="480p", num_frames=frames_vc,
                                 num_cond_frames=video_frames, num_inference_steps=STEPS, guidance_scale=GUIDANCE, latents=init.clone(),
                                 output_type="latent", max_sequence_length=TEXT_LEN, use_kv_cache=kv, offload_kv_cache=False,
                                 enhance_hf=True)
        recorder.name = name
        recorder.tensor("final", final)
        meta["scenarios"][name]["sigmas"] = pipe.scheduler.sigmas.tolist()

    # two-segment chain: segment 2's context is the last context latents of segment 1's output (latent-space stand-in for the
    # decoded pixel frames the demo re-encodes)
    chain = []
    context = cond_norm(COND_CHUNK_LATENTS)
    for seg in range(2):
        name = f"chain{seg}"
        init = noise(5)
        scenario(name, kind="vc", frames=frames_vc, num_cond=COND_CHUNK_LATENTS, use_kv_cache=True, enhance=True)
        record_inputs(name, init, context)
        pipe.vae.queue.append(denorm(context))
        video = torch.zeros(1, 3, video_frames, HEIGHT, WIDTH)
        out = pipe.generate_vc(video=video, prompt=PROMPT, negative_prompt=NEG_PROMPT, resolution="480p", num_frames=frames_vc,
                               num_cond_frames=video_frames, num_inference_steps=STEPS, guidance_scale=GUIDANCE, latents=init.clone(),
                               output_type="latent", max_sequence_length=TEXT_LEN, use_kv_cache=True, offload_kv_cache=False,
                               enhance_hf=True)
        recorder.name = name
        recorder.tensor("final", out)
        meta["scenarios"][name]["sigmas"] = pipe.scheduler.sigmas.tolist()
        chain.append(out)
        context = out[:, :, -COND_CHUNK_LATENTS:].contiguous()

    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1, sort_keys=True)
    print("wrote", out_dir, "scenarios:", sorted(meta["scenarios"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="assembled checkout of LongCat-Video at " + LONGCAT_COMMIT)
    ap.add_argument("--ckpt", required=True, help="the B1 tiny checkpoint folder (ckpt/ from the B1 dump)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=11)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
