"""Runs the pinned base LongCat-Video reference (meituan-longcat/LongCat-Video @ 58eb336) for a comparison with the engine, on a card that cannot
hold its BF16 DiT and umT5 together: the text is encoded once and umT5 freed, and the DiT is spread over the GPU and the host with accelerate,
block by block. The pipeline, scheduler, VAE and DiT are the reference's own; two things differ from run_demo_text_to_video.py, and both are
written into report.json:

- attention is torch SDPA (flash_attn and flash_attn_interface are replaced; block-sparse attention is not used), and
- one process, no torch.distributed, no context parallel.

The run starts from a saved latent (--noise, or one drawn here and saved), so the engine can start from the same array:
`hartsy video ... --set diagnostics.longcatNoiseNpy=<out>/noise.npy`.

    python run_base_reference.py --repo /path/LongCat-Video --checkpoint /path/models/LongCat-Video --out /path/run1

Defaults: the engine comparison's prompt, the demo's negative prompt, 832x480, 93 frames, 50 steps, guidance 4, seed 1234, BF16, GPU 0 with
13GiB for the DiT's weights and 40GiB of host memory. The rest of a 24 GB card is for the forward: an estimate, not a measurement (about 8 GB of
activations at 37,440 tokens with two guidance passes). If the first step runs out of GPU memory, lower --gpu-memory by 2GiB. --distill runs the demo's distilled call instead: lora/cfg_step_lora.safetensors, 16 steps, guidance 1.

Writes noise.npy and final_latent.npy (float32, [1, 16, latent frames, height / 8, width / 8]), the encoded text (prompt_embeds.npy,
prompt_mask.npy and, with guidance, the negative pair), frame_000.png onward (lossless, converted to 8 bits as the demo converts them),
report.json, and with --dump-steps steps/step_000.npy onward (the latent after each step).
"""

import argparse
import gc
import json
import os
import subprocess
import sys
import time
import types

import numpy as np
import torch
import torch.nn.functional as F

PROMPT = "a lighthouse on a cliff at dusk, waves breaking below, slow camera move"
# run_demo_text_to_video.py line 29.
NEGATIVE_PROMPT = ("Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst "
                   "quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, "
                   "deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the "
                   "background, walking backwards")


def install_stubs():
    """flash-attn and block-sparse attention, which the reference imports and a runner without them replaces."""
    # diffusers looks for an installed flash_attn when it is first imported, and a module without a spec breaks that look: it goes first.
    import diffusers.utils  # noqa: F401

    def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, **_):  # [B, S, H, D]
        assert not causal, "the SDPA stub has no causal mask"
        out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale=softmax_scale)
        return out.transpose(1, 2)

    def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q=None, max_seqlen_k=None, softmax_scale=None,
                               causal=False, **_):
        # Packed [S, H, D] with one segment per batch item (guidance runs two): attention within each segment, as flash-attn does.
        assert not causal, "the SDPA stub has no causal mask"
        outs = []
        for i in range(len(cu_seqlens_q) - 1):
            qi = q[cu_seqlens_q[i]:cu_seqlens_q[i + 1]].transpose(0, 1)[None]
            ki = k[cu_seqlens_k[i]:cu_seqlens_k[i + 1]].transpose(0, 1)[None]
            vi = v[cu_seqlens_k[i]:cu_seqlens_k[i + 1]].transpose(0, 1)[None]
            outs.append(F.scaled_dot_product_attention(qi, ki, vi, scale=softmax_scale)[0].transpose(0, 1))
        return torch.cat(outs, dim=0).contiguous()

    flash = types.ModuleType("flash_attn")
    flash.flash_attn_func = flash_attn_func
    flash.flash_attn_varlen_func = flash_attn_varlen_func
    # The flash-attn 3 interface returns a tuple.
    flash3 = types.ModuleType("flash_attn_interface")
    flash3.flash_attn_func = lambda *a, **k: (flash_attn_func(*a, **k),)
    flash3.flash_attn_varlen_func = lambda *a, **k: (flash_attn_varlen_func(*a, **k),)

    def refuse(*_a, **_k):
        raise RuntimeError("block-sparse attention is not part of this run")

    bsa = types.ModuleType("longcat_video.block_sparse_attention.bsa_interface")
    bsa.flash_attn_bsa_3d = refuse
    sys.modules.update({"flash_attn": flash, "flash_attn_interface": flash3,
                        "longcat_video.block_sparse_attention.bsa_interface": bsa})


def checkout_head(repo):
    """The reference checkout's commit, or None where git cannot say."""
    try:
        return subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() or None
    except FileNotFoundError:
        return None


def free(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    parser.add_argument("--repo", required=True, help="the pinned reference checkout")
    parser.add_argument("--checkpoint", required=True, help="the LongCat-Video model folder: tokenizer, text_encoder, vae, scheduler, dit, lora")
    parser.add_argument("--out", required=True)
    parser.add_argument("--noise", help="a float32 .npy of the starting latent; drawn from --seed and saved when absent")
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--negative-prompt", default=NEGATIVE_PROMPT)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--frames", type=int, default=93)
    parser.add_argument("--steps", type=int, help="50, or 16 with --distill")
    parser.add_argument("--guidance", type=float, help="4.0, or 1.0 with --distill")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--distill", action="store_true")
    parser.add_argument("--lora-dim", type=int, default=128, help="the LoRA's rank; the release's is the model's default")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "f32"), default="bf16")
    parser.add_argument("--gpu-memory", default="13GiB", help="what the DiT's weights may take of the GPU")
    parser.add_argument("--cpu-memory", default="40GiB")
    parser.add_argument("--dump-steps", action="store_true")
    args = parser.parse_args()

    steps = args.steps if args.steps is not None else (16 if args.distill else 50)
    guidance = args.guidance if args.guidance is not None else (1.0 if args.distill else 4.0)
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    # The reference's float32 regions are CUDA autocast: off the card, BF16 weights meet float32 inputs and fail.
    if args.dtype == "bf16" and device.type != "cuda":
        parser.error("--dtype bf16 needs a CUDA device; use --dtype f32 on the CPU")
    os.makedirs(args.out, exist_ok=True)
    started = time.time()
    times = {}

    def mark(name):
        times[name] = round(time.time() - started, 2)
        print(f"[runner] {name} at {times[name]} s", flush=True)

    sys.path.insert(0, args.repo)
    install_stubs()
    from transformers import AutoTokenizer, UMT5EncoderModel
    from longcat_video.pipeline_longcat_video import LongCatVideoPipeline
    from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
    from longcat_video.modules.longcat_video_dit import LongCatVideoTransformer3DModel
    from longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, subfolder="tokenizer")
    vae = AutoencoderKLWan.from_pretrained(args.checkpoint, subfolder="vae", torch_dtype=dtype)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.checkpoint, subfolder="scheduler")
    text_encoder = UMT5EncoderModel.from_pretrained(args.checkpoint, subfolder="text_encoder", torch_dtype=dtype).to(device).eval()
    pipe = LongCatVideoPipeline(tokenizer=tokenizer, text_encoder=text_encoder, vae=vae, scheduler=scheduler, dit=None)
    pipe.device = device

    # 1. The text, once, and umT5 gone before the DiT arrives.
    guided = guidance > 1
    with torch.no_grad():
        encoded = pipe.encode_prompt(prompt=args.prompt, negative_prompt=args.negative_prompt, do_classifier_free_guidance=guided,
                                     num_videos_per_prompt=1, max_sequence_length=512, dtype=dtype, device=device)
    encoded = tuple(None if value is None else value.cpu() for value in encoded)
    for name, value in zip(("prompt_embeds", "prompt_mask", "negative_prompt_embeds", "negative_prompt_mask"), encoded):
        if value is not None:
            np.save(os.path.join(args.out, name + ".npy"), value.float().numpy() if value.is_floating_point() else value.numpy())
    pipe.text_encoder = None
    del text_encoder
    free(device)
    pipe.encode_prompt = lambda **_: tuple(None if value is None else value.to(device) for value in encoded)
    mark("text encoded")

    # 2. The DiT on the host, its LoRA attached, then spread block by block.
    dit = LongCatVideoTransformer3DModel.from_pretrained(args.checkpoint, subfolder="dit", cp_split_hw=[1, 1], torch_dtype=dtype,
                                                         low_cpu_mem_usage=True).eval()
    if args.distill:
        dit.load_lora(os.path.join(args.checkpoint, "lora", "cfg_step_lora.safetensors"), "cfg_step_lora", lora_network_dim=args.lora_dim)
        dit.enable_loras(["cfg_step_lora"])
    lora_networks = list(dit.lora_dict.values())
    mark("DiT loaded")
    placement = {"": str(device)}
    if device.type == "cuda":
        from accelerate import dispatch_model, infer_auto_device_map
        device_map = infer_auto_device_map(dit, max_memory={device.index or 0: args.gpu_memory, "cpu": args.cpu_memory},
                                           no_split_module_classes=["LongCatSingleStreamBlock"], dtype=dtype)
        placed = sorted({str(where) for where in device_map.values()})
        assert "disk" not in placed, f"the DiT does not fit {args.gpu_memory} of GPU and {args.cpu_memory} of host memory"
        dit = dispatch_model(dit, device_map=device_map, main_device=device)
        placement = {where: sum(1 for value in device_map.values() if str(value) == where) for where in placed}
        print(f"[runner] placement (modules per device): {placement}", flush=True)
        # The LoRA modules are not the model's children: nothing else moves them.
        for network in lora_networks:
            for lora in network.loras:
                lora.to(device, dtype=dtype)
    pipe.dit = dit
    pipe.vae = vae

    # 3. The starting latent, saved whatever its source.
    shape = (1, dit.config.in_channels, (args.frames - 1) // pipe.vae_scale_factor_temporal + 1,
             args.height // pipe.vae_scale_factor_spatial, args.width // pipe.vae_scale_factor_spatial)
    if args.noise:
        noise = np.load(args.noise)
        assert noise.dtype == np.float32 and noise.shape == shape, f"{args.noise} is {noise.dtype} {noise.shape}; this run needs float32 {shape}"
        latents = torch.from_numpy(noise.copy())
    else:
        generator = torch.Generator(device=device).manual_seed(args.seed)
        latents = torch.randn(shape, generator=generator, device=device, dtype=torch.float32).cpu()
    np.save(os.path.join(args.out, "noise.npy"), latents.numpy().astype(np.float32))

    step_seconds = []
    own_step = scheduler.step
    last = [time.time()]

    def step(*a, **k):
        result = own_step(*a, **k)
        now = time.time()
        step_seconds.append(round(now - last[0], 3))
        last[0] = now
        index = len(step_seconds) - 1
        if args.dump_steps:
            os.makedirs(os.path.join(args.out, "steps"), exist_ok=True)
            np.save(os.path.join(args.out, "steps", f"step_{index:03d}.npy"), result[0].detach().float().cpu().numpy())
        if index == 0:
            print(f"[runner] first step {step_seconds[0]} s; {steps} steps predict {step_seconds[0] * steps / 60:.1f} min", flush=True)
        return result

    scheduler.step = step
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    last[0] = time.time()
    with torch.no_grad():
        final = pipe.generate_t2v(prompt=args.prompt, negative_prompt=args.negative_prompt, height=args.height, width=args.width,
                                  num_frames=args.frames, num_inference_steps=steps, use_distill=args.distill, guidance_scale=guidance,
                                  latents=latents.to(device), output_type="latent")
    mark("denoised")
    np.save(os.path.join(args.out, "final_latent.npy"), final.float().cpu().numpy())
    denoise_peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None

    # 4. The DiT gone before the decode, which then has the card.
    pipe.dit = None
    del dit
    free(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    vae = vae.to(device).eval()
    with torch.no_grad():
        decoded = vae.decode(pipe.denormalize_latents(final.to(device, vae.dtype)), return_dict=False)[0]
        # In the VAE's own dtype, as the pipeline passes it on: the processor maps to [0, 1] in that dtype.
        frames = pipe.video_processor.postprocess_video(decoded, output_type="np")[0]
    from PIL import Image
    for index, frame in enumerate(frames):
        # The demo's conversion: times 255, clamped, truncated.
        Image.fromarray(np.clip(frame * 255, 0, 255).astype(np.uint8)).save(os.path.join(args.out, f"frame_{index:03d}.png"))
    mark("decoded")

    import resource
    report = {
        "reference_head": checkout_head(args.repo),
        "differences_from_the_demo": ["SDPA in place of flash-attn", "one process without torch.distributed",
                                      "DiT spread over devices with accelerate", "umT5 freed before the DiT is loaded"],
        "arguments": {**vars(args), "steps": steps, "guidance": guidance},
        "scheduler_shift": scheduler.config.shift,
        "latent_shape": list(shape),
        "noise_source": args.noise or f"torch.Generator({device.type}) seed {args.seed}",
        "frames_written": len(frames),
        "placement": placement,
        "lora_modules": sum(len(network.loras) for network in lora_networks),
        "seconds": times,
        "step_seconds": step_seconds,
        "peak_gpu_bytes_denoise": denoise_peak,
        "peak_gpu_bytes_decode": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "peak_host_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
    with open(os.path.join(args.out, "report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({"frames": len(frames), "steps": len(step_seconds), "seconds": times}), flush=True)


if __name__ == "__main__":
    main()
