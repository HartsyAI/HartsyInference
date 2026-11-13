"""Reference values for one whole denoising step of Hunyuan3D-Paint 2.1, produced by upstream's own modules on the CPU: the VAE encodes of the reference and the geometry maps, the three
guidance branches through the real UNet, the view-dependent guidance, one `UniPCMultistepScheduler` step and the VAE decode, chained as `hunyuanpaintpbr/pipeline.py` chains them.

It transcribes `HunyuanPaintPipeline.__call__` and `denoise` of a checkout of Tencent-Hunyuan/Hunyuan3D-2.1 (`hy3dpaint`, pinned in docs/Research/HUNYUAN3D_PAINT_ARCHITECTURE.md) for a
single-view-batch run with `guidance_scale > 1` and `num_inference_steps = 1`, with the modules of `dump_paint21_unet.py` (the released UNet and VAE read by its restricted unpickler,
upstream's `unet/modules.py` and `unet/attn_processor.py` run as they are) and diffusers' `UniPCMultistepScheduler` built from the released `scheduler_config.json` with trailing spacing:

* the reference, normal and position maps are encoded with the posterior MEAN, where upstream draws `posterior.sample()` (the noise it adds is below 5e-4 of the latent RMS, see the
  VAE encode row of PARITY_VERIFICATION.md), so that the fixture is deterministic;
* the DINO hidden states are random fixture tokens (the encoder has its own comparison); the UNet projects them (`image_proj_model_dino`) and the projections are saved;
* the geometry maps are a synthetic sphere seen from six views, the reference a real picture, the start latents seeded noise: the point is the chain, not an image.

    python dump_paint21_step.py --upstream <hy3dpaint checkout>/hunyuanpaintpbr --folder <hunyuan3d-paint21> --reference <picture.png> --out <fixture.safetensors>

Fixture keys (the views are the six fixed ones, 128 px, latents 16 x 16, two materials, so batch index = material * 6 + view): `ref_u8` `[128, 128, 3]`, `normal_u8` and `position_u8`
`[6, 128, 128, 3]` (0 to 255 as floats), `azimuths` `[6]`, `dino_proj_cond` and `dino_proj_uncond` `[257, 1024]`, `init_latents` `[12, 4, 16, 16]`, `timestep`, `noise_uncond`, `noise_ref`,
`noise_full`, `noise_guided` (each `[12, 4, 16, 16]`), `latents_out` `[12, 4, 16, 16]`, `decoded_u8` `[12, 128, 128, 3]` (0 to 255 as floats).
"""
import argparse
import importlib.util
import json
import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def load_unet_oracle():
    spec = importlib.util.spec_from_file_location("dump_paint21_unet", os.path.join(HERE, "dump_paint21_unet.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cam_mapping(azim):
    """pipeline.py, copied: the view scale of the guidance."""
    if 0 <= azim < 90:
        return float(azim) / 90.0 + 1
    if 90 <= azim < 330:
        return 2.0
    return -float(azim) / 90.0 + 5.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--folder", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--guidance", type=float, default=3.0)
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    from diffusers import AutoencoderKL, UNet2DConditionModel, UniPCMultistepScheduler
    from PIL import Image
    from safetensors.torch import save_file

    base = load_unet_oracle()
    modules = base.load_upstream(args.upstream)
    size, h, n, n_pbr = 128, 16, 6, 2
    azimuths = [0, 90, 180, 270, 0, 180]  # the fixed views: front, right, back, left, top, bottom

    unet = UNet2DConditionModel.from_config(json.load(open(os.path.join(args.folder, "unet", "config.json"))))
    wrapped = modules.UNet2p5DConditionModel(unet)
    wrapped.unet.conv_in = torch.nn.Conv2d(12, unet.conv_in.out_channels, kernel_size=3, padding=1)
    state = {k: v.float() for k, v in base.read_state_dict(os.path.join(args.folder, "unet", "diffusion_pytorch_model.bin")).items()}
    missing, unexpected = wrapped.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(f"UNet weights do not match: missing {missing[:5]}, unexpected {unexpected[:5]}")
    wrapped.eval()
    vae = AutoencoderKL.from_config(json.load(open(os.path.join(args.folder, "vae", "config.json"))))
    vae_state = {base.modern_vae_name(k): v.float() for k, v in base.read_state_dict(os.path.join(args.folder, "vae", "diffusion_pytorch_model.bin")).items()}
    missing, unexpected = vae.load_state_dict(vae_state, strict=False)
    if missing or unexpected:
        raise SystemExit(f"VAE weights do not match: missing {missing[:5]}, unexpected {unexpected[:5]}")
    vae.eval()
    scaling = vae.config.scaling_factor
    scheduler = UniPCMultistepScheduler.from_config(json.load(open(os.path.join(args.folder, "scheduler", "scheduler_config.json"))), timestep_spacing="trailing")
    scheduler.set_timesteps(1)
    t = scheduler.timesteps[0]

    # Inputs as 8-bit pictures, the way the C# pipeline receives them.
    pos = base.position_maps(n, size)[0]  # [n, 3, S, S] in [0, 1]
    position_u8 = (pos.permute(0, 2, 3, 1) * 255).round().clamp(0, 255)
    normal_u8 = position_u8.flip(-1)  # a different 3-channel picture of the same sphere
    reference = Image.open(args.reference).convert("RGB").resize((size, size), Image.BICUBIC)
    ref_u8 = torch.from_numpy(np.asarray(reference, dtype=np.float32))

    def encode(images_u8):  # [k, S, S, 3] in 0..255 -> [1, k, 4, h, h] (posterior mean x scaling)
        x = (images_u8.permute(0, 3, 1, 2) / 255.0 - 0.5) * 2.0
        return (vae.encode(x).latent_dist.mean * scaling).unsqueeze(0)

    ref_latents = encode(ref_u8.unsqueeze(0))
    embeds_normal, embeds_position = encode(normal_u8), encode(position_u8)
    generator = torch.Generator().manual_seed(11)
    dino = torch.randn(1, 257, 1536, generator=generator)
    init = torch.randn(n_pbr * n, 4, h, h, generator=generator)

    text = torch.stack([wrapped.unet.learned_text_clip_albedo, wrapped.unet.learned_text_clip_mr], dim=0).unsqueeze(0)  # [1, 2, 77, 1024]
    prompt_embeds = torch.cat([text, text, text])  # negative, prompt, prompt: the learned tokens are the same for both
    # pipeline.__call__ with guidance_scale > 1
    cond = {
        "ref_latents": ref_latents.repeat(3, 1, 1, 1, 1), "ref_scale": torch.as_tensor([0.0, 1.0, 1.0]),
        "dino_hidden_states": torch.cat([torch.zeros_like(dino), torch.zeros_like(dino), dino]),
        "embeds_normal": embeds_normal.repeat(3, 1, 1, 1, 1), "embeds_position": embeds_position.repeat(3, 1, 1, 1, 1),
        "position_maps": pos.unsqueeze(0).repeat(3, 1, 1, 1, 1), "mva_scale": 1.0, "num_in_batch": n, "camera_azims": azimuths, "cache": {},
    }

    # pipeline.denoise, one step (i = 0)
    latents = init.clone()
    latents5 = latents.reshape(1, n_pbr, n, 4, h, h)
    model_input = latents5.repeat(3, 1, 1, 1, 1, 1).reshape(3 * n_pbr * n, 4, h, h)
    model_input = scheduler.scale_model_input(model_input, t).reshape(3, n_pbr, n, 4, h, h)
    start = time.time()
    noise_pred = wrapped(model_input, t, prompt_embeds, **cond)[0]
    print(f"UNet forward (3 branches, {n_pbr * n} images each): {time.time() - start:.1f} s")
    noise_uncond, noise_ref, noise_full = noise_pred.chunk(3)
    view_scale = torch.from_numpy(np.asarray([cam_mapping(a) for a in azimuths])).unsqueeze(0).repeat(n_pbr, 1).view(-1).to(noise_uncond)[:, None, None, None]
    guided = noise_uncond + args.guidance * view_scale * (noise_ref - noise_uncond)
    guided += args.guidance * view_scale * (noise_full - noise_ref)
    out_latents = scheduler.step(guided, t, latents[:, :4], return_dict=False)[0]
    decoded = vae.decode(out_latents / scaling, return_dict=False)[0]
    decoded_u8 = ((decoded / 2 + 0.5).clamp(0, 1) * 255).round().permute(0, 2, 3, 1)

    projected = cond["cache"]["dino_hidden_states_proj"]
    projected = projected.reshape(3, -1, projected.shape[-1])
    print("projected DINO tokens", tuple(projected.shape), "timestep", float(t))
    fixture = {
        "ref_u8": ref_u8, "normal_u8": normal_u8, "position_u8": position_u8, "azimuths": torch.tensor(azimuths, dtype=torch.float32),
        "dino_proj_cond": projected[2], "dino_proj_uncond": projected[0], "init_latents": init, "timestep": torch.tensor([float(t)]),
        "noise_uncond": noise_uncond, "noise_ref": noise_ref, "noise_full": noise_full, "noise_guided": guided,
        "latents_out": out_latents, "decoded_u8": decoded_u8,
    }
    save_file({k: v.detach().float().contiguous() for k, v in fixture.items()}, args.out)
    print("wrote", args.out, {k: tuple(v.shape) for k, v in fixture.items()})


if __name__ == "__main__":
    main()
