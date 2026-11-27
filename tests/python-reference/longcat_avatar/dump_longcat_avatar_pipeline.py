"""Reference loop for the LongCat-Video-Avatar pipeline on the tiny avatar DiT (dump_longcat_avatar_dit.py's weights and inputs).

Transcribes the denoising loops of pipeline_longcat_video_avatar.py @ 58eb336 (generate_at2v and generate_avc with the KV cache) around the
pinned model's own forward, so the C# loop can be checked step by step against the same math. The sigma list comes from the pinned
get_timesteps_sigmas (taken from the source by its AST) and the scheduler's static shift (7, the shipped 1.6 scheduler config) is applied
the way set_timesteps applies it to explicit sigmas, then the terminal zero is appended. The CFG is the reference's three-pass form; the
unconditional audio is the projection of zero raw audio, as the reference zeroes the raw audio before the DiT's projection.

Run in the hartsy-ref:cpu image only (never the Windows host). Needs the avatar DiT dump directory (--dit) and the pinned source (--repo):

    python dump_longcat_avatar_pipeline.py --repo /src --dit /out/longcat_avatar_dit_reference_tensors --out /out/longcat_avatar_dit_reference_tensors

Writes pipeline.at2v.latents, pipeline.avc.latents and pipeline.negative into the DiT dump directory (updating its meta.json).
"""

import argparse
import ast
import importlib.util
import json
import os
import sys
import textwrap

import numpy as np
import torch


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_sigma_method(repo):
    path = os.path.join(repo, "longcat_video", "pipeline_longcat_video_avatar.py")
    with open(path, encoding="utf-8") as handle:
        source = handle.read()
    for node in ast.parse(source).body:
        if isinstance(node, ast.ClassDef) and node.name == "LongCatVideoAvatarPipeline":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "get_timesteps_sigmas":
                    namespace = {"torch": torch, "np": np}
                    exec(textwrap.dedent(ast.get_source_segment(source, item)), namespace)
                    return namespace["get_timesteps_sigmas"]
    raise RuntimeError("get_timesteps_sigmas not found")


def read_tensor(dit_dir, meta, name):
    shape = meta["tensors"][name]
    values = np.fromfile(os.path.join(dit_dir, name + ".f32"), dtype=np.float32)
    return torch.from_numpy(values.reshape(shape).copy())


def sigmas_with_shift(method, steps, distill, model_type, shift):
    stand_in = type("Pipe", (), {})()
    stand_in.model_type = model_type
    stand_in.num_timesteps = 1000
    stand_in.num_distill_sample_steps = 8 if model_type == "avatar-v1.5" else 50
    base = method(stand_in, steps, use_distill=distill).float()
    shifted = shift * base / (1 + (shift - 1) * base)
    sigmas = torch.cat([shifted, torch.zeros(1)])
    timesteps = shifted * 1000
    return sigmas, timesteps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--dit", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    dit_script = load_script("dump_longcat_avatar_dit", os.path.join(here, "dump_longcat_avatar_dit.py"))
    dit_script.install_stubs()
    sys.path.insert(0, args.repo)
    import importlib as _importlib
    dit_module = _importlib.import_module("longcat_video.modules.avatar.longcat_video_dit_avatar")

    with open(os.path.join(args.dit, "meta.json"), encoding="utf-8") as handle:
        meta = json.load(handle)
    config = meta["config"]
    model = dit_module.LongCatVideoAvatarTransformer3DModel(**{
        **config, "patch_size": tuple(config["patch_size"]),
    }).eval()
    state = {k[len("dit."):]: read_tensor(args.dit, meta, k) for k in meta["tensors"] if k.startswith("dit.")}
    model.load_state_dict(state, strict=True)

    latent = read_tensor(args.dit, meta, "inputs.latent").reshape(1, config["in_channels"], meta["frames"], meta["h"], meta["w"])
    encoder = read_tensor(args.dit, meta, "inputs.encoder").reshape(1, 1, meta["text_len"], config["caption_channels"])
    audio = read_tensor(args.dit, meta, "inputs.audio").reshape(
        1, 1 + config["vae_scale"] * (meta["frames"] - 1), config["audio_window"], config["audio_block"], config["audio_channel"])
    # The negative prompt is a stand-in (the encoder scaled by 0.5): it is not a real negative prompt. It only makes the unconditional pass
    # differ from the conditional one, so the three-pass CFG is exercised.
    negative = encoder * 0.5
    # The reference zeroes the raw audio for the unconditional audio pass, and the DiT then projects those zeros.
    audio_zero = torch.zeros_like(audio)
    method = load_sigma_method(args.repo)
    steps, shift, text_scale, audio_scale = 4, 7.0, 4.0, 4.0

    sigmas, timesteps = sigmas_with_shift(method, steps, False, "avatar-v1.6", shift)
    frames_total = meta["frames"]

    def cfg(latents, timestep, prompt_pos, prompt_neg, audio_raw, audio_unc, **kwargs):
        """The reference's three passes and their combination (negated for the scheduler), as in generate_at2v and generate_avc."""
        pos = model(latents, timestep, prompt_pos, audio_embs=audio_raw, **kwargs)
        uncond_text = model(latents, timestep, prompt_neg, audio_embs=audio_raw, **kwargs)
        uncond = model(latents, timestep, prompt_neg, audio_embs=audio_unc, **kwargs)
        return -(uncond + text_scale * (pos - uncond_text) + audio_scale * (uncond_text - uncond))

    def ref_kwargs(cond_frames, num_ref):
        # The reference reads num_ref_latents and ref_img_index only when a reference frame exists (the rope depends on it).
        if num_ref:
            return {"num_cond_latents": cond_frames, "num_ref_latents": num_ref, "ref_img_index": 10}
        return {"num_cond_latents": cond_frames}

    def uncached_loop(latent_full, cond_frames, num_ref):
        """generate_* without use_kv_cache: every step runs all frames; the conditioning frames' timesteps are zero; only the noisy frames step."""
        latents = latent_full.clone()
        kwargs = ref_kwargs(cond_frames, num_ref)
        with torch.no_grad():
            for i in range(steps):
                t = float(timesteps[i])
                dt = float(sigmas[i + 1] - sigmas[i])
                time_t = torch.full((1, frames_total), t)
                time_t[:, :cond_frames] = 0.0
                noise = cfg(latents, time_t, encoder, negative, audio, audio_zero, **kwargs)
                latents[:, :, cond_frames:] = latents[:, :, cond_frames:] + dt * noise[:, :, cond_frames:]
        return latents

    # at2v: no conditioning frames, the full latent stepped.
    with torch.no_grad():
        latents = latent.clone()
        for i in range(steps):
            t = float(timesteps[i])
            dt = float(sigmas[i + 1] - sigmas[i])
            noise = cfg(latents, torch.tensor([t]), encoder, negative, audio, audio_zero)
            latents = latents + dt * noise
    at2v = latents

    # avc with the cache: two conditioning frames (the first two of the latent; the first is the reference slot), the cache built by a pass that
    # skips the text and audio steps (skip_crs_attn) and keeps each block's keys and values. Only the noisy frames are stepped.
    cond_frames = 2
    with torch.no_grad():
        cond = latent[:, :, :cond_frames].clone()
        _, kv = model(cond, torch.tensor([0.0]), encoder, audio_embs=audio, num_cond_latents=cond_frames, num_ref_latents=1,
                      ref_img_index=10, return_kv=True, skip_crs_attn=True)
        noisy = latent[:, :, cond_frames:].clone()
        kwargs = ref_kwargs(cond_frames, 1)
        for i in range(steps):
            t = float(timesteps[i])
            dt = float(sigmas[i + 1] - sigmas[i])
            noise = cfg(noisy, torch.tensor([t]), encoder, negative, audio, audio_zero, kv_cache_dict=kv, **kwargs)
            noisy = noisy + dt * noise
    avc = torch.cat([cond, noisy], dim=2)

    # avc without the cache: the same loop with full forwards and zero conditioning timesteps.
    avc_nocache = uncached_loop(latent, cond_frames, 1)

    # ai2v: one conditioning frame (the image), no reference, no cache.
    ai2v = uncached_loop(latent, 1, 0)

    os.makedirs(args.dit, exist_ok=True)
    outputs = (("pipeline.at2v.latents", at2v), ("pipeline.avc.latents", avc), ("pipeline.avc_nocache.latents", avc_nocache),
               ("pipeline.ai2v.latents", ai2v), ("pipeline.negative", negative))
    for name, value in outputs:
        array = np.ascontiguousarray(value.detach().cpu().numpy().astype(np.float32))
        array.tofile(os.path.join(args.dit, name + ".f32"))
        meta["tensors"][name] = list(array.shape)
    meta["pipeline"] = {"steps": steps, "shift": shift, "text_scale": text_scale, "audio_scale": audio_scale, "cond_frames": cond_frames}
    with open(os.path.join(args.dit, "meta.json"), "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    print(json.dumps({"at2v": list(at2v.shape), "avc": list(avc.shape), "sigmas": [round(float(s), 4) for s in sigmas]}))


if __name__ == "__main__":
    main()
