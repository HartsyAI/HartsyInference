"""Reference dump for LongCat-Video-Avatar's multi-speaker path (meituan-longcat/LongCat-Video @ 58eb336, modules/avatar/*,
audio_process/torch_utils.py, pipeline_longcat_video_avatar.py and run_demo_avatar_multi_audio_to_video.py).

Runs the pinned model on a tiny config with seeded weights, with the speaker masks given, and records per block the reference attention map,
the 1D RoPE positions of the queries and the audio keys and the audio cross-attention output, plus the velocity. Scenarios: ai2v with two
speakers, avc with a reference latent (uncached and through the KV cache), three tracks with a fourth mask (the silent background), and an
empty token mask. It also records the token masks of many source sizes and box layouts, the guided denoising loops, and the demo's audio
combination for both modes.

The empty-mask scenario divides by zero at the pin (the dumper checks that the pinned map has NaN), so its expected values are those of
upstream pull request 159 (the denominator clamped to at least one). The audio combination is that of upstream pull request 162, and the
dumper checks that the pin's conditioning tracks are the same as its own up to the padded length. Both files are read from --prs: pr159/
torch_utils.py at 5ef3065 and pr162/run_demo_avatar_multi_audio_to_video.py at 7327a66.

Run in the hartsy-ref:cpu image only (never the Windows host):

    python dump_longcat_avatar_multi.py --repo /src --prs /prs --out /out

What is the reference's own code: the DiT, its blocks and attention (get_attn_map_with_target included), SingleStreamAttention and the 1D
RoPE, _resize_and_centercrop_tensor and get_timesteps_sigmas (taken from the pipeline source by its AST, since the module needs diffusers),
and audio_prepare_multi (taken from the demo source the same way).

What is transcribed, because it is inline in a function that cannot be run here: the demo's mask construction (the default left and right
halves with 10 percent margins, the background and the extra regions; with the default boxes the demo halves its width variable, so extra
regions then fail to stack, which the dumper checks), the DiT's three lines that bring the masks to the token grid (the
model call checks them: the dumper compares its transcription with the masks the model hands to the attention), and the denoising loops
with the three-pass guidance.

Stubbed, so NOT used by the paths under test:
- diffusers and safetensors, as in dump_longcat_avatar_dit.py.
- flash_attn: torch SDPA with the same layouts and softmax scale.
- librosa, pyloudnorm and imageio: imported by torch_utils at module level; audio_prepare_multi gets a librosa.load that returns saved arrays.
- block_sparse_attention.bsa_interface.
- torch.compile is the identity, and torch.cuda.empty_cache and ipc_collect do nothing: calculate_x_ref_attn_map is compiled and calls them.
"""

import argparse
import ast
import importlib
import importlib.util
import json
import math
import os
import sys
import textwrap
import types

import numpy as np
import torch


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def install_stubs(repo):
    here = os.path.dirname(os.path.abspath(__file__))
    load_script("dump_longcat_avatar_dit", os.path.join(here, "dump_longcat_avatar_dit.py")).install_stubs()
    # The DiT dump stubs torch_utils; this one runs it.
    del sys.modules["longcat_video.audio_process.torch_utils"]
    for name in ("librosa", "pyloudnorm", "imageio"):
        sys.modules[name] = types.ModuleType(name)
    torch.compile = lambda function=None, **_: function
    torch.cuda.empty_cache = lambda: None
    torch.cuda.ipc_collect = lambda: None
    # The audio_process package imports its encoders; only torch_utils is wanted.
    package = types.ModuleType("longcat_video.audio_process")
    package.__path__ = [os.path.join(repo, "longcat_video", "audio_process")]
    sys.modules["longcat_video.audio_process"] = package


def function_from_source(path, name, namespace, class_name=None):
    with open(path, encoding="utf-8") as handle:
        source = handle.read()
    body = ast.parse(source).body
    if class_name:
        body = next(n for n in body if isinstance(n, ast.ClassDef) and n.name == class_name).body
    node = next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(textwrap.dedent(" " * node.col_offset + ast.get_source_segment(source, node)), namespace)
    return namespace[name]


class Dump:
    def __init__(self, out):
        self.out = out
        self.meta = {"tensors": {}}
        os.makedirs(out, exist_ok=True)

    def tensor(self, name, value):
        array = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
        array = np.ascontiguousarray(array.astype(np.float32))
        array.tofile(os.path.join(self.out, name + ".f32"))
        self.meta["tensors"][name] = list(array.shape)

    def save(self):
        with open(os.path.join(self.out, "meta.json"), "w", encoding="utf-8") as handle:
            json.dump(self.meta, handle, indent=2)


def source_masks(src_h, src_w, first, second, others):
    """The demo's mask construction (run_demo_avatar_multi_audio_to_video.py lines 383-409)."""
    background = torch.zeros([src_h, src_w])
    mask1 = torch.zeros([src_h, src_w])
    mask2 = torch.zeros([src_h, src_w])
    width = src_w
    if first is None and second is None:
        face_scale = 0.1
        left_y_min, left_y_max = int(src_h * face_scale), int(src_h * (1 - face_scale))
        right_y_min, right_y_max = left_y_min, left_y_max
        width = src_w // 2  # the demo halves its width variable here, and sizes the extra masks with it below
        left_x_min, left_x_max = int(width * face_scale), int(width * (1 - face_scale))
        right_x_min, right_x_max = int(width * face_scale + width), int(width * (1 - face_scale) + width)
    else:
        left_y_min, left_x_min, left_y_max, left_x_max = first
        right_y_min, right_x_min, right_y_max, right_x_max = second
    mask1[left_y_min:left_y_max, left_x_min:left_x_max] = 1
    mask2[right_y_min:right_y_max, right_x_min:right_x_max] = 1
    background += mask1
    background += mask2
    background = torch.where(background > 0, torch.tensor(0), torch.tensor(1))
    total = [mask1, mask2, background]
    for box in others or []:
        other = torch.zeros([src_h, width])
        other[box[0]:box[2], box[1]:box[3]] = 1
        total.append(other)
    return torch.stack(total, dim=0)


def token_masks(resized, n_h, n_w):
    """The DiT's lines 540-545 (longcat_video_dit_avatar.py)."""
    masks = resized.unsqueeze(0).to(torch.float32)
    masks = torch.nn.functional.interpolate(masks, size=(n_h, n_w), mode="nearest").squeeze(0)
    masks = masks > 0
    return masks.view(masks.shape[0], -1).float()


def mask_cases(rng):
    """Source sizes, target sizes and box layouts: fixed edge cases, then random ones."""
    cases = [
        dict(src=(75, 121), target=(64, 96), first=None, second=None, others=[]),
        dict(src=(121, 75), target=(64, 96), first=None, second=None, others=[]),
        dict(src=(480, 832), target=(480, 832), first=None, second=None, others=[]),
        dict(src=(1081, 1923), target=(480, 832), first=None, second=None, others=[]),
        dict(src=(799, 1279), target=(768, 1280), first=[40, 30, 760, 560], second=[60, 720, 770, 1250], others=[[0, 600, 799, 700]]),
        dict(src=(900, 900), target=(480, 832), first=[100, 80, 800, 440], second=[50, 450, 820, 870], others=[]),
        dict(src=(853, 1325), target=(480, 832), first=[100, 80, 800, 640], second=[50, 720, 820, 1300], others=[]),
        dict(src=(37, 53), target=(480, 832), first=[3, 5, 17, 19], second=[9, 31, 36, 52], others=[[0, 0, 5, 5], [30, 20, 37, 30]]),
        # Boxes past the image edge are clipped by the slicing.
        dict(src=(200, 300), target=(128, 192), first=[10, 10, 5000, 140], second=[0, 150, 190, 5000], others=[]),
    ]
    for _ in range(60):
        src = (int(rng.integers(17, 1500)), int(rng.integers(17, 1500)))
        target = (16 * int(rng.integers(2, 60)), 16 * int(rng.integers(2, 60)))

        def box():
            y0, x0 = int(rng.integers(0, src[0] - 1)), int(rng.integers(0, src[1] - 1))
            return [y0, x0, int(rng.integers(y0 + 1, src[0] + 1)), int(rng.integers(x0 + 1, src[1] + 1))]

        boxed = bool(rng.integers(0, 2))
        cases.append(dict(src=src, target=target, first=box() if boxed else None, second=box() if boxed else None,
                          others=[box() for _ in range(int(rng.integers(0, 3)))] if boxed else []))
    return cases


def dit_config():
    return dict(
        in_channels=4, out_channels=4, hidden_size=64, depth=2, num_heads=4, caption_channels=32, mlp_ratio=4,
        adaln_tembed_dim=32, frequency_embedding_size=32, patch_size=(1, 2, 2), enable_flashattn2=True,
        text_tokens_zero_pad=True, audio_window=5, audio_block=3, audio_channel=6, intermediate_dim=16, output_dim=16,
        context_tokens=4, vae_scale=4, class_range=24, class_interval=4, cp_split_hw=[1, 1],
    )


class Recorder:
    """Wraps the reference's attention map and 1D RoPE and records what each block saw."""

    def __init__(self, model, attention):
        self.maps, self.masks, self.positions, self.audio = [], [], [], []
        original = attention.get_attn_map_with_target

        def wrapped(noise_q, key, shape, ref_target_masks=None, **kwargs):
            result = original(noise_q, key, shape, ref_target_masks=ref_target_masks, **kwargs)
            self.maps.append(result.detach().clone())
            self.masks.append(ref_target_masks.detach().clone())
            return result

        attention.get_attn_map_with_target = wrapped
        for block in model.blocks:
            block.audio_cross_attn.rope_1d.register_forward_pre_hook(
                lambda _m, args: self.positions.append(args[1].detach().clone()))
            block.audio_cross_attn.register_forward_hook(lambda _m, _i, out: self.audio.append(out[1].detach().clone()))

    def clear(self):
        for values in (self.maps, self.masks, self.positions, self.audio):
            values.clear()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--prs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()

    torch.set_num_threads(2)
    sys.path.insert(0, args.repo)
    install_stubs(args.repo)
    attention = importlib.import_module("longcat_video.modules.avatar.attention")
    dit_module = importlib.import_module("longcat_video.modules.avatar.longcat_video_dit_avatar")
    pipeline_source = os.path.join(args.repo, "longcat_video", "pipeline_longcat_video_avatar.py")
    resize = function_from_source(pipeline_source, "_resize_and_centercrop_tensor", {"torch": torch, "F": torch.nn.functional},
                                  "LongCatVideoAvatarPipeline")
    dump = Dump(args.out)
    rng = np.random.default_rng(args.seed)

    # ---- token masks ----
    dump.meta["mask_cases"] = []
    for index, case in enumerate(mask_cases(rng)):
        masks = source_masks(*case["src"], case["first"], case["second"], case["others"])
        resized = resize(None, masks.float(), case["target"][0], case["target"][1], "crop")
        grid = (case["target"][0] // 16, case["target"][1] // 16)
        dump.tensor(f"mask.{index}", token_masks(resized, *grid))
        dump.meta["mask_cases"].append({**case, "src": list(case["src"]), "target": list(case["target"]), "grid": list(grid)})

    # Default boxes with a silent person cannot be built at the pin, so the engine refuses the combination.
    try:
        source_masks(100, 200, None, None, [[0, 0, 10, 10]])
    except RuntimeError as error:
        print(f"default boxes with a silent person fail in the demo's mask code: {str(error)[:80]}")
    else:
        raise AssertionError("the demo's default boxes with a silent person were expected to fail")

    # ---- the DiT ----
    config = dit_config()
    torch.manual_seed(args.seed)
    model = dit_module.LongCatVideoAvatarTransformer3DModel(**config).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("q_norm.weight") or name.endswith("k_norm.weight"):
                # Large norm weights sharpen the attention, so the map separates the regions instead of sitting near uniform.
                p.copy_(3.0 * (1.0 + 0.2 * torch.randn_like(p)))
            else:
                p.copy_(torch.randn_like(p) * 0.05)
    recorder = Recorder(model, attention)

    frames, height, width, text_len = 4, 8, 12, 6
    n_h, n_w = height // 2, width // 2
    target = (height * 8, width * 8)
    latent = torch.randn(1, config["in_channels"], frames, height, width)
    encoder = torch.randn(1, 1, text_len, config["caption_channels"])
    audio_len = 1 + config["vae_scale"] * (frames - 1)
    audio = torch.randn(3, audio_len, config["audio_window"], config["audio_block"], config["audio_channel"])
    audio[2] = 0.0  # the silent background track

    dump.meta.update({"config": config, "frames": frames, "h": height, "w": width, "text_len": text_len})
    for key, value in model.state_dict().items():
        dump.tensor("dit." + key, value)
    dump.tensor("inputs.latent", latent.reshape(-1))
    dump.tensor("inputs.encoder", encoder.reshape(text_len, -1))
    dump.tensor("inputs.audio", audio.reshape(3, audio_len, -1))

    layouts = {
        "two": dict(src=(75, 121), first=[7, 3, 68, 55], second=[11, 63, 71, 117], others=[], tracks=2),
        "defaults": dict(src=(75, 121), first=None, second=None, others=[], tracks=2),
        "three": dict(src=(75, 121), first=[7, 3, 68, 40], second=[11, 81, 71, 117], others=[[5, 44, 70, 77]], tracks=3),
        # The second speaker's box lies in the columns the centre crop removes, so its token mask is empty.
        "empty": dict(src=(64, 384), first=[4, 150, 60, 230], second=[0, 0, 64, 40], others=[], tracks=2),
    }
    layout_masks = {}
    dump.meta["layouts"] = {}
    for name, layout in layouts.items():
        masks = source_masks(*layout["src"], layout["first"], layout["second"], layout["others"])
        layout_masks[name] = resize(None, masks.float(), target[0], target[1], "crop")
        dump.tensor(f"layout.{name}.tokens", token_masks(layout_masks[name], n_h, n_w))
        dump.meta["layouts"][name] = {**layout, "src": list(layout["src"])}

    avc = dict(num_cond_latents=2, num_ref_latents=1, ref_img_index=10)
    scenarios = {
        "ai2v": ("two", dict(num_cond_latents=1)),
        "avc": ("two", avc),
        "avc_cache": ("two", avc),
        "ai2v_defaults": ("defaults", dict(num_cond_latents=1)),
        "three": ("three", dict(num_cond_latents=1)),
        "three_cache": ("three", avc),
        "empty": ("empty", dict(num_cond_latents=1)),
    }
    dump.meta["scenarios"] = {}
    timestep = torch.tensor([500.0])

    def forward(name, layout_name, kwargs):
        tracks = layouts[layout_name]["tracks"]
        masks = layout_masks[layout_name]
        if name.endswith("_cache"):
            cond = kwargs["num_cond_latents"]
            _, cache = model(latent[:, :, :cond], torch.tensor([0.0]), encoder, audio_embs=audio[:tracks], return_kv=True,
                             skip_crs_attn=True, **kwargs)
            recorder.clear()
            return model(latent[:, :, cond:], timestep, encoder, audio_embs=audio[:tracks], kv_cache_dict=cache,
                         ref_target_masks=masks, **kwargs)
        recorder.clear()
        return model(latent, timestep, encoder, audio_embs=audio[:tracks], ref_target_masks=masks, **kwargs)

    pr159 = os.path.join(args.prs, "pr159", "torch_utils.py")
    pinned_map = attention.get_attn_map_with_target
    for name, (layout_name, kwargs) in scenarios.items():
        if name == "empty":
            with torch.no_grad():
                pinned = forward(name, layout_name, kwargs)
            assert torch.isnan(recorder.maps[0]).any() and torch.isnan(pinned).any(), "the pin should divide by zero here"
            spec = importlib.util.spec_from_file_location("longcat_video.audio_process.torch_utils_pr159", pr159)
            fixed = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(fixed)
            torch_utils = sys.modules["longcat_video.audio_process.torch_utils"]
            torch_utils.calculate_x_ref_attn_map = fixed.calculate_x_ref_attn_map
        with torch.no_grad():
            velocity = forward(name, layout_name, kwargs)
        assert torch.isfinite(velocity).all()
        tracks = layouts[layout_name]["tracks"]
        dump.tensor(f"{name}.velocity", velocity.reshape(-1))
        expected_tokens = token_masks(layout_masks[layout_name], n_h, n_w)
        gaps = []
        for block in range(config["depth"]):
            assert torch.equal(recorder.masks[block].float(), expected_tokens), "the transcribed token masks differ from the model's"
            attn_map = recorder.maps[block]
            dump.tensor(f"{name}.block{block}.ref_attn_map", attn_map)
            dump.tensor(f"{name}.block{block}.query_positions", recorder.positions[2 * block])
            dump.tensor(f"{name}.block{block}.key_positions", recorder.positions[2 * block + 1])
            dump.tensor(f"{name}.block{block}.audio_cross_attn", recorder.audio[block][0])
            top = torch.topk(attn_map, 2, dim=0).values
            gaps.append(float((top[0] - top[1]).abs().min()))
        classes = sorted({int(c) for block in recorder.maps[: config["depth"]] for c in block.argmax(0).clamp(max=2).tolist()})
        print(f"{name}: classes {classes}, smallest top-two gap of the map {min(gaps):.3e}, map range "
              f"{float(recorder.maps[0].min()):.3e}..{float(recorder.maps[0].max()):.3e}")
        dump.meta["scenarios"][name] = {"layout": layout_name, "tracks": tracks, "smallest_gap": min(gaps), "classes": classes,
                                        "num_cond": kwargs["num_cond_latents"], "num_ref": kwargs.get("num_ref_latents", 0),
                                        "ref_index": kwargs.get("ref_img_index", 10), "cached": name.endswith("_cache")}
    attention.get_attn_map_with_target = pinned_map
    seen = {c for s in dump.meta["scenarios"].values() for c in s["classes"]}
    assert seen == {0, 1, 2}, f"all three classes must occur, saw {sorted(seen)}"
    for required in ("ai2v", "avc", "three"):
        assert dump.meta["scenarios"][required]["classes"] == [0, 1, 2], f"{required} must use all three classes"

    # ---- the guided loops (multi-speaker three-pass CFG) ----
    sigma_method = function_from_source(pipeline_source, "get_timesteps_sigmas", {"torch": torch, "np": np},
                                        "LongCatVideoAvatarPipeline")
    stand_in = types.SimpleNamespace(model_type="avatar-v1.5", num_timesteps=1000, num_distill_sample_steps=8)
    steps, shift, text_scale, audio_scale = 4, 7.0, 4.0, 4.0
    base = sigma_method(stand_in, steps, use_distill=False).float()
    shifted = shift * base / (1 + (shift - 1) * base)
    sigmas, timesteps = torch.cat([shifted, torch.zeros(1)]), shifted * 1000
    negative = encoder * 0.5  # a stand-in that makes the unconditional passes differ
    audio_zero = torch.zeros_like(audio)
    dump.tensor("pipeline.negative", negative.reshape(text_len, -1))
    masks = layout_masks["two"]

    def guided(latents, time_t, **kwargs):
        def call(prompt, speech):
            return model(latents, time_t, prompt, audio_embs=speech, ref_target_masks=masks, **kwargs)

        uncond_text = call(negative, audio[:2])
        positive = call(encoder, audio[:2])
        uncond = call(negative, audio_zero[:2])
        return -(uncond + text_scale * (positive - uncond_text) + audio_scale * (uncond_text - uncond))

    with torch.no_grad():
        ai2v = latent.clone()
        for i in range(steps):
            time_t = torch.full((1, frames), float(timesteps[i]))
            time_t[:, :1] = 0.0
            noise = guided(ai2v, time_t, num_cond_latents=1)
            ai2v[:, :, 1:] = ai2v[:, :, 1:] + float(sigmas[i + 1] - sigmas[i]) * noise[:, :, 1:]
        cond = latent[:, :, :2].clone()
        _, cache = model(cond, torch.tensor([0.0]), encoder, audio_embs=audio[:2], return_kv=True, skip_crs_attn=True, **avc)
        noisy = latent[:, :, 2:].clone()
        for i in range(steps):
            noise = guided(noisy, torch.tensor([float(timesteps[i])]), kv_cache_dict=cache, **avc)
            noisy = noisy + float(sigmas[i + 1] - sigmas[i]) * noise
    dump.tensor("pipeline.ai2v.latents", ai2v)
    dump.tensor("pipeline.avc.latents", torch.cat([cond, noisy], dim=2))
    dump.meta["pipeline"] = {"steps": steps, "shift": shift, "text_scale": text_scale, "audio_scale": audio_scale, "cond_frames": 2}

    # ---- the audio combination ----
    saved = {}
    librosa = types.SimpleNamespace(load=lambda path, sr: (saved[path], sr))
    namespace = lambda: {"np": np, "math": math, "librosa": librosa}
    demo = "run_demo_avatar_multi_audio_to_video.py"
    pinned_prepare = function_from_source(os.path.join(args.repo, demo), "audio_prepare_multi", namespace())
    fixed_prepare = function_from_source(os.path.join(args.prs, "pr162", demo), "audio_prepare_multi", namespace())
    rate = 16000

    def noise(count):
        return (rng.standard_normal(count) * 0.1).astype(np.float32)

    # (vocals, raw) lengths per speaker; None is an absent speaker. The vocals are what the separator returns for the clip plus one
    # second, so they are never longer than the recording.
    audio_cases = [
        ("para", (3000, 3000), (2000, 2600), 0.5),
        ("add", (3000, 3000), (2000, 2600), 0.5),
        ("para", (4000, 9000), (4000, 7000), 0.25),
        ("add", (4000, 9000), (4000, 7000), 0.25),
        ("add", (4000, 9000), None, 1.0),
        ("add", None, (2500, 2500), 1.0),
        ("para", (4000, 6000), None, 0.5),
        ("para", None, (2500, 2500), 0.5),
        ("add", (8000, 8000), (8000, 8000), 0.1),
    ]
    dump.meta["audio_cases"] = []
    for index, (mode, first, second, duration) in enumerate(audio_cases):
        values = []
        for speaker, lengths in enumerate((first, second)):
            if lengths is None:
                values.append((None, None))
                continue
            raw = noise(lengths[1])
            values.append((raw[: lengths[0]] * 0.7, raw))
            saved[f"raw{speaker}"] = raw
            dump.tensor(f"audio.{index}.vocals{speaker}", values[speaker][0])
            dump.tensor(f"audio.{index}.raw{speaker}", raw)
        call = (values[0][0], values[1][0], duration, "raw0", "raw1")
        left, right, mix = fixed_prepare(*call, sample_rate=rate, audio_type=mode)
        pin_left, pin_right, pin_mix = pinned_prepare(*call, sample_rate=rate, audio_type=mode)
        for a, b in ((left, pin_left), (right, pin_right)):
            shorter = min(len(a), len(b))
            assert abs(len(a) - len(b)) <= 1 and np.array_equal(a[:shorter], b[:shorter]), "the conditioning should match the pin"
        shorter = min(len(mix), len(pin_mix))
        same_mix = len(mix) == len(pin_mix) and np.array_equal(mix[:shorter], pin_mix[:shorter])
        print(f"audio {index} {mode}: {len(left)} samples (pin {len(pin_left)}), soundtrack {'as the pin' if same_mix else 'differs from the pin'}")
        dump.tensor(f"audio.{index}.track0", left)
        dump.tensor(f"audio.{index}.track1", right)
        dump.tensor(f"audio.{index}.mix", mix)
        dump.meta["audio_cases"].append({"mode": mode, "present": [first is not None, second is not None], "duration": duration,
                                         "pin_soundtrack_matches": bool(same_mix)})

    dump.save()
    print(json.dumps({"tensors": len(dump.meta["tensors"]), "mask_cases": len(dump.meta["mask_cases"]),
                      "scenarios": list(dump.meta["scenarios"])}))


if __name__ == "__main__":
    main()
