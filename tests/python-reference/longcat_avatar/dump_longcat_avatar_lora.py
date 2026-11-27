"""Reference dump for the LongCat-Video-Avatar release LoRAs (modules/lora_utils.py and the LoRA methods of
modules/avatar/longcat_video_dit_avatar.py @ 58eb336) on the tiny avatar DiT of dump_longcat_avatar_dit.py, CPU F32.

Writes two LoRA files from the reference's own LoRANetwork (the seven linears of every block, fused projections split as the release splits
them, every weight random, an alpha_scale per module that is not alpha / rank) and loads them back through the model's load_lora. Then runs one
ai2v call (one conditioning latent) in each state the avatar demo (run_demo_avatar_single_audio_to_video.py) can put the model in:

- base: no LoRA.
- distill15: the 1.5 path, merge_lora of the first file at the demo's --lora_multiplier 0.8.
- mix_a1.0_n4_s0 and _s4: the 1.6 path, load_lora of both at multiplier 1, enable_lora_mix(alpha 1.0, 4 steps), set_lora_mix_step(0) and (4).
- mix_a0.5_n3_s2 and _s3: the same with the demo's default mix (alpha 0.5, 3 steps), the last mixed step and the first after it.

Run in the hartsy-ref:cpu image only (never the Windows host), with the pinned source (longcat_video/) in a folder:

    python dump_longcat_avatar_lora.py --repo /src --out /out/longcat_avatar_lora_reference_tensors

Stubs as in dump_longcat_avatar_dit.py, except safetensors, which is the real package: the LoRA files are written and read with it.
"""

import argparse
import copy
import importlib
import json
import os
import sys

import torch

from dump_longcat_avatar_dit import install_stubs, write_tensor

RANK = 4
DEMO_NETWORK_ALPHA = 64
# Fused projections and the number of up blocks the release splits each into.
MODULES = (("attn.qkv", 3), ("attn.proj", 1), ("cross_attn.q_linear", 1), ("cross_attn.kv_linear", 2), ("ffn.w1", 1), ("ffn.w2", 1),
           ("ffn.w3", 1))


def lora_key(module):
    return "lora___lorahyphen___" + module.replace(".", "___lorahyphen___")


def write_lora(lora_utils, model, path, alpha_scales, save_file):
    """A LoRA file for every listed module of every block: the reference's network, filled with seeded random weights."""
    wanted = {}
    for block in range(len(model.blocks)):
        for name, splits in MODULES:
            key = lora_key(f"blocks.{block}.{name}")
            wanted[key + ".lora_down.weight"] = None
            for index in range(splits if splits > 1 else 0):
                wanted[f"{key}.lora_up.blocks.{index}.weight"] = None
    network = lora_utils.create_lora_network(model, wanted, multiplier=1.0, network_dim=RANK, network_alpha=DEMO_NETWORK_ALPHA)
    assert len(network.loras) == len(model.blocks) * len(MODULES), f"{len(network.loras)} modules"
    with torch.no_grad():
        for index, lora in enumerate(sorted(network.loras, key=lambda item: item.lora_name)):
            for parameter in lora.parameters():
                parameter.copy_(torch.randn_like(parameter) * 0.15)
            lora.alpha_scale.fill_(alpha_scales[index % len(alpha_scales)])
    save_file({key: value.contiguous() for key, value in network.state_dict().items()}, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()

    sys.path.insert(0, args.repo)
    install_stubs()
    for name in ("safetensors", "safetensors.torch"):
        del sys.modules[name]
    import safetensors
    from safetensors.torch import save_file
    dit_module = importlib.import_module("longcat_video.modules.avatar.longcat_video_dit_avatar")
    lora_utils = importlib.import_module("longcat_video.modules.lora_utils")

    config = dict(
        in_channels=4, out_channels=4, hidden_size=64, depth=2, num_heads=4, caption_channels=32, mlp_ratio=4,
        adaln_tembed_dim=32, frequency_embedding_size=32, patch_size=(1, 2, 2), enable_flashattn2=True,
        text_tokens_zero_pad=True, audio_window=5, audio_block=3, audio_channel=6, intermediate_dim=16, output_dim=16,
        context_tokens=4, vae_scale=4, class_range=24, class_interval=4, cp_split_hw=[1, 1],
    )
    torch.manual_seed(args.seed)
    model = dit_module.LongCatVideoAvatarTransformer3DModel(**config).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn_like(p) * 0.05)

    frames, height, width, text_len = 4, 4, 4, 6
    latent = torch.randn(1, config["in_channels"], frames, height, width)
    encoder = torch.randn(1, 1, text_len, config["caption_channels"])
    audio_len = 1 + config["vae_scale"] * (frames - 1)
    audio = torch.randn(1, audio_len, config["audio_window"], config["audio_block"], config["audio_channel"])
    timestep = torch.tensor([500.0])

    os.makedirs(args.out, exist_ok=True)
    meta = {"config": config, "frames": frames, "h": height, "w": width, "text_len": text_len, "tensors": {}, "cases": {},
            "torch": torch.__version__, "safetensors": safetensors.__version__}
    for key, value in model.state_dict().items():
        write_tensor(args.out, meta, "dit." + key, value)
    write_tensor(args.out, meta, "inputs.latent", latent.reshape(-1))
    write_tensor(args.out, meta, "inputs.encoder", encoder.reshape(text_len, -1))
    write_tensor(args.out, meta, "inputs.audio", audio.reshape(audio_len, -1))

    first = os.path.join(args.out, "loraA.safetensors")
    second = os.path.join(args.out, "loraB.safetensors")
    write_lora(lora_utils, model, first, (0.5, 0.25, 1.5), save_file)
    write_lora(lora_utils, model, second, (0.75, 2.0, 0.125), save_file)

    def run(name, subject, **case):
        with torch.no_grad():
            velocity = subject(latent, timestep, encoder, audio_embs=audio, num_cond_latents=1)
        write_tensor(args.out, meta, f"velocity.{name}", velocity.reshape(-1))
        meta["cases"][name] = case
        return velocity

    base = run("base", model, path="base")

    def moved(velocity):
        return float((velocity - base).pow(2).mean().sqrt() / base.pow(2).mean().sqrt())

    merged = copy.deepcopy(model)
    count = merged.merge_lora(first, lora_key="dmd", multiplier=0.8, lora_network_dim=RANK, lora_network_alpha=DEMO_NETWORK_ALPHA)
    assert count == len(model.blocks) * len(MODULES), f"merged {count}"
    moves = {"distill15": moved(run("distill15", merged, path="merge", multiplier=0.8))}

    mixed = copy.deepcopy(model)
    for path, key in ((first, "dmd"), (second, "opsd")):
        mixed.load_lora(path, key, multiplier=1.0, lora_network_dim=RANK, lora_network_alpha=DEMO_NETWORK_ALPHA)
    for alpha, mix_steps, steps in ((1.0, 4, (0, 4)), (0.5, 3, (2, 3))):
        mixed.enable_lora_mix("dmd", "opsd", alpha, mix_steps)
        for step in steps:
            mixed.set_lora_mix_step(step)
            name = f"mix_a{alpha}_n{mix_steps}_s{step}"
            moves[name] = moved(run(name, mixed, path="mix", alpha=alpha, mix_steps=mix_steps, step=step))
    assert min(moves.values()) > 1e-2, f"a LoRA case barely moves the output: {moves}"

    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    print(json.dumps({"torch": torch.__version__, "tensors": len(meta["tensors"]), "moved_from_base": moves}))


if __name__ == "__main__":
    main()
