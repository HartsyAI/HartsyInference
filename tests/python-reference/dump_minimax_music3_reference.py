"""Dump MiniMax Music 3 reference tensors for the C# parity tests.

Run under a venv with diffusers @ dafe3733fcfdbf3c48915fe77be3aef65b5d6a2d (the PR #14456 commit):

    python dump_minimax_music3_reference.py --model <checkpoint-dir> --stage components
    python dump_minimax_music3_reference.py --model <checkpoint-dir> --stage flow
    python dump_minimax_music3_reference.py --model <checkpoint-dir> --stage ar

`components` and `flow` need ~11 GB of RAM (transformer + vocoder + condition encoder in F32);
`ar` needs ~35 GB (the 8B language model in F32) and runs in its own process for that reason.

Writes to tests/python-reference/minimax_music3_reference/:

  meta.json                      # shapes, constants and the checkpoint file sizes
  cond_in.bin                    # F32 [1, frames, 32768]   synthetic frame hiddens
  cond_out.bin                   # F32 [1, latents, 2048]   condition encoder output
  dit_latents.bin                # F32 [1, 128, latents]
  dit_block0_in.bin              # F32 [1, latents+1, 2048]
  dit_block0_out.bin             # F32 [1, latents+1, 2048]
  dit_out_t{i}_{cond,uncond}.bin # F32 [1, 128, latents]    velocity at each probed timestep
  vocoder_out.bin                # F32 [1, 2, samples]
  flow_noise_{k}.bin             # F32 [1, 128, L_k]        every noise draw, in order
  flow_chunk_{k}.bin             # F32 [1, 128, L_k]        per-window denoised latents
  flow_audio.bin                 # F32 [1, 2, samples]      stitched waveform
  ar_frame_hiddens.bin           # F32 [1, frames, 32768]
  ar_last_hidden_{i}.bin         # F32 [2, 4096]            per-frame LM hidden (cond, uncond)
  ar_codes.bin                   # I32 [frames+1, 8]        sampled frame codes
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

OUT_DIR = Path(__file__).resolve().parent / "minimax_music3_reference"

COMPONENT_FRAMES = 40
FLOW_FRAMES = 300
FLOW_STEPS = 4
AR_FRAMES = 8
VOCODER_SHORT_LATENTS = 32
AR_CAPTION = "A warm acoustic pop song with intimate female vocals, fingerpicked guitar and soft piano."
AR_LYRICS = "[Verse]\nMorning light filtering through the pine\n[Chorus]\nSoftly the world begins to breathe"


def save(out: Path, name: str, tensor: torch.Tensor) -> list[int]:
    array = tensor.detach().to(torch.float32).cpu().numpy()
    array.tofile(out / f"{name}.bin")
    return list(array.shape)


def dump_components(model: Path, out: Path, meta: dict) -> None:
    from diffusers import MiniMaxMusic3ConditionEncoder, MiniMaxMusic3Transformer1DModel, MiniMaxMusic3Vocoder

    condition_encoder = MiniMaxMusic3ConditionEncoder.from_pretrained(
        model, subfolder="condition_encoder", dtype=torch.float32
    ).eval()
    transformer = MiniMaxMusic3Transformer1DModel.from_pretrained(
        model, subfolder="transformer", dtype=torch.float32
    ).eval()
    vocoder = MiniMaxMusic3Vocoder.from_pretrained(model, subfolder="vocoder", dtype=torch.float32).eval()

    alphas = [p for name, p in vocoder.named_parameters() if name.endswith("alpha")]
    stacked = torch.cat([a.flatten() for a in alphas])
    meta["vocoder_snake_alpha"] = {
        "count": int(stacked.numel()),
        "min": float(stacked.min()),
        "max": float(stacked.max()),
        "mean": float(stacked.mean()),
    }

    generator = torch.Generator().manual_seed(0)
    frame_hiddens = torch.randn(1, COMPONENT_FRAMES, 32768, generator=generator)
    meta["cond_in"] = save(out, "cond_in", frame_hiddens)
    with torch.no_grad():
        condition = condition_encoder(frame_hiddens)
    meta["cond_out"] = save(out, "cond_out", condition)

    latent_length = condition.shape[1]
    latents = torch.randn(1, 128, latent_length, generator=generator)
    meta["dit_latents"] = save(out, "dit_latents", latents)

    captured: dict[str, torch.Tensor] = {}

    def hook(_module, args, output):
        captured["in"] = args[0].detach().clone()
        captured["out"] = output.detach().clone()

    handle = transformer.transformer_blocks[0].register_forward_hook(hook)
    timesteps = [0.0, 0.5]
    meta["dit_timesteps"] = timesteps
    with torch.no_grad():
        for index, value in enumerate(timesteps):
            timestep = torch.tensor([value], dtype=torch.float32)
            conditional = transformer(latents, timestep, condition, return_dict=False)[0]
            meta[f"dit_out_t{index}_cond"] = save(out, f"dit_out_t{index}_cond", conditional)
            if index == 0:
                meta["dit_block0_in"] = save(out, "dit_block0_in", captured["in"])
                meta["dit_block0_out"] = save(out, "dit_block0_out", captured["out"])
            unconditional = transformer(latents, timestep, torch.zeros_like(condition), return_dict=False)[0]
            meta[f"dit_out_t{index}_uncond"] = save(out, f"dit_out_t{index}_uncond", unconditional)
    handle.remove()

    with torch.no_grad():
        waveform = vocoder(latents)
    meta["vocoder_out"] = save(out, "vocoder_out", waveform)

    # A short case as well: the full one takes over 20 minutes to decode on a CPU backend.
    short_latents = latents[..., :VOCODER_SHORT_LATENTS].contiguous()
    meta["vocoder_short_in"] = save(out, "vocoder_short_in", short_latents)
    with torch.no_grad():
        meta["vocoder_short_out"] = save(out, "vocoder_short_out", vocoder(short_latents))


def dump_flow(model: Path, out: Path, meta: dict) -> None:
    import diffusers.modular_pipelines.minimax_music3.denoise as denoise_module
    from diffusers.modular_pipelines import SequentialPipelineBlocks
    from diffusers.modular_pipelines.minimax_music3.decoders import MiniMaxMusic3VocoderDecodeStep
    from diffusers.modular_pipelines.minimax_music3.before_denoise import MiniMaxMusic3PrepareChunksStep
    from diffusers.modular_pipelines.minimax_music3.denoise import MiniMaxMusic3ChunkDenoiseStep

    draws: list[torch.Tensor] = []
    original = denoise_module.randn_tensor

    def recording_randn(*args, **kwargs):
        drawn = original(*args, **kwargs)
        draws.append(drawn.detach().clone())
        return drawn

    denoise_module.randn_tensor = recording_randn

    class FlowOnlyBlocks(SequentialPipelineBlocks):
        model_name = "minimax-music3"
        block_classes = [MiniMaxMusic3PrepareChunksStep, MiniMaxMusic3ChunkDenoiseStep, MiniMaxMusic3VocoderDecodeStep]
        block_names = ["prepare_chunks", "denoise", "decode"]

    # update_components, not load_components: the checkpoint's modular_model_index.json names the Hub repo, so
    # load_components re-downloads all 28 GB even when --model already points at a complete local copy.
    from diffusers import (
        FlowMatchEulerDiscreteScheduler,
        MiniMaxMusic3ConditionEncoder,
        MiniMaxMusic3Transformer1DModel,
        MiniMaxMusic3Vocoder,
    )

    pipe = FlowOnlyBlocks().init_pipeline(str(model))
    pipe.update_components(
        condition_encoder=MiniMaxMusic3ConditionEncoder.from_pretrained(
            model, subfolder="condition_encoder", dtype=torch.float32
        ),
        transformer=MiniMaxMusic3Transformer1DModel.from_pretrained(model, subfolder="transformer", dtype=torch.float32),
        scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(model, subfolder="scheduler"),
        vocoder=MiniMaxMusic3Vocoder.from_pretrained(model, subfolder="vocoder", dtype=torch.float32),
    )

    generator = torch.Generator().manual_seed(0)
    frame_hiddens = torch.randn(1, FLOW_FRAMES, 32768, generator=generator)
    state = pipe(
        frame_hiddens=frame_hiddens,
        num_inference_steps=FLOW_STEPS,
        generator=generator,
        output_type="pt",
    )

    denoise_module.randn_tensor = original

    meta["flow_frames"] = FLOW_FRAMES
    meta["flow_steps"] = FLOW_STEPS
    meta["flow_frame_hiddens"] = save(out, "flow_frame_hiddens", frame_hiddens)
    meta["flow_chunk_starts"] = list(state.get("chunk_starts"))
    chunks = state.get("latent_chunks")
    meta["flow_chunks"] = [save(out, f"flow_chunk_{k}", chunk) for k, chunk in enumerate(chunks)]
    meta["flow_noise"] = [save(out, f"flow_noise_{k}", drawn) for k, drawn in enumerate(draws)]
    meta["flow_audio"] = save(out, "flow_audio", state.get("audios"))


def dump_ar(model: Path, out: Path, meta: dict) -> None:
    """Runs the reference semantic-generation block itself; hooks and a `_sample_top_k` recorder capture the
    per-frame internals so nothing about the loop is re-transcribed here."""
    import diffusers.modular_pipelines.minimax_music3.encoders as encoders
    from diffusers.modular_pipelines import SequentialPipelineBlocks
    from diffusers.modular_pipelines.minimax_music3.encoders import (
        MiniMaxMusic3SemanticGenerationStep,
        MiniMaxMusic3TextEncoderStep,
    )

    samples: list[int] = []
    original_sample = encoders._sample_top_k

    def recording_sample(logits, generator):
        drawn = original_sample(logits, generator)
        samples.append(int(drawn.reshape(-1)[0].item()))
        return drawn

    encoders._sample_top_k = recording_sample

    class ArOnlyBlocks(SequentialPipelineBlocks):
        model_name = "minimax-music3"
        block_classes = [MiniMaxMusic3TextEncoderStep, MiniMaxMusic3SemanticGenerationStep]
        block_names = ["text_encoder", "semantic_generator"]

    # See dump_flow: load_components would re-download the whole repo from the Hub.
    from transformers import Qwen2Tokenizer, Qwen3ForCausalLM

    from diffusers import MiniMaxMusic3RVQDepthDecoder

    pipe = ArOnlyBlocks().init_pipeline(str(model))
    pipe.update_components(
        tokenizer=Qwen2Tokenizer.from_pretrained(str(model / "tokenizer")),
        language_model=Qwen3ForCausalLM.from_pretrained(model / "language_model", dtype=torch.float32),
        rvq_depth_decoder=MiniMaxMusic3RVQDepthDecoder.from_pretrained(
            model, subfolder="rvq_depth_decoder", dtype=torch.float32
        ),
    )

    hiddens: list[torch.Tensor] = []

    def lm_hook(_module, _args, output):
        hiddens.append(output.last_hidden_state[:, -1].detach().clone())

    handle = pipe.language_model.model.register_forward_hook(lm_hook)
    state = pipe(
        prompt=AR_CAPTION,
        lyrics=AR_LYRICS,
        audio_duration=AR_FRAMES / 25.0,
        generator=torch.Generator().manual_seed(7),
    )
    handle.remove()
    encoders._sample_top_k = original_sample

    frame_hiddens = state.get("frame_hiddens")
    meta["ar_caption"] = AR_CAPTION
    meta["ar_lyrics"] = AR_LYRICS
    meta["ar_seed"] = 7
    meta["ar_text_ids"] = state.get("text_ids")[0].tolist()
    meta["ar_frames"] = int(frame_hiddens.shape[1])
    meta["ar_frame_hiddens"] = save(out, "ar_frame_hiddens", frame_hiddens)
    # One semantic sample then seven depth samples per frame, in loop order.
    meta["ar_samples"] = samples
    codes = [samples[i : i + 8] for i in range(0, len(samples) - len(samples) % 8, 8)]
    meta["ar_codes"] = codes
    np.asarray(codes, dtype=np.int32).tofile(out / "ar_codes.bin")
    for index, hidden in enumerate(hiddens):
        meta[f"ar_last_hidden_{index}"] = save(out, f"ar_last_hidden_{index}", hidden)
    meta["ar_last_hidden_count"] = len(hiddens)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--stage", choices=["components", "flow", "ar"], required=True)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    meta_path = args.out / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    torch.set_grad_enabled(False)

    {"components": dump_components, "flow": dump_flow, "ar": dump_ar}[args.stage](args.model, args.out, meta)

    meta_path.write_text(json.dumps(meta, indent=1))
    print("wrote", meta_path)


if __name__ == "__main__":
    main()
