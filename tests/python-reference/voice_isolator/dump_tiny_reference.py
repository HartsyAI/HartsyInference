"""Tiny-config reference tensors for the Inworld Voice Isolator C# port.

Builds two small random-weight checkpoints (offline and causal) with the upstream PyTorch network, runs the upstream whole-clip
forward, offline renderer and streaming session over saved inputs, and writes the checkpoints, inputs and outputs for
tests/HartsyInference.Audio.Tests/VoiceIsolator/Parity/VoiceIsolatorTinyReferenceTests.cs. Runs on the CPU reference image:

  run.sh -v "<worktree>:/wt" -v "<upstream-src>:/up:ro" -- python /wt/tests/python-reference/voice_isolator/dump_tiny_reference.py \
      --upstream /up --out /wt/tests/python-reference/voice_isolator_reference_tensors

<upstream-src> is the `src` folder of inworld-ai/voice-isolator @ c73b3fab81a870410cbac50d2f05f25edc4d10aa (Apache-2.0); only
model.py, scan.py and streaming.py of inworld/voice_isolation are imported, so torchaudio is not needed. The tiny configs
mirror tests/HartsyInference.Audio.Tests/VoiceIsolator/TinyVoiceIsolatorCheckpoint.cs.
"""

import argparse
import importlib.util
import json
import os
import sys
import types

import numpy as np
import safetensors.torch
import torch


def load_upstream(src: str):
    package = types.ModuleType("inworld")
    package.__path__ = [os.path.join(src, "inworld")]
    sys.modules["inworld"] = package
    sub = types.ModuleType("inworld.voice_isolation")
    sub.__path__ = [os.path.join(src, "inworld", "voice_isolation")]
    sys.modules["inworld.voice_isolation"] = sub
    modules = {}
    for name in ("scan", "model", "streaming"):
        path = os.path.join(src, "inworld", "voice_isolation", f"{name}.py")
        spec = importlib.util.spec_from_file_location(f"inworld.voice_isolation.{name}", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"inworld.voice_isolation.{name}"] = module
        setattr(sub, name, module)
        spec.loader.exec_module(module)
        modules[name] = module
    return modules["model"], modules["streaming"]


def tiny_config(causal: bool) -> dict:
    return {
        "width": 16,
        "block_types": ["transformer", "mamba", "transformer"],
        "num_heads": 2,
        "n_fft": 64,
        "hop_length": 16,
        "sample_rate": 1600,
        "band_plan": [[100, 400], [200, 800]],
        "correction_bound": 0.5,
        "chunk_seconds": 1.0,
        "overlap_seconds": 0.25,
        "exclusive_self_attention": True,
        "exclusive_axes": "time",
        "mamba_state": 4,
        "mamba_conv": 4,
        "mamba_expand": 2,
        "causal": causal,
        "lookahead_frames": 2 if causal else 0,
        "lookahead_frames_choices": [0, 2, 4] if causal else [],
        "attention_window_frames": 6 if causal else 0,
    }


def signal(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float32)
    return (0.3 * np.sin(t * 0.05) + 0.1 * (rng.random(n, dtype=np.float32) - 0.5)).astype(np.float32)


def build(modeling, causal: bool, seed: int):
    config = modeling.VoiceIsolatorConfig.from_dict(tiny_config(causal))
    torch.manual_seed(seed)
    network = modeling.VoiceIsolatorNetwork(config)
    for name, parameter in network.named_parameters():
        if name.endswith("A_log"):
            continue
        parameter.data.normal_(0.0, 0.2)
    return network.eval(), tiny_config(causal)


def save_checkpoint(network, config: dict, path: str) -> None:
    state = {k: v.detach().clone().contiguous() for k, v in network.state_dict().items()}
    safetensors.torch.save_file(state, path, metadata={"config": json.dumps(config)})


def stream_run(streaming, network, audio: np.ndarray, piece: int, window, crossfade, cache) -> np.ndarray:
    session = streaming.Stream(network, block_frames=4, cache_frames=cache, window_frames=window, crossfade_frames=crossfade)
    out = []
    for start in range(0, len(audio), piece):
        out.append(session.push(torch.from_numpy(audio[start:start + piece])).numpy())
    out.append(session.flush().numpy())
    return np.concatenate(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    modeling, streaming = load_upstream(args.upstream)
    os.makedirs(args.out, exist_ok=True)

    offline, offline_config = build(modeling, causal=False, seed=31)
    save_checkpoint(offline, offline_config, os.path.join(args.out, "tiny_offline.safetensors"))
    short = signal(1_200, 4)
    long = signal(4_100, 6)
    with torch.inference_mode():
        np.save(os.path.join(args.out, "offline_short_input.npy"), short)
        np.save(os.path.join(args.out, "offline_short_expected.npy"), offline.enhance(torch.from_numpy(short)).numpy())
        np.save(os.path.join(args.out, "offline_long_input.npy"), long)
        np.save(os.path.join(args.out, "offline_long_expected.npy"), offline.enhance(torch.from_numpy(long)).numpy())

    causal, causal_config = build(modeling, causal=True, seed=32)
    save_checkpoint(causal, causal_config, os.path.join(args.out, "tiny_causal.safetensors"))
    audio = signal(3_200, 7)
    np.save(os.path.join(args.out, "causal_input.npy"), audio)
    with torch.inference_mode():
        # whole-clip causal forward: the upstream reference zero-pads a stream, so compare with the stream outputs below
        np.save(os.path.join(args.out, "causal_whole_expected.npy"), causal(torch.from_numpy(audio)[None])[0].numpy())
        np.save(os.path.join(args.out, "causal_stream_plain_expected.npy"), stream_run(streaming, causal, audio, 500, None, None, None))
        np.save(os.path.join(args.out, "causal_stream_windowed_expected.npy"), stream_run(streaming, causal, audio, 500, 40, 12, 40))
    print("wrote", sorted(os.listdir(args.out)))


if __name__ == "__main__":
    main()
