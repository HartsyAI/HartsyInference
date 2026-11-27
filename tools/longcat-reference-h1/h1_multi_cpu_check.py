"""CPU run of the multi-speaker runner on a tiny checkpoint: the patched demo's real generate(), pipeline, DiT, LoRA mix and dump.

  python h1_multi_cpu_check.py <patched reference root> <pristine reference root> <work dir>

The patched root is the pinned package with h1-sdpa.patch, h1-sdpa-cross.patch and h1-multi.patch applied and h1_seq.py,
h1_multi_dump.py and h1_multi_runner.py copied in; the pristine root is an untouched checkout (only its multi demo is run, on the
patched root's package, since the package needs the SDPA patches without flash-attn). The work dir receives the tiny checkpoint,
the inputs and one folder per run. The checkpoint is stored in BF16 like the release: the engine keeps DiT weights in BF16,
so an F32 file would make the two sides differ by rounding alone. Needs torch, diffusers, transformers, einops, ftfy, safetensors, numpy and PIL (the server's
pins; the lc-ref-base image has them). Each run is its own process.

Runs, all on the distilled 1.6 path with the two LoRAs mixed, two speakers (para), the default boxes:
  pristine   the untouched demo, BF16, two segments
  patched    the patched demo, H1_SEQ=0, BF16, two segments: the latents must equal pristine's bit for bit
  seq        the patched demo, H1_SEQ=1, BF16, two segments, with the step-0 dump: the latents must equal patched's bit for bit
  f32        the patched demo, H1_SEQ=0, the DiT in F32, one segment, with the step-0 dump (for the comparison with the engine)
  guided     the dump asked for with guidance: refused

Real: the runner's entry point (with H1_EAGER_MAP=1 in the f32 run), the demo's generate() and argument parsing, the audio combination (audio_prepare_multi), the mask construction, the
pipeline's generate_ai2v and generate_avc, the scheduler, the avatar DiT loaded from the tiny checkpoint, its LoRA loading and mix,
H1_SEQ's placement code, the runner's size override and the dump hooks.
Stubbed: the tokenizer, a tiny random umT5 (8 tokens), the VAE (average pooling and nearest upsampling), Whisper and its embedding
(seeded by the speech's samples), librosa.load (a WAV reader), soundfile, ffmpeg, the vocal separator (never run), scipy, pyloudnorm,
imageio, torchvision and triton (imported, not used on this path), torch.compile (identity), the distributed setup, the demo's
integer device arguments, and the cuda float32 autocast (a linear layer given mixed types computes in float32). The LoRA rank is 4, not the demo's 128. Not covered: INT8 loading (h1_int8_meta_check.py
covers the loader), memory, speed, BF16 numerics on a GPU, guidance.
"""

import hashlib
import importlib.machinery
import json
import os
import runpy
import subprocess
import sys
import types
import wave

SIZE = (64, 96)
TOKENS = 8
CAPTION = 48
RANK_DIM = 4
CONFIG = dict(
    in_channels=4, out_channels=4, hidden_size=128, depth=2, num_heads=4, caption_channels=CAPTION, mlp_ratio=4, adaln_tembed_dim=64,
    frequency_embedding_size=64, patch_size=[1, 2, 2], enable_flashattn2=True, text_tokens_zero_pad=True, audio_window=5,
    audio_block=5, audio_channel=1280, intermediate_dim=16, output_dim=16, context_tokens=4, vae_scale=4, class_range=24,
    class_interval=4,
)


def stub_modules():
    """Stand-ins for the packages the reference imports and this path does not use, or uses only to read and write audio."""
    import numpy as np

    def load(path, sr=16000, mono=True):
        with wave.open(path, "rb") as handle:
            assert handle.getframerate() == sr and handle.getnchannels() == 1 and handle.getsampwidth() == 2
            raw = handle.readframes(handle.getnframes())
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0, sr

    librosa = types.ModuleType("librosa")
    librosa.load = load
    sys.modules["librosa"] = librosa
    soundfile = types.ModuleType("soundfile")
    soundfile.write = lambda path, data, rate: None
    sys.modules["soundfile"] = soundfile
    for name in ("pyloudnorm", "imageio", "torchvision", "scipy", "scipy.signal"):
        sys.modules[name] = types.ModuleType(name)
    sys.modules["scipy"].signal = sys.modules["scipy.signal"]
    package = types.ModuleType("audio_separator")
    package.__path__ = []
    separator = types.ModuleType("audio_separator.separator")

    class Separator:
        onnx_execution_provider = ["CPUExecutionProvider"]

        def __init__(self, *args, **kwargs):
            # The pristine demo always builds one; the patched demo must not without H1_SEPARATE=1.
            if os.environ.get("H1_CHECK_PRISTINE") != "1":
                raise RuntimeError("the separator must not be constructed without H1_SEPARATE=1")

        def load_model(self, name):
            pass

    separator.Separator = Separator
    sys.modules["audio_separator"] = package
    sys.modules["audio_separator.separator"] = separator
    vocals = types.ModuleType("longcat_video.audio_process.vocal_separator")

    def separate_vocals(_separator, source_path, duration=None):
        # Only the pristine demo calls this: it returns what the pass-through returns, so the two demos hear the same speech.
        import math

        mono, rate = load(source_path)
        return mono[: math.ceil(duration * rate)], rate

    vocals.separate_vocals = separate_vocals
    sys.modules["longcat_video.audio_process.vocal_separator"] = vocals
    for name in ("librosa", "soundfile", "pyloudnorm", "imageio", "torchvision", "scipy", "scipy.signal", "audio_separator",
                 "audio_separator.separator"):
        # diffusers asks importlib for each optional package, which refuses a module without a spec
        sys.modules[name].__spec__ = importlib.machinery.ModuleSpec(name, None)


def map_cpu_devices():
    """The demo passes the integer local rank as a device; on a CPU machine it becomes 'cpu'."""
    import torch
    import torch.nn as nn
    from torch.overrides import TorchFunctionMode

    def cpu(value):
        return "cpu" if isinstance(value, int) and not isinstance(value, bool) else value

    class CpuDevices(TorchFunctionMode):
        def __torch_function__(self, function, types_, args=(), kwargs=None):
            kwargs = dict(kwargs or {})
            if "device" in kwargs:
                kwargs["device"] = cpu(kwargs["device"])
            if function is torch.Tensor.to:
                args = (args[0], *(cpu(a) for a in args[1:]))
            if function is nn.functional.linear and args[0].dtype != args[1].dtype:
                # The reference's autocast('cuda', float32) blocks do nothing on a CPU: mixed types there compute in float32.
                args = tuple(a.float() if a is not None else None for a in args)
            return function(*args, **kwargs)

    module_to = nn.Module.to
    nn.Module.to = lambda self, *a, **k: module_to(self, *(cpu(v) for v in a), **{n: cpu(v) if n == "device" else v for n, v in k.items()})
    generator = torch.Generator
    torch.Generator = lambda device=None: generator(device="cpu" if isinstance(device, int) else (device or "cpu"))
    torch.cuda.device_count = lambda: 1
    torch.cuda.set_device = lambda *a, **k: None
    torch.cuda.empty_cache = lambda: None
    torch.cuda.ipc_collect = lambda: None
    torch.distributed.init_process_group = lambda *a, **k: None
    torch.distributed.get_rank = lambda *a, **k: 0
    torch.distributed.get_world_size = lambda *a, **k: 1
    return CpuDevices()


def write_inputs(work):
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(5)
    image = os.path.join(work, "portrait.png")
    pixels = rng.integers(0, 255, (75, 121, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(image)
    pixels.tofile(os.path.join(work, "portrait.rgb"))  # the same picture without a PNG decoder, for a harness that runs the engine
    audio = {}
    for name, seconds, hertz in (("person1", 5.0, 220.0), ("person2", 6.0, 330.0)):
        t = np.arange(int(seconds * 16000)) / 16000
        samples = (0.3 * np.sin(2 * np.pi * hertz * t) + 0.05 * rng.standard_normal(t.size)).clip(-1, 1)
        audio[name] = os.path.join(work, name + ".wav")
        with wave.open(audio[name], "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes((samples * 32767).astype("<i2").tobytes())
    with open(os.path.join(work, "input.json"), "w", encoding="utf-8") as handle:
        json.dump({"prompt": "two people speaking", "cond_image": image, "cond_audio": audio, "audio_type": "para"}, handle)


def write_checkpoint(work):
    """A tiny release the engine can read too: base_model (one shard and its index), loraA, loraB and the scheduler config."""
    import torch
    from safetensors.torch import save_file

    from longcat_video.modules.avatar.longcat_video_dit_avatar import LongCatVideoAvatarTransformer3DModel
    from longcat_video.modules.lora_utils import create_lora_network

    root = os.path.join(work, "avatar-tiny")
    base = os.path.join(root, "base_model")
    os.makedirs(base, exist_ok=True)
    torch.manual_seed(41)
    model = LongCatVideoAvatarTransformer3DModel(**{**CONFIG, "patch_size": tuple(CONFIG["patch_size"]), "cp_split_hw": [1, 1]})
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("q_norm.weight") or name.endswith("k_norm.weight"):
                parameter.copy_(3.0 * (1.0 + 0.2 * torch.randn_like(parameter)))  # a map that separates the regions
            else:
                parameter.copy_(torch.randn_like(parameter) * 0.05)
    # BF16 like the release: an F32 file would let the two sides disagree on who rounds.
    state = {k: v.contiguous().bfloat16() for k, v in model.state_dict().items()}
    shard = "diffusion_pytorch_model-00001-of-00001.safetensors"
    save_file(state, os.path.join(base, shard))
    with open(os.path.join(base, "diffusion_pytorch_model.safetensors.index.json"), "w", encoding="utf-8") as handle:
        json.dump({"metadata": {}, "weight_map": {k: shard for k in state}}, handle)
    with open(os.path.join(base, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(CONFIG, handle)
    targets = (("attn.qkv", 3), ("attn.proj", 1), ("cross_attn.q_linear", 1), ("cross_attn.kv_linear", 2), ("ffn.w1", 1),
               ("ffn.w2", 1), ("ffn.w3", 1))
    for folder, seed in (("loraA", 81), ("loraB", 82)):
        names = {}
        for block in range(CONFIG["depth"]):
            for target, parts in targets:
                key = "lora___lorahyphen___" + f"blocks.{block}.{target}".replace(".", "___lorahyphen___")
                names[key + ".lora_down.weight"] = None
                for part in range(parts if parts > 1 else 0):
                    names[f"{key}.lora_up.blocks.{part}.weight"] = None
        network = create_lora_network(model, names, multiplier=1.0, network_dim=RANK_DIM, network_alpha=2)
        torch.manual_seed(seed)
        weights = {k: (v if k.endswith("alpha_scale") else (torch.randn_like(v) * 0.1).bfloat16()).contiguous()
                   for k, v in sorted(network.state_dict().items())}  # sorted: the module order varies between processes
        os.makedirs(os.path.join(root, folder), exist_ok=True)
        save_file(weights, os.path.join(root, folder, "lora.safetensors"))
    os.makedirs(os.path.join(root, "scheduler"), exist_ok=True)
    with open(os.path.join(root, "scheduler", "scheduler_config.json"), "w", encoding="utf-8") as handle:
        json.dump({"num_train_timesteps": 1000, "shift": 7.0, "use_dynamic_shifting": False}, handle)


def run_one(name, root, pristine, work):
    """One process: import the demo, put the stand-ins on it, run generate(), save every segment's latents."""
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import UMT5Config, UMT5EncoderModel

    import importlib.util

    import longcat_video.context_parallel.context_parallel_util as parallel
    from longcat_video.pipeline_longcat_video_avatar import LongCatVideoAvatarPipeline

    parallel.init_context_parallel = lambda **_: None
    parallel.get_cp_rank = lambda: 0
    parallel.get_cp_size = lambda: 1
    parallel.get_optimal_split = lambda _: [1, 1]

    demo_name = "run_demo_avatar_multi_audio_to_video"
    demo_root = pristine if name == "pristine" else root
    spec = importlib.util.spec_from_file_location(demo_name, os.path.join(demo_root, demo_name + ".py"))
    demo = importlib.util.module_from_spec(spec)
    sys.modules[demo_name] = demo
    spec.loader.exec_module(demo)

    class Tokenizer:
        def __call__(self, prompts, padding, max_length, truncation, add_special_tokens, return_attention_mask, return_tensors):
            ids = torch.zeros(len(prompts), max_length, dtype=torch.long)
            mask = torch.zeros(len(prompts), max_length, dtype=torch.long)
            for i, prompt in enumerate(prompts):
                tokens = [(ord(c) % 60) + 1 for c in prompt][: max_length - 2]
                ids[i, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)
                mask[i, : len(tokens)] = 1
            return types.SimpleNamespace(input_ids=ids, attention_mask=mask)

    def text_encoder(*_, **__):
        torch.manual_seed(0)
        config = UMT5Config(vocab_size=64, d_model=CAPTION, d_ff=64, num_layers=1, num_heads=2, d_kv=16,
                            relative_attention_num_buckets=8, num_decoder_layers=1)
        return UMT5EncoderModel(config).eval().to(torch.bfloat16)

    prompt_embeds = LongCatVideoAvatarPipeline._get_t5_prompt_embeds

    def short_prompt(self, *args, **kwargs):
        kwargs["max_sequence_length"] = TOKENS
        return prompt_embeds(self, *args, **kwargs)

    class Vae(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("unused", torch.zeros(1, dtype=torch.bfloat16))
            self.config = types.SimpleNamespace(scale_factor_temporal=4, scale_factor_spatial=8, z_dim=4, latents_mean=[0.0] * 4,
                                                latents_std=[1.0] * 4)

        @property
        def dtype(self):
            return torch.bfloat16

        def encode(self, x):
            x = x.float()
            frames = (x.shape[2] - 1) // 4 + 1
            pooled = F.adaptive_avg_pool3d(torch.cat([x, x.mean(1, keepdim=True)], 1), (frames, x.shape[3] // 8, x.shape[4] // 8))
            latent = pooled.to(self.dtype)
            return types.SimpleNamespace(latent_dist=types.SimpleNamespace(mode=lambda: latent))

        def decode(self, latents, return_dict=False):
            frames = (latents.shape[2] - 1) * 4 + 1
            size = (frames, latents.shape[3] * 8, latents.shape[4] * 8)
            return (F.interpolate(latents[:, :3].float(), size=size, mode="nearest").clamp(-1, 1),)

    def embedding(self, speech_array, fps=32, device="cpu", sample_rate=16000, model_type=None, use_loudness_norm=True):
        seed = int.from_bytes(hashlib.sha256(np.asarray(speech_array, dtype=np.float32).tobytes()).digest()[:4], "little")
        frames = int(len(speech_array) / sample_rate * fps)
        return torch.from_numpy(np.random.default_rng(seed).standard_normal((frames, 5, 1280)).astype(np.float32))

    class Whisper:
        dtype = torch.float32

        def to(self, *_, **__):
            return self

    demo.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *a, **k: Tokenizer())
    demo.UMT5EncoderModel = types.SimpleNamespace(from_pretrained=text_encoder)
    demo.AutoencoderKLWan = types.SimpleNamespace(from_pretrained=lambda *a, **k: Vae())
    demo.get_audio_encoder = lambda *a, **k: Whisper()
    demo.get_audio_feature_extractor = lambda *a, **k: None
    demo.save_video_ffmpeg = lambda *a, **k: None
    LongCatVideoAvatarPipeline._get_t5_prompt_embeds = short_prompt
    LongCatVideoAvatarPipeline.get_audio_embedding = embedding

    dit_class = demo.LongCatVideoAvatarTransformer3DModel
    load_lora = dit_class.load_lora
    dit_class.load_lora = lambda self, path, key, **k: load_lora(self, path, key, **{**k, "lora_network_dim": RANK_DIM})
    if name == "f32":
        original = dit_class.from_pretrained.__func__
        dit_class.from_pretrained = classmethod(lambda cls, *a, **k: original(cls, *a, **{**k, "torch_dtype": torch.float32}))

    recorded = []
    for method in ("generate_ai2v", "generate_avc"):
        inner = getattr(LongCatVideoAvatarPipeline, method)

        def keep(self, *args, inner=inner, **kwargs):
            result = inner(self, *args, **kwargs)
            recorded.append(result[1].detach().float().numpy().copy())
            return result

        setattr(LongCatVideoAvatarPipeline, method, keep)

    segments = "1" if name in ("f32", "guided") else "2"
    argv = ["demo", "--checkpoint_dir", os.path.join(work, "avatar-tiny"), "--model_type", "avatar-v1.6",
            "--input_json", os.path.join(work, "input.json"), "--resolution", "480p", "--num_segments", segments,
            "--output_dir", os.path.join(work, name, "out")]
    if name == "guided":
        argv += ["--num_inference_steps", "2", "--mix_steps", "0"]
    else:
        argv += ["--use_distill", "--mix_steps", "4", "--mix_alpha", "1.0"]
    sys.argv = argv
    with map_cpu_devices():
        # The runner as the server starts it: it imports the demo (the module prepared above), installs and calls generate().
        runpy.run_path(os.path.join(root, "h1_multi_runner.py"), run_name="__main__")
    for index, latent in enumerate(recorded):
        np.save(os.path.join(work, name, f"latent_{index}.npy"), latent)
    print(f"[check] {name}: {len(recorded)} segments")


def orchestrate(root, pristine, work):
    import numpy as np

    os.makedirs(work, exist_ok=True)
    write_inputs(work)
    subprocess.run([sys.executable, __file__, root, pristine, work, "--checkpoint"], check=True)
    results = {}
    for name in ("pristine", "patched", "seq", "f32", "guided"):
        folder = os.path.join(work, name)
        os.makedirs(folder, exist_ok=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", RANK="0", WORLD_SIZE="1", H1_ATTN="sdpa", H1_SIZE=f"{SIZE[0]}x{SIZE[1]}",
                   H1_SEQ="1" if name == "seq" else "0", H1_SEPARATE="0", H1_MEM_LOG=os.path.join(folder, "h1_mem.log"),
                   H1_MULTI_BLOCKS="0,1")
        if name in ("seq", "f32", "guided"):
            env["H1_MULTI_DUMP_DIR"] = os.path.join(folder, "dump")
        env["H1_EAGER_MAP"] = "1" if name == "f32" else "0"
        env["H1_CHECK_PRISTINE"] = "1" if name == "pristine" else "0"
        if name != "pristine":
            env["H1_DUMP_DIR"] = os.path.join(folder, "frames")
        done = subprocess.run([sys.executable, __file__, root, pristine, work, "--run", name], env=env, cwd=folder,
                              capture_output=True, text=True)
        with open(os.path.join(folder, "log.txt"), "w", encoding="utf-8") as handle:
            handle.write(done.stdout + done.stderr)
        results[name] = done
        if name == "guided":
            assert done.returncode != 0 and "covers the distilled path only" in done.stderr, done.stderr[-2000:]
            print("PASS guided: the dump refuses a guided run")
        else:
            assert done.returncode == 0, f"{name} failed:\n{done.stderr[-4000:]}"

    def latent(name, index):
        return np.load(os.path.join(work, name, f"latent_{index}.npy"))

    for index in range(2):
        assert np.array_equal(latent("pristine", index), latent("patched", index)), f"segment {index}: the patch changed the latents"
        assert np.array_equal(latent("patched", index), latent("seq", index)), f"segment {index}: H1_SEQ changed the latents"
    print("PASS latents: pristine == patched == H1_SEQ=1, both segments, bit for bit")
    with open(os.path.join(work, "seq", "h1_mem.log"), encoding="utf-8") as handle:
        stages = [line.split("stage=")[1].split()[0] for line in handle if "stage=" in line]
    wanted = ["begin:text_load", "end:prompt", "text_released", "begin:audio", "end:audio", "begin:dit_to_gpu", "begin:denoise_1",
              "denoise_fwd", "begin:decode"]
    remaining = iter(stages)
    assert all(any(stage == found for found in remaining) for stage in wanted), stages
    print("PASS sequential order:", " -> ".join(dict.fromkeys(stages)))
    assert os.path.exists(os.path.join(work, "seq", "frames", "frame_000.png"))
    assert os.path.exists(os.path.join(work, "seq", "frames", "segment_01", "final_latent.npy"))
    for name in ("seq", "f32"):
        dump = os.path.join(work, name, "dump")
        conditioning = {f: np.load(os.path.join(dump, "conditioning", f)) for f in sorted(os.listdir(os.path.join(dump, "conditioning")))}
        assert sorted(conditioning) == ["audio_embedding_step00.npy", "input_latent_step00.npy", "text_embeddings_step00.npy",
                                        "text_mask_step00.npy"], list(conditioning)
        assert conditioning["audio_embedding_step00.npy"].shape == (2, 93, 5 * 5 * 1280)
        assert conditioning["input_latent_step00.npy"].shape == (1, 4, 24, SIZE[0] // 8, SIZE[1] // 8)
        assert conditioning["text_embeddings_step00.npy"].shape == (1, TOKENS, CAPTION)
        probe = lambda p: np.load(os.path.join(dump, "probe", f"call00_probe_{p}.npy"))
        noisy = 23 * (SIZE[0] // 16) * (SIZE[1] // 16)
        for block in (0, 1):
            attention_map = probe(f"block{block}.ref_attn_map")
            assert attention_map.shape == (3, noisy), attention_map.shape
            assert probe(f"block{block}.query_positions").shape == (noisy,)
            assert probe(f"block{block}.key_positions").shape == (23 * 2 * CONFIG["context_tokens"],)
            assert probe(f"block{block}.audio_cross_attn").shape == (noisy, CONFIG["hidden_size"])
            keys = probe(f"block{block}.key_positions")[: 2 * CONFIG["context_tokens"]]
            assert np.array_equal(keys, np.repeat([2.0, 22.0], CONFIG["context_tokens"]).astype(np.float32)), keys
            print(f"  {name} block {block}: classes {np.bincount(np.argmax(attention_map, 0), minlength=3).tolist()}")
        assert probe("velocity").shape == conditioning["input_latent_step00.npy"].shape
        assert not np.isnan(probe("velocity")).any()
    print("PASS dumps: conditioning files and probes of the seq and f32 runs")
    print("ALL PASS")


def main():
    root, pristine, work = (os.path.abspath(p) for p in sys.argv[1:4])
    if len(sys.argv) == 4:
        orchestrate(root, pristine, work)
        return
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    import h1_sdpa_check

    h1_sdpa_check.install_stubs(root)
    sys.path.insert(0, root)
    stub_modules()
    import torch

    # The reference compiles its map function as it is imported; this image has no triton, so compile is the identity here.
    torch.compile = lambda function=None, **_: function if function is not None else (lambda f: f)
    if sys.argv[4] == "--checkpoint":
        write_checkpoint(work)
    else:
        run_one(sys.argv[5], root, pristine, work)


if __name__ == "__main__":
    main()
