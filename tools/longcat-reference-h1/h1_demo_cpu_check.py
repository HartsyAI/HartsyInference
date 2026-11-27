"""CPU run of the H1 demo's generate() through to the save, with the models stubbed.

Run: python h1_demo_cpu_check.py <reference root (pinned package)> <directory with the H1-patched demo and h1_seq.py>

Real: the demo's generate() and its argument parsing; the H1_SEQ block (umT5 precompute under no_grad, release with the weakref
check, whisper alone, DiT, LoRA calls, pipe construction and pipe.to); the pipeline's encode_prompt with its cache and the H1
guard; the pipeline's _decode_video_latents with the optional DiT offload; the step-dump-free output path; _h1_dump; the save.
Stubbed: the tokenizer, umT5 (a tiny real UMT5 encoder), the VAE (a tiny interpolating decoder), the scheduler, the DiT (a tiny
module with the attributes the demo reads, and LoRA methods that record), whisper and its embedding (random, right shape), the
flow of the denoise (generate_ai2v returns a decoded latent, not a sampled one), and ffmpeg (the save call is recorded).
On a CPU-only machine the demo's int device arguments (local_rank = 0) are mapped to 'cpu' for tensors, modules and generators.
This proves the control flow and the H1 placement code run to the save. It does not prove the numerics, the sampler, or memory.
"""

import argparse
import contextlib
import importlib.util
import json
import os
import struct
import sys
import tempfile
import types
import wave

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["H1_SEQ"] = "1"
os.environ["H1_SEPARATE"] = "0"

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

REF_ROOT = os.path.abspath(sys.argv[1])
H1_DIR = os.path.abspath(sys.argv[2])
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REF_ROOT)
sys.path.insert(0, H1_DIR)  # last insert wins: the H1-patched demo must come before the pristine copy in REF_ROOT
sys.path.insert(0, HERE)

import h1_sdpa_check as stubs  # noqa: E402
import h1_int8_meta_check as chk  # noqa: E402

stubs.install_stubs(REF_ROOT)
chk.install_flash_placeholders()
torch.compile = lambda *a, **k: (a[0] if a and callable(a[0]) else (lambda f: f))

# A CPU machine has no audio separator; the demo imports it at module level. Stub only if missing.
if importlib.util.find_spec("audio_separator") is None:
    pkg = types.ModuleType("audio_separator")
    pkg.__path__ = []
    sep = types.ModuleType("audio_separator.separator")

    class Separator:  # never constructed with H1_SEPARATE=0
        def __init__(self, *a, **k):
            raise RuntimeError("Separator must not be constructed with H1_SEPARATE=0")

    sep.Separator = Separator
    sys.modules["audio_separator"] = pkg
    sys.modules["audio_separator.separator"] = sep

# Map the demo's int devices (local_rank = 0) to CPU, and make the CUDA-only calls no-ops.
_orig_tensor_to = torch.Tensor.to
_orig_module_to = nn.Module.to


def _cpu_arg(args):
    return tuple("cpu" if (isinstance(a, int) and not isinstance(a, bool)) else a for a in args)


def _cpu_kwargs(kwargs):
    if isinstance(kwargs.get("device"), int) and not isinstance(kwargs.get("device"), bool):
        kwargs = dict(kwargs, device="cpu")
    return kwargs


torch.Tensor.to = lambda self, *a, **k: _orig_tensor_to(self, *_cpu_arg(a), **_cpu_kwargs(k))
nn.Module.to = lambda self, *a, **k: _orig_module_to(self, *_cpu_arg(a), **_cpu_kwargs(k))
_orig_generator = torch.Generator
torch.Generator = lambda device=None: _orig_generator(device="cpu" if isinstance(device, int) else device)
torch.cuda.device_count = lambda: 1
torch.cuda.set_device = lambda *a, **k: None
torch.cuda.ipc_collect = lambda *a, **k: None
torch.distributed.init_process_group = lambda *a, **k: None
torch.distributed.get_rank = lambda *a, **k: 0
torch.distributed.get_world_size = lambda *a, **k: 1

sys.path.insert(0, os.path.join(HERE))
from h1_seq_check import FakeTokenizer, make_text_encoder  # noqa: E402  (tiny umT5 and fake tokenizer)

if importlib.util.find_spec("longcat_video.audio_process.vocal_separator") is None:
    # the scratch copy has no separator module; with H1_SEPARATE=0 the demo only needs the name to import
    _vs = types.ModuleType("longcat_video.audio_process.vocal_separator")
    _vs.separate_vocals = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("separate_vocals must not run with H1_SEPARATE=0"))
    sys.modules["longcat_video.audio_process.vocal_separator"] = _vs

import longcat_video.context_parallel.context_parallel_util as cp  # noqa: E402
import longcat_video.pipeline_longcat_video_avatar as pipeline_mod  # noqa: E402
from longcat_video.pipeline_longcat_video_avatar import LongCatVideoAvatarPipeline  # noqa: E402

cp.init_context_parallel = lambda **k: None
cp.get_cp_rank = lambda: 0
cp.get_cp_size = lambda: 1
cp.get_optimal_split = lambda n: None

# load the H1-patched demo by path: install_stubs() puts REF_ROOT (which has the pristine demo) first on sys.path
_demo_path = os.path.join(H1_DIR, "run_demo_avatar_single_audio_to_video.py")
_demo_spec = importlib.util.spec_from_file_location("run_demo_avatar_single_audio_to_video", _demo_path)
demo = importlib.util.module_from_spec(_demo_spec)
sys.modules["run_demo_avatar_single_audio_to_video"] = demo
_demo_spec.loader.exec_module(demo)
assert os.path.dirname(os.path.abspath(demo.__file__)) == H1_DIR, demo.__file__

calls = {"save": [], "lora": [], "mix": [], "encode": 0, "generate": []}


class FakeVae(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(scale_factor_temporal=4, scale_factor_spatial=8, z_dim=16,
                                            latents_mean=[0.0] * 16, latents_std=[1.0] * 16)

    def decode(self, latents, return_dict=False):
        out = F.interpolate(latents[:, :3].float(), scale_factor=(4, 8, 8), mode="nearest")
        return (out,)

    def encode(self, x):
        raise RuntimeError("the stub generate_ai2v does not encode the condition image")


class FakeScheduler:
    def __init__(self, *a, **k):
        self.config = types.SimpleNamespace()

    @classmethod
    def from_pretrained(cls, *a, **k):
        return cls()


class TinyDit(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 4).to(torch.bfloat16)
        self.config = types.SimpleNamespace(in_channels=16)
        self.lora_dict = {}
        self.lora_mix_config = None
        self.cp_split_hw = None

    @property
    def dtype(self):
        return torch.bfloat16

    def forward(self, x):
        return self.lin(x)

    def load_lora(self, path, key, **kwargs):
        calls["lora"].append((os.path.basename(os.path.dirname(path)), key))
        self.lora_dict[key] = types.SimpleNamespace(loras=[])

    def enable_lora_mix(self, dmd_key, opsd_key, mix_alpha, mix_steps):
        calls["mix"].append((dmd_key, opsd_key, mix_alpha, mix_steps))
        self.lora_mix_config = {"mix_steps": mix_steps}


class FakeWhisper:
    dtype = torch.float32

    def to(self, *a, **k):
        return self


def fake_generate_ai2v(self, image, prompt, negative_prompt, resolution, num_frames, num_inference_steps,
                       text_guidance_scale, audio_guidance_scale, output_type, generator, audio_emb, use_distill, **kwargs):
    """Runs the H1-relevant part of the real method: the same self-attribute setup, the real encode_prompt call with the
    same arguments, and the real _decode_video_latents. The denoise loop is replaced by a fixed latent."""
    self._text_guidance_scale = text_guidance_scale
    self._audio_guidance_scale = audio_guidance_scale
    dit_dtype = self.dit.dtype
    prompt_embeds, mask, neg_embeds, neg_mask = self.encode_prompt(
        prompt=prompt, negative_prompt=negative_prompt, do_classifier_free_guidance=self.do_classifier_free_guidance,
        num_videos_per_prompt=1, max_sequence_length=512, dtype=dit_dtype, device=self.device)
    calls["encode"] += 1
    calls["generate"].append({"stage": "ai2v", "frames": num_frames, "steps": num_inference_steps,
                              "cfg": self.do_classifier_free_guidance, "dit_dtype": str(dit_dtype),
                              "prompt_embed_shape": tuple(prompt_embeds.shape)})
    latent = torch.randn(1, 16, 2, 4, 4, generator=None)
    self.dit(torch.zeros(1, 4, dtype=dit_dtype))  # one denoise forward through the (wrapped) DiT
    video = self._decode_video_latents(latent)  # the real decode path, with the H1 offload wrapper when enabled
    out = ((video.clamp(-1, 1) + 1) / 2).permute(0, 2, 3, 4, 1).float().numpy()
    return (out, latent)


def fake_get_audio_embedding(self, speech_array, fps=32, device="cpu", sample_rate=16000, model_type=None,
                             use_loudness_norm=False):
    return torch.randn(300, 5, 8)


def fake_save_video(output_tensor, path, audio_path, fps, quality):
    calls["save"].append((os.path.basename(path), tuple(output_tensor.shape)))


def write_inputs(tmp):
    from PIL import Image
    png = os.path.join(tmp, "cond.png")
    Image.new("RGB", (16, 16), (120, 60, 30)).save(png)
    wav = os.path.join(tmp, "speech.wav")
    with wave.open(wav, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(struct.pack("<" + "h" * 16000, *([1000] * 16000)))
    ckpt = os.path.join(tmp, "avatar-1.6")
    os.makedirs(os.path.join(ckpt, "loraA"))
    os.makedirs(os.path.join(ckpt, "loraB"))
    open(os.path.join(ckpt, "loraA", "lora.safetensors"), "wb").close()
    open(os.path.join(ckpt, "loraB", "lora.safetensors"), "wb").close()
    inp = os.path.join(tmp, "input.json")
    with open(inp, "w", encoding="utf-8") as f:
        json.dump({"prompt": "a person speaking", "cond_audio": {"person1": wav}, "cond_image": png}, f)
    return ckpt, inp


def main():
    tmp = tempfile.mkdtemp()
    ckpt, inp = write_inputs(tmp)
    out_dir = os.path.join(tmp, "out")
    os.environ["H1_MEM_LOG"] = os.path.join(tmp, "h1_mem.log")
    import h1_seq  # noqa: E402
    h1_seq.MEM_LOG = os.environ["H1_MEM_LOG"]  # the module read the variable when it was first imported
    os.environ["H1_DUMP_DIR"] = os.path.join(tmp, "dump")
    os.environ["H1_DECODE_OFFLOAD"] = os.environ.get("H1_CHECK_DECODE_OFFLOAD", "1")
    os.environ["H1_ATTN"] = "sdpa"
    os.chdir(tmp)

    # stubs for the demo's module-level names
    class FakeTokenizerFactory:
        @staticmethod
        def from_pretrained(*a, **k):
            return FakeTokenizer()

    def umt5_from_pretrained(*a, **k):
        return make_text_encoder()

    demo.AutoTokenizer = FakeTokenizerFactory
    demo.UMT5EncoderModel = types.SimpleNamespace(from_pretrained=umt5_from_pretrained)
    demo.AutoencoderKLWan = types.SimpleNamespace(from_pretrained=lambda *a, **k: FakeVae())
    demo.FlowMatchEulerDiscreteScheduler = FakeScheduler
    demo.load_quantized_dit = lambda *a, **k: TinyDit()
    demo.get_audio_encoder = lambda path, mt: FakeWhisper()
    demo.get_audio_feature_extractor = lambda path, mt: None
    demo.save_video_ffmpeg = fake_save_video
    LongCatVideoAvatarPipeline.generate_ai2v = fake_generate_ai2v
    LongCatVideoAvatarPipeline.get_audio_embedding = fake_get_audio_embedding

    argv = ["demo", "--checkpoint_dir", ckpt, "--model_type", "avatar-v1.6", "--use_distill", "--use_int8",
            "--mix_steps", "4", "--mix_alpha", "1.0", "--input_json", inp, "--resolution", "480p",
            "--stage_1", "ai2v", "--num_segments", "1", "--num_inference_steps", "8", "--output_dir", out_dir]
    sys.argv = argv
    args = demo._parse_args()
    demo.generate(args)

    # assertions on what ran
    assert calls["encode"] == 1, calls["encode"]
    assert calls["generate"][0]["dit_dtype"] == "torch.bfloat16"
    assert calls["generate"][0]["cfg"] is False, "distilled 1.5 path runs without CFG"
    assert [c[1] for c in calls["lora"]] == ["dmd", "opsd"], calls["lora"]
    assert calls["mix"] and calls["mix"][0][2:] == (1.0, 4), calls["mix"]
    assert calls["save"] == [("ai2v_demo_1", (8, 32, 32, 3))], calls["save"]
    assert os.path.exists(os.path.join(os.environ["H1_DUMP_DIR"], "final_latent.npy"))
    frames = sorted(f for f in os.listdir(os.environ["H1_DUMP_DIR"]) if f.startswith("frame_"))
    assert len(frames) == 8, frames
    with open(os.environ["H1_MEM_LOG"], encoding="utf-8") as f:
        log = f.read().splitlines()
    stages = [l.split("stage=")[1].split()[0] for l in log if "stage=" in l]
    expected_order = ["begin:text_load", "end:text_load", "begin:prompt", "end:prompt", "text_released",
                      "begin:audio", "end:audio", "begin:dit_to_gpu", "end:dit_to_gpu", "begin:denoise_1",
                      "denoise_fwd", "begin:decode", "end:decode"]
    it = iter(stages)
    missing = [s for s in expected_order if not any(s == x for x in it)]
    assert not missing, (missing, stages)
    print("ORDER:", " -> ".join(dict.fromkeys(stages)))
    print("PASS: demo generate() ran to the save with H1_SEQ=1 and H1_SEPARATE=0 (separator not constructed)")
    print(f"PASS: save recorded {calls['save']}; {len(frames)} frames dumped; LoRA calls {calls['lora']}")
    print("ALL PASS")


if __name__ == "__main__":
    main()
