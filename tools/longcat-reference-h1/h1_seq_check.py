"""CPU check of h1_seq.py (H1_SEQ placement) against the pinned LongCatVideoAvatarPipeline, with a tiny umT5 and a fake tokenizer.

Run: python h1_seq_check.py <reference root (pinned package)> <directory holding h1_seq.py>

Checks, in float32 on the CPU:
1. The prompt result cached after the umT5 is freed equals a fresh encode_prompt of the same prompts (values, shapes, dtypes).
2. The guard refuses a different prompt, a different dtype, and a different classifier-free-guidance flag (cache miss).
3. pipe.to() works with the freed-umT5 shim and a DiT of None.
4. The frame override: unset gives 93; 49 is accepted; 50 is refused.
5. move_dit_and_loras moves the DiT and the LoRA objects both ways (CPU only: it proves the code path, not the GPU residency).
6. The DiT forward and decode hooks log and call through; the decode offload moves the DiT to the host and the next prompt call
   moves it back before the denoise.
Not covered: the demo's H1 block (it needs the GPU and the audio separator), LoRA residency on the GPU, and the memory numbers.
"""

import importlib.util
import os
import sys
import tempfile
import types

# CPU only: hide the GPU so torch.cuda reports unavailable (the scratch environment has no triton).
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

REF_ROOT = os.path.abspath(sys.argv[1])
H1_DIR = os.path.abspath(sys.argv[2])
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, H1_DIR)

import h1_sdpa_check as stubs  # noqa: E402
import h1_int8_meta_check as chk  # noqa: E402

stubs.install_stubs(REF_ROOT)
chk.install_flash_placeholders()
torch.compile = lambda *a, **k: (a[0] if a and callable(a[0]) else (lambda f: f))

# The local scratch copy of the pinned package lacks two modules the pipeline imports for type hints only. Stubs stand in
# for them here; the server has the real files.
for _name, _attr in [("longcat_video.modules.scheduling_flow_match_euler_discrete", "FlowMatchEulerDiscreteScheduler"),
                     ("longcat_video.modules.autoencoder_kl_wan", "AutoencoderKLWan")]:
    if importlib.util.find_spec(_name) is None:
        _mod = types.ModuleType(_name)
        setattr(_mod, _attr, type(_attr, (object,), {}))
        sys.modules[_name] = _mod
# longcat_video/utils/bukcet_config.py (resolution buckets) is not in the scratch copy; the encode path does not call it.
if not os.path.isdir(os.path.join(REF_ROOT, "longcat_video", "utils")):
    _utils = types.ModuleType("longcat_video.utils")
    _utils.__path__ = []
    _bucket = types.ModuleType("longcat_video.utils.bukcet_config")
    _bucket.get_bucket_config = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("bucket config is not used by this check"))
    sys.modules["longcat_video.utils"] = _utils
    sys.modules["longcat_video.utils.bukcet_config"] = _bucket

import h1_seq  # noqa: E402
from transformers import UMT5Config, UMT5EncoderModel  # noqa: E402
from longcat_video.pipeline_longcat_video_avatar import LongCatVideoAvatarPipeline  # noqa: E402

os.environ["H1_MEM_LOG"] = os.path.join(tempfile.gettempdir(), "h1_seq_check_mem.log")
h1_seq.MEM_LOG = os.environ["H1_MEM_LOG"]
if os.path.exists(h1_seq.MEM_LOG):
    os.remove(h1_seq.MEM_LOG)


class FakeTokenizer:
    def __call__(self, prompts, padding, max_length, truncation, add_special_tokens, return_attention_mask, return_tensors):
        ids = torch.zeros(len(prompts), max_length, dtype=torch.long)
        mask = torch.zeros(len(prompts), max_length, dtype=torch.long)
        for i, p in enumerate(prompts):
            toks = [(ord(c) % 60) + 1 for c in p][:max_length]
            ids[i, : len(toks)] = torch.tensor(toks, dtype=torch.long)
            mask[i, : len(toks)] = 1
        return types.SimpleNamespace(input_ids=ids, attention_mask=mask)


def make_text_encoder():
    torch.manual_seed(0)
    cfg = UMT5Config(vocab_size=64, d_model=32, d_ff=64, num_layers=1, num_heads=2, d_kv=16,
                     relative_attention_num_buckets=8, num_decoder_layers=1)
    return UMT5EncoderModel(cfg).eval()


def make_pipe(text_encoder, dit=None):
    class _Vae(types.SimpleNamespace):
        def to(self, *args, **kwargs):
            return self

    vae = _Vae(config=types.SimpleNamespace(scale_factor_temporal=4, scale_factor_spatial=8))
    pipe = LongCatVideoAvatarPipeline(tokenizer=FakeTokenizer(), text_encoder=text_encoder, vae=vae, scheduler=None,
                                      dit=dit, audio_encoder=None, audio_feature_extractor=None, model_type="avatar-v1.5")
    return pipe


PROMPT = "a person speaking"
NEG = "blurry, low quality"
DEVICE = "cpu"
DTYPE = torch.float32


def encode(pipe, prompt=PROMPT, neg=NEG, cfg=True, dtype=DTYPE):
    return pipe.encode_prompt(prompt, negative_prompt=neg, do_classifier_free_guidance=cfg, num_videos_per_prompt=1,
                              max_sequence_length=512, device=DEVICE, dtype=dtype)


def expect_raise(label, fn):
    try:
        fn()
    except RuntimeError as exc:
        print(f"PASS (refused, {label}): {str(exc)[:110]}")
        return
    raise AssertionError(f"{label}: accepted")


def build_cached_pipe(no_grad=True):
    """The demo's H1 order: the encode runs on a pipeline whose text encoder is only held by the pipeline, then the caller drops it.
    Returns the pipeline and the cached result; no local here keeps the module alive."""
    te = make_text_encoder()
    pipe = make_pipe(te)
    pipe._text_guidance_scale = 4.0
    pipe._audio_guidance_scale = 4.0
    cfg = pipe.do_classifier_free_guidance
    assert cfg is True
    if no_grad:
        with torch.no_grad():
            pipe.encode_prompt(PROMPT, negative_prompt=NEG, do_classifier_free_guidance=cfg, num_videos_per_prompt=1,
                               max_sequence_length=512, device=DEVICE, dtype=DTYPE)
    else:
        pipe.encode_prompt(PROMPT, negative_prompt=NEG, do_classifier_free_guidance=cfg, num_videos_per_prompt=1,
                           max_sequence_length=512, device=DEVICE, dtype=DTYPE)
    te = None
    return pipe


def main():
    # 1. reference values from a separate encoder with the same weights (seeded), then the cached encode after release
    ref_te = make_text_encoder()
    ref_pipe = make_pipe(ref_te)
    ref_pipe._text_guidance_scale = 4.0
    ref_pipe._audio_guidance_scale = 4.0
    ref = encode(ref_pipe)
    del ref_pipe, ref_te

    pipe = build_cached_pipe(no_grad=True)
    cached = pipe._prompt_cache
    assert all(t is None or (not t.requires_grad and t.grad_fn is None) for t in cached), "cached tensors carry autograd"
    state = h1_seq.release_text_encoder(pipe, base_bytes=0)  # CPU: no GPU bytes to compare, the referrer check still runs
    h1_seq.install_prompt_guard(pipe, state)
    pipe.device = DEVICE
    got = pipe.encode_prompt(PROMPT, negative_prompt=NEG, do_classifier_free_guidance=True, num_videos_per_prompt=1,
                             max_sequence_length=512, device=DEVICE, dtype=DTYPE)
    for i, (a, b) in enumerate(zip(ref, got)):
        if a is None:
            assert b is None, i
            continue
        assert a.shape == b.shape and a.dtype == b.dtype, (i, a.shape, b.shape, a.dtype, b.dtype)
        assert torch.equal(a, b), f"tensor {i} differs"
    print("PASS: cached result equals a fresh encode_prompt; umT5 collected (weakref dead) after release")

    # 1b. regression: the encode without no_grad keeps the module alive through the autograd graph, so release must refuse
    pipe_graph = build_cached_pipe(no_grad=False)
    try:
        h1_seq.release_text_encoder(pipe_graph, base_bytes=0)
    except RuntimeError as exc:
        print(f"PASS (refused, encode without no_grad keeps umT5 alive): {str(exc)[:100]}")
    else:
        raise AssertionError("release accepted a module still held by the autograd graph")
    del pipe_graph

    # 2. misses
    expect_raise("different prompt", lambda: pipe.encode_prompt("a dog", negative_prompt=NEG, do_classifier_free_guidance=True,
                                                                num_videos_per_prompt=1, max_sequence_length=512, device=DEVICE, dtype=DTYPE))
    expect_raise("different dtype", lambda: pipe.encode_prompt(PROMPT, negative_prompt=NEG, do_classifier_free_guidance=True,
                                                              num_videos_per_prompt=1, max_sequence_length=512, device=DEVICE, dtype=torch.bfloat16))
    expect_raise("different cfg flag", lambda: pipe.encode_prompt(PROMPT, negative_prompt=NEG, do_classifier_free_guidance=False,
                                                                 num_videos_per_prompt=1, max_sequence_length=512, device=DEVICE, dtype=DTYPE))

    # 3. pipe.to with the shim and no DiT
    pipe.to("cpu")
    assert isinstance(pipe.text_encoder, h1_seq.TextEncoderShim)
    print("PASS: pipe.to() with the freed-umT5 shim and dit=None")

    # 4. frame override
    old = os.environ.pop("H1_NUM_FRAMES", None)
    os.environ["H1_SEQ"] = "1"
    assert h1_seq.num_frames(93) == 93
    os.environ["H1_NUM_FRAMES"] = "49"
    assert h1_seq.num_frames(93) == 49
    os.environ["H1_NUM_FRAMES"] = "50"
    try:
        h1_seq.num_frames(93)
    except ValueError:
        print("PASS: frames 93 by default, 49 accepted, 50 refused")
    else:
        raise AssertionError("50 frames accepted")
    os.environ.pop("H1_NUM_FRAMES", None)
    if old is not None:
        os.environ["H1_NUM_FRAMES"] = old

    # 5. DiT and LoRA move helper (CPU round trip)
    class Lora(nn.Module):
        def __init__(self):
            super().__init__()
            self.down = nn.Linear(4, 2)

    class FakeNetwork:
        def __init__(self):
            self.loras = [Lora()]

    class FakeDit(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, 4)
            self.lora_dict = {"dmd": FakeNetwork()}

    dit = FakeDit()
    h1_seq.move_dit_and_loras(dit, "cpu")
    h1_seq.move_dit_and_loras(dit, "cpu")
    assert all(p.device.type == "cpu" for p in dit.parameters())
    assert all(p.device.type == "cpu" for l in dit.lora_dict["dmd"].loras for p in l.parameters())
    print("PASS: move_dit_and_loras round trip (CPU)")

    # 6. hooks on the cached pipeline: forward log, decode offload, and the prompt guard brings the DiT back
    class FakeVae:
        def decode(self, latents, return_dict=False):
            return (latents * 2,)

    class FakeDitForward(FakeDit):
        def forward(self, x):
            return self.lin(x)

    dit6 = FakeDitForward()
    pipe.vae = FakeVae()
    pipe.dit = dit6
    moves = []
    real_move = h1_seq.move_dit_and_loras

    def recording_move(d, device):
        moves.append(str(device))
        real_move(d, device)

    h1_seq.move_dit_and_loras = recording_move
    os.environ["H1_DECODE_OFFLOAD"] = "1"
    h1_seq.install_stage_hooks(pipe, dit6, state, "cpu")
    dit6(torch.ones(1, 4))
    out = pipe._decode_video_latents(torch.ones(2))
    assert torch.equal(out, torch.full((2,), 2.0))
    assert moves == ["cpu"], moves
    assert state["dit_on_host"] is True
    pipe.encode_prompt(PROMPT, negative_prompt=NEG, do_classifier_free_guidance=True, num_videos_per_prompt=1,
                       max_sequence_length=512, device=DEVICE, dtype=DTYPE)
    assert moves == ["cpu", "cpu"], moves  # the guard moved the DiT back
    assert state["dit_on_host"] is False
    h1_seq.move_dit_and_loras = real_move
    os.environ.pop("H1_DECODE_OFFLOAD", None)
    print("PASS: forward log and decode offload run; the next cached prompt call moves the DiT back")

    with open(h1_seq.MEM_LOG, encoding="utf-8") as f:
        lines = f.read().splitlines()
    assert any("denoise_fwd" in l for l in lines) and any("end:decode" in l for l in lines), lines[-5:]
    print(f"PASS: memory log written ({len(lines)} lines) at {h1_seq.MEM_LOG}")
    print("ALL PASS")


if __name__ == "__main__":
    main()
