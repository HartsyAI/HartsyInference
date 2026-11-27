"""H1 sequential device placement and memory log (H1_SEQ=1). Device placement only: no tensor value or dtype changes.

Order on one GPU (23.52 GiB):
  1. umT5 (text_encoder) to the GPU alone; encode the prompt and the negative prompt once through the pipeline's own
     encode_prompt (same key, dtype and device as generate uses); keep the result in pipe._prompt_cache.
  2. umT5 is freed. From then on encode_prompt is guarded: a call whose key is not the cached key raises. It never reloads umT5.
  3. whisper (audio encoder) alone on the GPU for the audio embedding; freed.
  4. DiT and LoRAs to the GPU (the stock order: dit.to, then the LoRA block), then generation as stock.
  5. Optional (H1_DECODE_OFFLOAD=1): DiT and LoRA objects go to the host before each VAE decode, and back before the next denoise.

Memory: torch.cuda peaks are reset at each stage and logged after every denoise forward, flushed to stdout and to H1_MEM_LOG
(default h1_mem.log in the working directory), so a failed run still leaves the numbers.
"""

import gc
import os
import time
import types
import weakref
import torch

GIB = float(2 ** 30)
_T0 = time.monotonic()
MEM_LOG = os.environ.get("H1_MEM_LOG", "h1_mem.log")


def enabled():
    return os.environ.get("H1_SEQ", "0") == "1"


def num_frames(default):
    """The reference's frame count, unless H1_SEQ=1 and H1_NUM_FRAMES is set (the fallback). Frames must be 1 mod 4."""
    if not enabled() or "H1_NUM_FRAMES" not in os.environ:
        return default
    n = int(os.environ["H1_NUM_FRAMES"])
    if n < 14 or (n - 1) % 4 != 0:
        raise ValueError(f"H1_NUM_FRAMES={n}: need n >= 14 (13 condition frames) and (n-1) % 4 == 0")
    return n


def check_supported(model_type, context_parallel_size, use_fp8, compile_dit, separate):
    """Refuse configurations this placement does not cover, rather than run them differently."""
    if model_type != "avatar-v1.5":
        raise NotImplementedError(f"H1_SEQ covers avatar-v1.5 (1.6 maps to it), got {model_type}")
    if context_parallel_size != 1:
        raise NotImplementedError("H1_SEQ covers context_parallel_size 1 only")
    if use_fp8 or compile_dit:
        raise NotImplementedError("H1_SEQ does not cover --use_fp8 or --compile_dit")
    if separate:
        raise NotImplementedError("H1_SEQ needs H1_SEPARATE=0 (pass-through speech)")


def log(stage, **fields):
    if torch.cuda.is_available():
        dev = torch.cuda.current_device()
        now_a = torch.cuda.memory_allocated(dev) / GIB
        now_r = torch.cuda.memory_reserved(dev) / GIB
        peak_a = torch.cuda.max_memory_allocated(dev) / GIB
        peak_r = torch.cuda.max_memory_reserved(dev) / GIB
        body = f"alloc_now={now_a:.3f} reserved_now={now_r:.3f} peak_alloc={peak_a:.3f} peak_reserved={peak_r:.3f} (GiB)"
    else:
        body = "cuda unavailable (CPU check)"
    extra = " ".join(f"{k}={v}" for k, v in fields.items())
    line = f"[H1_MEM] t={time.monotonic() - _T0:9.1f}s stage={stage} {extra} {body}".rstrip()
    print(line, flush=True)
    with open(MEM_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


def stage_begin(name):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    log(f"begin:{name}")


def stage_end(name):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    log(f"end:{name}")


def free_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class TextEncoderShim:
    """Stands in for the freed umT5 so pipe.to() and the config reads keep working. Calling it is an error."""

    def __init__(self, config, dtype):
        self.config = config
        self.dtype = dtype

    def to(self, *args, **kwargs):
        return self

    def __call__(self, *args, **kwargs):
        raise RuntimeError("H1_SEQ: umT5 is freed; a prompt must be in the cache before generation")


def prompt_key(prompt, negative_prompt, do_classifier_free_guidance, num_videos_per_prompt, max_sequence_length, device, dtype):
    """The same tuple LongCatVideoAvatarPipeline.encode_prompt stores in _prompt_cache_key (pipeline lines 318-326)."""
    return (
        tuple([prompt] if isinstance(prompt, str) else prompt),
        None if negative_prompt is None else tuple(
            [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
        ),
        do_classifier_free_guidance,
        num_videos_per_prompt,
        max_sequence_length,
        str(device),
        dtype,
    )


def allocated_bytes():
    return torch.cuda.memory_allocated() if torch.cuda.is_available() else 0


def _describe(obj):
    if isinstance(obj, types.FrameType):
        return f"frame {obj.f_code.co_name} at line {obj.f_lineno}"
    if isinstance(obj, dict):
        return f"dict with keys {sorted(map(str, obj.keys()))[:8]}"
    return type(obj).__name__


def release_text_encoder(pipe, base_bytes, tolerance_gib=0.25):
    """Swap umT5 out of the pipeline for the shim and release every reference to it. The caller must already have set its
    own variable to None. Raises if the module is still alive (printing its referrers), or if the GPU allocation has not
    returned to base_bytes plus the prompt cache (tolerance_gib)."""
    module = pipe.text_encoder
    ref = weakref.ref(module)
    # every parameter and buffer too: an autograd graph that saved a weight keeps the tensor alive even after the module is gone
    tensor_refs = [weakref.ref(t) for t in list(module.parameters()) + list(module.buffers())]
    pipe.text_encoder = TextEncoderShim(module.config, dtype=module.dtype)
    del module
    free_cuda()
    alive = ref()
    alive_tensors = [r() for r in tensor_refs if r() is not None]
    if alive is not None or alive_tensors:
        target = alive if alive is not None else alive_tensors[0]
        holders = [_describe(o) for o in gc.get_referrers(target) if o is not globals()]
        log("text_release_FAILED", module_alive=alive is not None, tensors_alive=len(alive_tensors), holders=len(holders))
        for h in holders:
            print(f"[H1_MEM] referrer: {h}", flush=True)
        del alive, alive_tensors, target
        raise RuntimeError("H1_SEQ: umT5 is still referenced after release (referrers printed above)")
    del alive
    grew = (allocated_bytes() - base_bytes) / GIB
    log("text_released", grew_since_base_GiB=f"{grew:.3f}")
    if grew > tolerance_gib:
        raise RuntimeError(f"H1_SEQ: GPU allocation is {grew:.3f} GiB above the base after release (limit {tolerance_gib})")
    return {"text_freed": True, "dit": None, "dit_on_host": False}


def move_dit_and_loras(dit, device):
    """DiT parameters and LoRA objects together. The LoRA objects live in dit.lora_dict, which nn.Module.to does not reach.
    This does not call dit.disable_all_loras(), which would remove the forward hooks."""
    dit.to(device)
    for network in getattr(dit, "lora_dict", {}).values():
        for lora in network.loras:
            lora.to(device)


def install_prompt_guard(pipe, state):
    """After umT5 is freed, encode_prompt may only return the cached result for the same key."""
    original = pipe.encode_prompt

    def guarded(prompt, negative_prompt=None, do_classifier_free_guidance=True, num_videos_per_prompt=1,
                max_sequence_length=512, device=None, dtype=None):
        if state["text_freed"]:
            key = prompt_key(prompt, negative_prompt, do_classifier_free_guidance, num_videos_per_prompt,
                             max_sequence_length, device, dtype)
            cached = getattr(pipe, "_prompt_cache_key", None)
            if key != cached:
                raise RuntimeError(f"H1_SEQ prompt cache miss: requested {key!r}, cached {cached!r}")
        state["denoise_calls"] = state.get("denoise_calls", 0) + 1
        stage_begin(f"denoise_{state['denoise_calls']}")
        if state.get("dit") is not None and state.get("dit_on_host"):
            move_dit_and_loras(state["dit"], state["device"])
            state["dit_on_host"] = False
        return original(prompt, negative_prompt=negative_prompt,
                        do_classifier_free_guidance=do_classifier_free_guidance,
                        num_videos_per_prompt=num_videos_per_prompt, max_sequence_length=max_sequence_length,
                        device=device, dtype=dtype)

    pipe.encode_prompt = guarded


def install_stage_hooks(pipe, dit, state, device):
    """Per-forward memory log on the DiT, and the decode stage (with the optional DiT offload before decode)."""
    state["dit"] = dit
    state["device"] = device
    forward = dit.forward
    counter = {"n": 0}

    def logged_forward(*args, **kwargs):
        out = forward(*args, **kwargs)
        counter["n"] += 1
        log("denoise_fwd", n=counter["n"])
        return out

    dit.forward = logged_forward

    decode = pipe._decode_video_latents
    offload = os.environ.get("H1_DECODE_OFFLOAD", "0") == "1"

    def logged_decode(latents):
        stage_begin("decode")
        if offload:
            move_dit_and_loras(dit, "cpu")
            state["dit_on_host"] = True
            free_cuda()
        out = decode(latents)
        stage_end("decode")
        return out

    pipe._decode_video_latents = logged_decode
