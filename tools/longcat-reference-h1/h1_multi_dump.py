"""Step-0 dump of the multi-speaker reference run (H1_MULTI_DUMP_DIR). Off unless the variable is set; reads only.

Installed at run time by h1_multi_runner.py: it wraps the avatar pipeline's generate_ai2v and the reference-map function, and puts
forward hooks on the DiT for its first forward. No reference file is changed. With the distilled path (the only one accepted here)
that forward is the single pass of step 0, the same call the engine's diagnostics.longcatProbeDump records.

Writes, under H1_MULTI_DUMP_DIR, float32 .npy files:

  conditioning/  text_embeddings_step00 [1,S,C], text_mask_step00 [1,S], audio_embedding_step00 [tracks,frames,window*blocks*channels],
                 input_latent_step00 [1,C,T,H,W]: the engine's file names and element order, so this folder can be given to the engine as
                 diagnostics.longcatConditioningSubstitute. Do not add other files to it.
  probe/         call00_probe_<name>.npy, the engine's probe names: x_embedder, t_embedder, velocity, block{i}.out for every block,
                 and for the blocks in H1_MULTI_BLOCKS (default 0,1,24,47) block{i}.self_attn, ffn, ref_attn_map [regions, noisy
                 tokens], query_positions, key_positions and audio_cross_attn. Also audio_tokens [frames, tracks*tokens, width] (the
                 audio projection of every track, one after another per frame) and ref_target_masks [regions, height, width].
  meta.json      shapes and settings.
"""

import json
import os

import numpy as np
import torch

_STATE = {"done": False, "armed": False, "block": -1, "named": set(), "rope": {}, "maps": set(), "handles": []}


def _directory():
    return os.environ.get("H1_MULTI_DUMP_DIR")


def named_blocks(depth):
    text = os.environ.get("H1_MULTI_BLOCKS", "0,1,24,47")
    return {int(part) for part in text.split(",") if part.strip() and int(part) < depth}


def _array(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().float().numpy()  # to the host first: no F32 copy on the card in the middle of a forward
    return np.ascontiguousarray(np.asarray(value, dtype=np.float32))


def _save(folder, name, value):
    directory = os.path.join(_directory(), folder)
    os.makedirs(directory, exist_ok=True)
    file = f"call00_probe_{name}.npy" if folder == "probe" else f"{name}.npy"
    np.save(os.path.join(directory, file), _array(value))


def _hook(handler):
    """A forward (pre-)hook that does nothing once the dump is disarmed and never replaces the module's input or output."""
    def hook(*args):
        if _STATE["armed"]:
            handler(*args)

    return hook


def _arm(dit):
    depth = len(dit.blocks)
    named = named_blocks(depth)
    _STATE.update(armed=True, named=named, rope={}, maps=set(), block=-1)
    handles = _STATE["handles"]

    def conditioning(_module, _args, kwargs):
        if not _STATE["armed"]:
            return
        required = ("hidden_states", "timestep", "encoder_hidden_states", "encoder_attention_mask", "audio_embs", "ref_target_masks")
        missing = [name for name in required if kwargs.get(name) is None]
        if missing:
            raise RuntimeError(f"H1 multi dump: the DiT call has no {missing}; is this the multi-speaker ai2v call?")
        hidden = kwargs["hidden_states"]
        audio = kwargs["audio_embs"]  # [tracks, frames, window, blocks, channels]
        text = kwargs["encoder_hidden_states"]  # [B, 1, S, C]
        if hidden.shape[0] != 1 or text.shape[0] != 1 or audio.dim() != 5:
            raise RuntimeError(f"H1 multi dump: unexpected shapes {tuple(hidden.shape)}, {tuple(text.shape)}, {tuple(audio.shape)}")
        _save("conditioning", "input_latent_step00", hidden)
        _save("conditioning", "text_embeddings_step00", text.reshape(1, text.shape[-2], text.shape[-1]))
        _save("conditioning", "text_mask_step00", kwargs["encoder_attention_mask"].reshape(1, -1))
        _save("conditioning", "audio_embedding_step00", audio.reshape(audio.shape[0], audio.shape[1], -1))
        _save("probe", "ref_target_masks", kwargs["ref_target_masks"])
        meta = {"latent": list(hidden.shape), "audio": list(audio.shape), "regions": int(kwargs["ref_target_masks"].shape[0]),
                "timestep": _array(kwargs["timestep"]).reshape(-1).tolist(), "num_cond_latents": kwargs.get("num_cond_latents"),
                "blocks": sorted(named), "depth": depth, "torch": torch.__version__, "dit_dtype": str(next(dit.parameters()).dtype)}
        with open(os.path.join(_directory(), "meta.json"), "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2)

    handles.append(dit.register_forward_pre_hook(conditioning, with_kwargs=True))

    def finished(_module, _inputs, output):
        _save("probe", "velocity", output)
        _STATE.update(armed=False, done=True)
        print(f"[H1_MULTI] step-0 dump written to {_directory()}", flush=True)

    handles.append(dit.register_forward_hook(_hook(finished)))
    handles.append(dit.x_embedder.register_forward_hook(_hook(lambda _m, _i, out: _save("probe", "x_embedder", out[0]))))
    handles.append(dit.t_embedder.register_forward_hook(_hook(lambda _m, _i, out: _save("probe", "t_embedder", out))))

    def audio_tokens(_module, _inputs, output):  # [tracks, frames, tokens, width]
        tracks, frames, tokens, width = output.shape
        _save("probe", "audio_tokens", output.permute(1, 0, 2, 3).reshape(frames, tracks * tokens, width))

    handles.append(dit.audio_proj.register_forward_hook(_hook(audio_tokens)))

    for index, block in enumerate(dit.blocks):
        def enter(_m, _i, index=index):
            _STATE["block"] = index

        def leave(_m, _i, output, index=index):
            output = output[0] if isinstance(output, tuple) else output
            _save("probe", f"block{index}.out", output[0])

        handles.append(block.register_forward_pre_hook(_hook(enter)))
        handles.append(block.register_forward_hook(_hook(leave)))
        if index not in named:
            continue

        def stage(name, pick, index=index):
            return _hook(lambda _m, _i, output: _save("probe", f"block{index}.{name}", pick(output)))

        handles.append(block.attn.register_forward_hook(stage("self_attn", lambda out: out[0][0])))
        handles.append(block.ffn.register_forward_hook(stage("ffn", lambda out: out[0])))
        handles.append(block.audio_cross_attn.register_forward_hook(stage("audio_cross_attn", lambda out: out[1][0])))

        def positions(_module, inputs, index=index):  # called for the queries first, then for the audio keys
            seen = _STATE["rope"].get(index, 0)
            if seen > 1:
                raise RuntimeError(f"H1 multi dump: block {index} rotated more than twice")
            _save("probe", f"block{index}.{'query_positions' if seen == 0 else 'key_positions'}", inputs[1])
            _STATE["rope"][index] = seen + 1

        handles.append(block.audio_cross_attn.rope_1d.register_forward_pre_hook(_hook(positions)))


def _disarm():
    for handle in _STATE["handles"]:
        handle.remove()
    _STATE["handles"].clear()
    _STATE["armed"] = False


def install(pipeline_class, attention_module):
    """Wraps generate_ai2v and the reference-map function. Call once, before the demo runs."""
    original_map = attention_module.get_attn_map_with_target

    def recorded_map(*args, **kwargs):
        result = original_map(*args, **kwargs)
        if _STATE["armed"] and _STATE["block"] in _STATE["named"]:
            _save("probe", f"block{_STATE['block']}.ref_attn_map", result)
            _STATE["maps"].add(_STATE["block"])
        return result

    attention_module.get_attn_map_with_target = recorded_map
    original = pipeline_class.generate_ai2v

    def generate_ai2v(self, *args, **kwargs):
        if not _directory() or _STATE["done"]:
            return original(self, *args, **kwargs)
        guidance = (kwargs.get("text_guidance_scale", 4.0), kwargs.get("audio_guidance_scale", 4.0))
        if not kwargs.get("use_distill") or max(guidance) > 1.0:
            raise NotImplementedError("H1 multi dump covers the distilled path only: with guidance the first forward is the negative pass.")
        _arm(self.dit)
        try:
            result = original(self, *args, **kwargs)
        finally:
            _disarm()
        if not _STATE["done"]:
            raise RuntimeError("H1 multi dump: the DiT ran no forward, nothing was written")
        missing = sorted(i for i in _STATE["named"] if _STATE["rope"].get(i) != 2 or i not in _STATE["maps"])
        if missing:
            raise RuntimeError(f"H1 multi dump: blocks {missing} gave no reference map or no query and key positions")
        return result

    pipeline_class.generate_ai2v = generate_ai2v
