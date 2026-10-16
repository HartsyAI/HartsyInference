"""Writes a tiny random Fish Audio Drama 3 (fish_omni) checkpoint and what fish-speech's own modules compute on it, for
FishAudioDrama3ParityTests.

The checkpoint is written twice under the released key layout (model.language_model.* + lm_head.weight + audio_decoder*): once
with per-expert MoE tensors (drama3_tiny.safetensors, as the FP8 release stores them) and once with Hugging Face's fused
experts.gate_up_proj / experts.down_proj stacks (drama3_tiny_fused.safetensors). The reference runs FishOmniBackbone and
FishOmniAudioDecoder (fish-speech fish_omni at 5415669) in float32 on CPU: the prompt prefill, the tower's teacher-forced prefill,
then two decode steps with forced frames, recording the backbone hidden states and logits and every codebook head's logits.

Usage: python -I drama3_reference.py <fish-speech checkout> <output dir>   (needs torch, safetensors, loguru)
"""
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

repo, out = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(repo))

from fish_speech.models.text2semantic.fish_omni.backbone import FishOmniBackbone  # noqa: E402
from fish_speech.models.text2semantic.fish_omni.config import FishOmniConfig  # noqa: E402
from fish_speech.models.text2semantic.fish_omni.tower import (  # noqa: E402
    TOWER_CHECKPOINT_PREFIXES,
    FishOmniAudioDecoder,
)

CONFIG = {
    "model_type": "fish_omni",
    "speech_template_version": "reference-bracket-v1",
    "semantic_start_token_id": 40,
    "semantic_end_token_id": 47,
    "eos_token_id": 2,
    "text_config": {
        "model_type": "qwen3_5_moe_text",
        "vocab_size": 64,
        "hidden_size": 32,
        "num_hidden_layers": 4,
        "layer_types": ["linear_attention", "full_attention", "linear_attention", "full_attention"],
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 16,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.5},
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 256,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 8,
        "linear_value_head_dim": 8,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 16,
        "shared_expert_intermediate_size": 16,
        "norm_topk_prob": True,
        "mlp_layer_types": ["moe"] * 4,
        "tie_word_embeddings": False,
    },
    "audio_decoder_config": {
        "decoder_type": "codebook_mtp_shared",
        "dim": 32,
        "text_dim": 32,
        "intermediate_size": 48,
        "n_head": 2,
        "n_local_heads": 1,
        "head_dim": 16,
        "norm_eps": 1e-6,
        "rope_base": 10000.0,
        "rotary_percent": 0.5,
        "rotary_interleaved": True,
        "attention_output_gate": True,
        "layernorm_zero_centered_gamma": True,
        "use_moe": False,
        "mtp_use_swa": False,
        "attention_qkv_bias": False,
        "attention_o_bias": False,
        "dropout": 0.0,
        "vocab_size": 8,
        "num_codebooks": 4,
        "n_layer": 2,
        "shared_enrichment_layers": 2,
        "vq_backbone_codebooks": 1,
        "vq_feature_placement": "backbone_partial",
    },
}
# Rows are [token, code 0, ..., code 3]; audio rows carry 40 + code 0. The last row ends the prompt on text, as <|voice|> does.
PROMPT = [
    [3, 0, 0, 0, 0], [5, 0, 0, 0, 0], [41, 1, 2, 3, 4], [46, 6, 0, 7, 1], [40, 0, 5, 5, 2],
    [7, 0, 0, 0, 0], [8, 0, 0, 0, 0], [43, 3, 1, 0, 6], [9, 0, 0, 0, 0], [10, 0, 0, 0, 0], [12, 0, 0, 0, 0],
]
FRAME1 = [44, 4, 2, 6, 3]
FRAME2 = [42, 2, 7, 1, 0]


def randomize(module: torch.nn.Module) -> None:
    with torch.no_grad():
        for name, p in module.named_parameters():
            if name.endswith("A_log"):
                p.copy_(torch.rand_like(p) * 1.5 - 0.5)
            elif name.endswith("dt_bias"):
                p.copy_(torch.randn_like(p) * 0.5)
            elif name.endswith("linear_attn.norm.weight"):
                p.copy_(1.0 + torch.randn_like(p) * 0.2)
            elif "norm" in name:
                p.copy_(torch.randn_like(p) * 0.2)
            elif "embed" in name:
                p.copy_(torch.randn_like(p) * 0.5)
            else:
                p.copy_(torch.randn_like(p) * 0.15)


def backbone_checkpoint(backbone: FishOmniBackbone, fused: bool) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for name, tensor in backbone.state_dict().items():
        if name == "lm_head.weight":
            state[name] = tensor
            continue
        key = "model.language_model." + name[len("model."):]
        if key.endswith("mlp.experts.w13_weight"):
            prefix = key[: -len("w13_weight")]
            if fused:
                state[prefix + "gate_up_proj"] = tensor
            else:
                rows = tensor.shape[1] // 2
                for e in range(tensor.shape[0]):
                    state[f"{prefix}{e}.gate_proj.weight"] = tensor[e, :rows]
                    state[f"{prefix}{e}.up_proj.weight"] = tensor[e, rows:]
        elif key.endswith("mlp.experts.w2_weight"):
            prefix = key[: -len("w2_weight")]
            if fused:
                state[prefix + "down_proj"] = tensor
            else:
                for e in range(tensor.shape[0]):
                    state[f"{prefix}{e}.down_proj.weight"] = tensor[e]
        else:
            state[key] = tensor
    return state


def tower_checkpoint(tower: FishOmniAudioDecoder) -> dict[str, torch.Tensor]:
    inverse = {new: old for old, new in TOWER_CHECKPOINT_PREFIXES.items()}
    state = {}
    for name, tensor in tower.state_dict().items():
        prefix = next(new for new in inverse if name.startswith(new))
        state[inverse[prefix] + name[len(prefix):]] = tensor
    return state


@torch.no_grad()
def forced_step(tower, hidden, position, frame, main, codes):
    """decode_step with the main token and codes given: the head logits each residual layer produces."""
    positions = torch.tensor([position])
    c0 = (torch.tensor([main]) - tower.semantic_begin).clamp(0, tower.config.vocab_size - 1)
    semantic = tower.is_semantic(torch.tensor([main]))
    hidden = tower.run_shared(hidden, c0, semantic, positions, frame)
    logits, previous = [], c0
    for j, (layer, head) in enumerate(zip(tower.residual_layers, tower.codebook_heads)):
        hidden = layer(hidden, previous, semantic, positions)
        logits.append(head(hidden)[0])
        previous = torch.tensor([codes[j + 1]])
    return torch.stack(logits)


@torch.no_grad()
def main() -> None:
    config = FishOmniConfig.from_dict(CONFIG)
    torch.manual_seed(7)
    backbone = FishOmniBackbone(config.text, dtype=torch.float32).eval()
    tower = FishOmniAudioDecoder(config, dtype=torch.float32).eval()
    randomize(backbone)
    randomize(tower)

    frames = torch.tensor(PROMPT, dtype=torch.long)
    rows = len(frames)
    backbone.setup_caches(1, 64)
    tower.setup_caches(64)
    embeds = tower.embed_backbone(frames, backbone.embed(frames[:, 0]))
    hidden = backbone(inputs_embeds=embeds[None], input_pos=torch.arange(rows))[0]
    tower.prefill(hidden[: rows - 1], torch.arange(rows - 1), frames[: rows - 1], frames[1:rows])
    logits0 = backbone.compute_logits(hidden[-1:])[0]
    head1 = forced_step(tower, hidden[-1:], rows - 1, frames[-1:], FRAME1[0], FRAME1[1:])

    frame1 = torch.tensor([FRAME1])
    embeds1 = tower.embed_backbone(frame1, backbone.embed(frame1[:, 0]))
    hidden1 = backbone(inputs_embeds=embeds1[None], input_pos=torch.tensor([rows]))[0]
    logits1 = backbone.compute_logits(hidden1)[0]
    head2 = forced_step(tower, hidden1, rows, frame1, FRAME2[0], FRAME2[1:])

    out.mkdir(parents=True, exist_ok=True)
    towers = tower_checkpoint(tower)
    for fused, name in ((False, "drama3_tiny.safetensors"), (True, "drama3_tiny_fused.safetensors")):
        state = {**backbone_checkpoint(backbone, fused), **towers}
        save_file({k: v.contiguous() for k, v in state.items()}, str(out / name))
    expected = {
        "in.prompt": frames.to(torch.int32),
        "in.frame1": torch.tensor(FRAME1, dtype=torch.int32),
        "in.frame2": torch.tensor(FRAME2, dtype=torch.int32),
        "ref.hidden": hidden,
        "ref.logits0": logits0,
        "ref.head1": head1,
        "ref.hidden1": hidden1[0],
        "ref.logits1": logits1,
        "ref.head2": head2,
    }
    save_file({k: v.contiguous() for k, v in expected.items()}, str(out / "drama3_tiny_expected.safetensors"))
    (out / "config.json").write_text(json.dumps(CONFIG, indent=2) + "\n")
    print("hidden", tuple(hidden.shape), float(hidden.abs().max()), "head1", float(head1.abs().max()))


main()
