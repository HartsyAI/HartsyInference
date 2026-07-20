"""Generates ZipVoice parity fixtures (*.bin, gitignored) checked by ZipVoiceParityTests.cs, using the REAL
k2-fsa/ZipVoice source (`zipvoice.models.modules.zipformer.TTSZipformer`) + the REAL k2-fsa/ZipVoice checkpoint.

We import `TTSZipformer` (+ its `scaling.py` dependency) standalone rather than the full `zipvoice` package /
`zipvoice.models.zipvoice`, because `zipvoice.utils.common` (pulled in transitively by the top-level package)
imports `torch.utils.tensorboard`, which needs the `tensorboard` package that is NOT installed in the
disk-constrained venv used here. `zipformer.py` itself only needs plain torch + its own `scaling.py` (which
soft-imports `k2` in a try/except and falls back to pure PyTorch), so it imports cleanly standalone.

For the same reason, `prepare_avg_tokens_durations` / `get_tokens_index` (normally in `zipvoice.utils.common`)
are reproduced verbatim below (trivial, torch-optional logic) rather than imported.

Requires: /tmp/benchvenv (torch + safetensors + numpy already installed, no heavy deps needed beyond that).
"""
import sys
sys.path.insert(0, "/tmp/zipvoice_src")

import numpy as np
import torch
from safetensors.torch import load_file

from zipvoice.models.modules.zipformer import TTSZipformer

CKPT = ("/home/hartsy/.cache/huggingface/hub/models--k2-fsa--ZipVoice/snapshots/"
        "4ed45fb6e7e9527b780bef9e097a04bf13fe4e6b/zipvoice/model.safetensors")
OUT_DIR = "/home/hartsy/Desktop/HartsyInference/tests/python-reference/zipvoice_parity"

state_dict = load_file(CKPT)


def load_backbone(model: torch.nn.Module, prefix: str) -> None:
    sub = {k[len(prefix) + 1:]: v for k, v in state_dict.items() if k.startswith(prefix + ".")}
    missing, unexpected = model.load_state_dict(sub, strict=False)
    assert not missing, f"missing keys for {prefix}: {missing}"
    assert not unexpected, f"unexpected keys for {prefix}: {unexpected}"
    model.eval()


# ---------------------------------------------------------------------------
# 1. fm_decoder (flow-matching velocity backbone): in_dim=300 (xt|text_cond|speech_cond, 100 each), out_dim=100,
#    encoder_dim=512, 5 U-Net-shaped stages, time-conditioned.
# ---------------------------------------------------------------------------
fm_decoder = TTSZipformer(
    in_dim=300,
    out_dim=100,
    downsampling_factor=(1, 2, 4, 2, 1),
    num_encoder_layers=(2, 2, 4, 4, 4),
    cnn_module_kernel=(31, 15, 7, 15, 31),
    encoder_dim=512,
    query_head_dim=32,
    pos_head_dim=4,
    value_head_dim=12,
    num_heads=4,
    feedforward_dim=1536,
    pos_dim=48,
    use_time_embed=True,
    time_embed_dim=192,
    use_guidance_scale_embed=False,
)
load_backbone(fm_decoder, "fm_decoder")

torch.manual_seed(0)
T_FM = 17
x_fm = torch.randn(1, T_FM, 300)  # (batch, seq_len, in_dim) -- NOTE: forward()'s docstring confirms
# batch-first; internally it does x.permute(1, 0, 2) to go seq-first before in_proj.
t_fm = torch.tensor([0.3])

with torch.no_grad():
    out_fm = fm_decoder(x_fm, t=t_fm)  # -> (1, T_FM, 100)

assert out_fm.shape == (1, T_FM, 100), out_fm.shape
assert torch.isfinite(out_fm).all(), "fm_decoder output has NaN/Inf"
print("fm_decoder out", out_fm.shape, "mean", out_fm.mean().item(), "std", out_fm.std().item())

x_fm.squeeze(0).contiguous().numpy().astype(np.float32).tofile(f"{OUT_DIR}/fm_x.bin")
out_fm.squeeze(0).contiguous().numpy().astype(np.float32).tofile(f"{OUT_DIR}/fm_out_ref.bin")

# ---------------------------------------------------------------------------
# 2. text_encoder: in_dim=192, out_dim=100, encoder_dim=192, single stage (no downsampling), no time embed.
# ---------------------------------------------------------------------------
text_encoder = TTSZipformer(
    in_dim=192,
    out_dim=100,
    downsampling_factor=(1,),
    num_encoder_layers=(4,),
    cnn_module_kernel=(9,),
    encoder_dim=192,
    query_head_dim=32,
    pos_head_dim=4,
    value_head_dim=12,
    num_heads=4,
    feedforward_dim=512,
    pos_dim=48,
    use_time_embed=False,
    use_guidance_scale_embed=False,
)
load_backbone(text_encoder, "text_encoder")

T_TE = 9
x_te = torch.randn(1, T_TE, 192)  # continues the same RNG stream after x_fm/t_fm above (still seed-0 derived,
# deterministic given the fixed script order -- no separate manual_seed call needed).

with torch.no_grad():
    out_te = text_encoder(x_te, t=None)  # -> (1, T_TE, 100)

assert out_te.shape == (1, T_TE, 100), out_te.shape
assert torch.isfinite(out_te).all(), "text_encoder output has NaN/Inf"
print("text_encoder out", out_te.shape, "mean", out_te.mean().item(), "std", out_te.std().item())

x_te.squeeze(0).contiguous().numpy().astype(np.float32).tofile(f"{OUT_DIR}/te_x.bin")
out_te.squeeze(0).contiguous().numpy().astype(np.float32).tofile(f"{OUT_DIR}/te_out_ref.bin")

with open(f"{OUT_DIR}/meta.txt", "w") as f:
    f.write(f"fm_T={T_FM}\nfm_in_dim=300\nfm_out_dim=100\nfm_t=0.3\n")
    f.write(f"te_T={T_TE}\nte_in_dim=192\nte_out_dim=100\n")

# ---------------------------------------------------------------------------
# 3. "Average upsampling" duration/frame-to-token-index logic (zipvoice/utils/common.py, reproduced verbatim
#    -- pure Python/torch-optional, no need to import the heavy `zipvoice.utils.common` module).
# ---------------------------------------------------------------------------
def prepare_avg_tokens_durations(features_lens, tokens_lens):
    tokens_durations = []
    for i in range(len(features_lens)):
        utt_duration = features_lens[i]
        avg_token_duration = utt_duration // tokens_lens[i]
        tokens_durations.append([avg_token_duration] * tokens_lens[i])
    return tokens_durations


def get_tokens_index(durations, num_frames):
    durations = [x + [num_frames - sum(x)] for x in durations]
    batch_size = len(durations)
    ans = torch.zeros(batch_size, num_frames, dtype=torch.int64)
    for b in range(batch_size):
        this_dur = durations[b]
        cur_frame = 0
        for i, d in enumerate(this_dur):
            ans[b, cur_frame: cur_frame + d] = i
            cur_frame += d
        assert cur_frame == num_frames, (cur_frame, num_frames)
    return ans


TOKENS_LEN = 9
FEATURES_LEN = 41
durations = prepare_avg_tokens_durations([FEATURES_LEN], [TOKENS_LEN])
idx = get_tokens_index(durations, FEATURES_LEN)[0]  # (FEATURES_LEN,)
print("frame->token index", idx.tolist())

idx.numpy().astype(np.int32).tofile(f"{OUT_DIR}/frame_to_token_index_ref.bin")
with open(f"{OUT_DIR}/duration_meta.txt", "w") as f:
    f.write(f"tokens_len={TOKENS_LEN}\nfeatures_len={FEATURES_LEN}\n")

print("saved.")
