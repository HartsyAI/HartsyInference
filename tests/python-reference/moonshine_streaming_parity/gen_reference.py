"""Generates Moonshine-Streaming-Tiny parity fixtures (*.bin, gitignored) checked by
MoonshineStreamingParityTests.cs, using the real `transformers` (>=5.14) MoonshineStreamingForConditionalGeneration
class + the real UsefulSensors/moonshine-streaming-tiny checkpoint. Requires:
pip install torch transformers>=5.14 safetensors numpy huggingface_hub.
"""
import numpy as np
import torch
from transformers import MoonshineStreamingForConditionalGeneration

REPO = "UsefulSensors/moonshine-streaming-tiny"
OUT_DIR = "/home/hartsy/Desktop/HartsyInference/tests/python-reference/moonshine_streaming_parity"

torch.manual_seed(0)

model = MoonshineStreamingForConditionalGeneration.from_pretrained(REPO, torch_dtype=torch.float32)
model.eval()

FRAME_LEN = 80
T0 = 100  # frames -> 8000 raw samples (0.5s)
audio = (torch.rand(1, T0 * FRAME_LEN) * 2 - 1) * 0.1  # small-magnitude fake waveform, deterministic seed

attention_mask = torch.ones_like(audio)  # all-ones: forces the sliding-window mask path (see
# modeling_moonshine_streaming.py's MoonshineStreamingEncoder.forward — when attention_mask is None,
# per_layer_attention_mask is never built and every layer gets attention_mask=None (full unrestricted
# attention instead of the trained sliding window) — a reference-code quirk, not the intended default.

with torch.no_grad():
    enc_out = model.model.encoder(audio, attention_mask=attention_mask)
    encoder_hidden = enc_out.last_hidden_state  # [1, T, 320]

    prompt = torch.tensor([[1, 5, 10, 20]], dtype=torch.long)  # BOS + 3 arbitrary tokens
    out = model(input_values=audio, attention_mask=attention_mask, decoder_input_ids=prompt)
    logits_last = out.logits[0, -1, :]  # [vocab]

print("encoder_hidden", encoder_hidden.shape)
print("logits_last", logits_last.shape)

audio.numpy().astype(np.float32).tofile(f"{OUT_DIR}/audio.bin")
encoder_hidden.numpy().astype(np.float32).tofile(f"{OUT_DIR}/encoder_hidden_ref.bin")
prompt.numpy().astype(np.int32).tofile(f"{OUT_DIR}/prompt_tokens.bin")
logits_last.numpy().astype(np.float32).tofile(f"{OUT_DIR}/logits_last_ref.bin")

with open(f"{OUT_DIR}/meta.txt", "w") as f:
    f.write(f"T0={T0}\nencoder_T={encoder_hidden.shape[1]}\nvocab={logits_last.shape[0]}\nprompt_len={prompt.shape[1]}\n")

print("saved.")
