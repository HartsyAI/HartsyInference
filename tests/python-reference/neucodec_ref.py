#!/usr/bin/env python3
"""
NeuCodec (neuphonic/neucodec, pip `neucodec==0.0.6`) ENCODE reference dumper.

Loads the official model, encodes the fixed 16 kHz jfk clip, and dumps the integer
FSQ code indices so the C# NeuCodecEncoder can be diffed against ground truth.

Outputs:
  /tmp/neucodec_ref_codes.npy       int32 [F]   the 50 Hz FSQ code indices
  /tmp/neucodec_ref_roundtrip.wav   24 kHz mono decode(encode(x))
  (prints shape, min/max, first 32 codes)

Notes:
  * jfk.wav is already 16 kHz mono, so `_prepare_audio` does NOT resample it; we
    feed a [B=1, 1, T] tensor (the clean documented input shape).
  * The encode path is: waveform*2^15 -> SeamlessM4T Kaldi log-fbank (80 mel, povey
    window 400/hop 160/fft 512, preemph 0.97, dc-offset removed, mel_floor 1.19e-7),
    per-mel-bin zero-mean-unit-var (ddof=1), stride-2 stack -> 160-dim; Wav2Vec2BertModel
    (facebook/w2v-bert-2.0, relative_key attn) hidden_states[16]; SemanticEncoder adapter;
    concat [semantic, acoustic] -> fc_prior(2048->2048) -> ResidualFSQ (project_in 2048->8,
    levels [4]*8, DOUBLE bound then round).  DO NOT RUN HERE — coordinator runs on GPU.
"""
import os
import numpy as np
import torch
import torchaudio

JFK = os.path.expanduser("~/.cache/hartsyinference/models/test-clips/jfk.wav")
OUT_CODES = "/tmp/neucodec_ref_codes.npy"
OUT_WAV = "/tmp/neucodec_ref_roundtrip.wav"


def main():
    from neucodec import NeuCodec

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = NeuCodec.from_pretrained("neuphonic/neucodec")
    model = model.eval().to(device)

    # Load the fixed clip as [B=1, 1, T]; jfk is already 16 kHz mono so no resample happens.
    y, sr = torchaudio.load(JFK)          # [C, T]
    assert sr == 16000, f"expected 16 kHz jfk clip, got {sr}"
    wav = y[0]                             # [T]  (mono)
    x = wav.view(1, 1, -1).to(device)      # [B, 1, T]

    with torch.no_grad():
        fsq_codes = model.encode_code(x)   # [B, 1, F]
        recon = model.decode_code(fsq_codes)  # [B, 1, T24]

    codes = fsq_codes.squeeze().to(torch.int64).cpu().numpy().astype(np.int32)  # [F]
    np.save(OUT_CODES, codes)

    audio = recon.squeeze().float().cpu().numpy()
    torchaudio.save(OUT_WAV, torch.from_numpy(audio).view(1, -1), 24000)

    print(f"codes shape: {codes.shape}")
    print(f"codes min/max: {int(codes.min())} / {int(codes.max())}")
    print(f"first 32 codes: {codes[:32].tolist()}")
    print(f"wrote {OUT_CODES} and {OUT_WAV}")


if __name__ == "__main__":
    main()
