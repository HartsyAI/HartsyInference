"""Dumps upstream CosyVoice 2 zero-shot prompt conditioning for the engine's CosyVoice2ZeroShotParityTests.

Runs CosyVoiceFrontEnd.frontend_zero_shot (FunAudioLLM/CosyVoice cosyvoice/cli/frontend.py) with resample_rate 24000 on
a reference clip, CPU: the speech tokens (speech_tokenizer_v2.onnx on whisper.log_mel_spectrogram), the prompt mel
(matcha mel_spectrogram), both cut to min(frames / 2, tokens), and the CAM++ embedding (campplus.onnx). Also writes the
16 kHz and 24 kHz signals load_wav hands those extractors.

Writes into <out_dir>: ref16.f32, ref24.f32, prompt_tokens.i32, prompt_feat_tm.f32 ([frames, 80]), embedding.f32
([192]), meta.json.

    python tools/parity/cosyvoice2_zeroshot_dump.py <CosyVoice2-0.5B dir> <reference.wav> <out_dir>

Run with the CosyVoice repo (and its third_party/Matcha-TTS) on PYTHONPATH.
"""
import json
import os
import sys

import numpy as np
import torch

from cosyvoice.cli.frontend import CosyVoiceFrontEnd
from cosyvoice.utils.file_utils import load_wav
from hyperpyyaml import load_hyperpyyaml


def main() -> None:
    model_dir, ref, out = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(model_dir, "cosyvoice2.yaml")) as f:
        configs = load_hyperpyyaml(f, overrides={"qwen_pretrain_path": os.path.join(model_dir, "CosyVoice-BlankEN")})
    frontend = CosyVoiceFrontEnd(configs["get_tokenizer"], configs["feat_extractor"],
                                 os.path.join(model_dir, "campplus.onnx"),
                                 os.path.join(model_dir, "speech_tokenizer_v2.onnx"),
                                 os.path.join(model_dir, "spk2info.pt"), configs["allowed_special"])
    with torch.inference_mode():
        mi = frontend.frontend_zero_shot("Yes.", "The quick brown fox jumps over the lazy dog near the river bank.",
                                         ref, 24000, "")

    def f32(name, t):
        np.asarray(t, dtype=np.float32).tofile(os.path.join(out, name + ".f32"))

    f32("ref16", load_wav(ref, 16000)[0])
    f32("ref24", load_wav(ref, 24000)[0])
    np.asarray(mi["flow_prompt_speech_token"][0], dtype=np.int32).tofile(os.path.join(out, "prompt_tokens.i32"))
    f32("prompt_feat_tm", mi["prompt_speech_feat"][0])
    f32("embedding", mi["flow_embedding"][0])
    meta = dict(tokens=int(mi["flow_prompt_speech_token"].shape[1]), feat=list(mi["prompt_speech_feat"].shape[1:]),
                llm_tokens_equal_flow=bool(torch.equal(mi["llm_prompt_speech_token"], mi["flow_prompt_speech_token"])))
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
