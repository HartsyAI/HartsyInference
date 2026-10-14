"""Records upstream IndexTTS-2.5 greedy codes (num_beams 3 and 1, do_sample=False) for the C# beam test.

Runs the unmodified index-tts `IndexTTS2` (infer_v2_5.py) on CPU and captures the semantic codes the GPT returns
for each line, with the reference clip given. The C# test feeds its own pipeline the same clip and text with
NumBeams=3 and sampling off, and compares the codes.

    PYTHONHASHSEED=0 idx25-venv/bin/python tools/parity/indextts25_beam_reference.py <index-tts checkout> \
        <checkpoint dir with config.yaml and hf_cache/> <reference.wav> \
        tests/HartsyInference.Audio.Tests/Fixtures/indextts25_beam_reference.json

The checkpoint dir is IndexTeam/IndexTTS-2.5 plus its config.yaml and the aux models upstream puts under hf_cache/
(w2v-bert-2.0, campplus_cn_common.bin, bigvgan). WeTextProcessing is Linux-only upstream; the wetext branch is used.
"""
import hashlib
import json
import os
import platform
import sys
import types

LINES = ["Yes.", "Okay, see you."]


def install_whisper_stub():
    # indextts/utils/tokenizer.py subclasses whisper.tokenizer.Tokenizer only for encode/decode forwarding.
    whisper = types.ModuleType("whisper")
    tok = types.ModuleType("whisper.tokenizer")

    class Tokenizer:
        def __init__(self, encoding, num_languages=99, language=None, task=None):
            self.encoding = encoding

        def encode(self, text, **kwargs):
            return self.encoding.encode(text, **kwargs)

        def decode(self, ids, **kwargs):
            return self.encoding.decode(ids, **kwargs)

    tok.Tokenizer = Tokenizer
    whisper.tokenizer = tok
    sys.modules["whisper"] = whisper
    sys.modules["whisper.tokenizer"] = tok


def main():
    repo, ckpt, ref_wav, out_path = sys.argv[1:5]
    sys.path.insert(0, repo)
    install_whisper_stub()
    import torch
    from indextts.infer_v2_5 import IndexTTS2

    platform.system = lambda: "Darwin"  # TextNormalizer.load(): wetext branch
    tts = IndexTTS2(cfg_path=os.path.join(ckpt, "config.yaml"), model_dir=ckpt, device="cpu", use_qwen_emo=False)
    captured = []
    original = tts.gpt.inference_speech

    def capture(*args, **kwargs):
        codes, latent = original(*args, **kwargs)
        captured.append(codes[0].tolist())
        return codes, latent

    tts.gpt.inference_speech = capture
    cases = []
    for text in LINES:
        for beams in (3, 1):
            captured.clear()
            torch.manual_seed(0)
            tts.infer(spk_audio_prompt=ref_wav, text=text, output_path=None, lang="en", num_beams=beams, do_sample=False)
            cases.append({"text": text, "lang": "en", "num_beams": beams, "codes": captured[0]})
            print(text, beams, len(captured[0]), captured[0])
    with open(ref_wav, "rb") as f:
        ref_sha = hashlib.sha256(f.read()).hexdigest()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"do_sample": False, "reference_wav_sha256": ref_sha, "cases": cases}, f, indent=1)
        f.write("\n")


if __name__ == "__main__":
    main()
