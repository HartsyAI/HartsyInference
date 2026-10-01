"""HF transformers reference for Whisper's log-mel front end and greedy tokens.

Read by tests/HartsyInference.Audio.Tests/Parity/WhisperLogMelParityTests.cs and WhisperTokenParityTests.cs. The C#
tests build the clips (16 kHz mono, raw little-endian float32) and write them to <dir>/clips with a clips.tsv
manifest, so both sides read identical samples. This script writes:

  features  <dir>/hf/{clip}.mel{80,128}.f32       WhisperFeatureExtractor()(clip): its default path, torch.stft when
                                                  torch is installed. [n_mels, 3000] row-major float32.
            <dir>/hf/{clip}.mel{80,128}.np.f32    the extractor's numpy path on the same padded window: the
                                                  reference's own noise floor.
  tokens    <dir>/hf/{org}_{name}/{clip}.{notimestamps,timestamps}.json
                                                  a greedy decode of WhisperForConditionalGeneration on those
                                                  features with the ENGINE's decoding rule, not generate(): the
                                                  engine's prompt, its suppressed ids (SOT through <|transcribe|>,
                                                  no-speech, no-timestamps) masked at every step, argmax, stop on
                                                  end-of-text, at most 224 new tokens. generate() adds suppress lists
                                                  and timestamp rules the engine does not apply. Every step records
                                                  its top-1/top-2 logit margin and the runner-up's id.

Requires numpy, torch and transformers (run with numpy 2.5.2, torch 2.13.0, transformers 5.15.0). CPU only and offline;
the weights come from local directories (config.json, model.safetensors, vocab.json, added_tokens.json):

  CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 python whisper_reference.py features --dir DIR
  CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 python whisper_reference.py tokens --dir DIR \\
      --model openai/whisper-tiny=/mnt/model-storage/Models/audio/stt/openai--whisper-tiny [--model ...]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import transformers
from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration

SAMPLE_RATE = 16_000
WINDOW = 30 * SAMPLE_RATE
MAX_NEW_TOKENS = 224  # WhisperOptions.MaxNewTokens
MULTILINGUAL_VOCAB = 51865  # OpenAI's rule, as WhisperTokenizer.MultilingualVocabSize


def read_clips(directory):
    clips = []
    with open(os.path.join(directory, "clips", "clips.tsv"), encoding="utf-8") as manifest:
        for line in manifest:
            name, samples = line.rstrip("\n").split("\t")
            audio = np.fromfile(os.path.join(directory, "clips", name + ".f32"), dtype="<f4")
            if audio.size != int(samples):
                sys.exit(f"{name}: {audio.size} samples on disk, manifest says {samples}")
            clips.append((name, audio))
    return clips


def window(audio):
    """WhisperFeatureExtractor.__call__'s pad/truncate: zero-pad (right) or cut to 30 s before the STFT."""
    padded = np.zeros(WINDOW, dtype=np.float32)
    count = min(audio.size, WINDOW)
    padded[:count] = audio[:count]
    return padded


def features(clip, extractor):
    """The default path, checked against the torch path on the padded window so the numpy path sees the same input."""
    default = extractor(clip, sampling_rate=SAMPLE_RATE, return_tensors="np")["input_features"][0]
    padded = window(clip)
    torch_path = extractor._torch_extract_fbank_features(padded[None, :], "cpu")[0]
    if not np.array_equal(default, torch_path):
        sys.exit("WhisperFeatureExtractor's padding differs from window(); the numpy twin would see other samples")
    numpy_path = extractor._np_extract_fbank_features(padded[None, :], "cpu")[0]
    return default.astype(np.float32), numpy_path.astype(np.float32)


def run_features(args):
    out_dir = os.path.join(args.dir, "hf")
    os.makedirs(out_dir, exist_ok=True)
    for n_mels in (80, 128):
        extractor = WhisperFeatureExtractor(feature_size=n_mels)
        for name, audio in read_clips(args.dir):
            default, numpy_path = features(audio, extractor)
            if default.shape != (n_mels, 3000):
                sys.exit(f"{name}: features {default.shape}, expected ({n_mels}, 3000)")
            default.astype("<f4").tofile(os.path.join(out_dir, f"{name}.mel{n_mels}.f32"))
            numpy_path.astype("<f4").tofile(os.path.join(out_dir, f"{name}.mel{n_mels}.np.f32"))
            floor = float(np.max(np.abs(default - numpy_path)))
            print(f"mel{n_mels} {name}: {audio.size} samples, torch vs numpy max abs {floor:.3e}")
    print(f"transformers {transformers.__version__}, torch {torch.__version__}, numpy {np.__version__}")


def special_ids(model_dir):
    with open(os.path.join(model_dir, "vocab.json"), encoding="utf-8") as f:
        ids = json.load(f)
    with open(os.path.join(model_dir, "added_tokens.json"), encoding="utf-8") as f:
        ids.update(json.load(f))
    no_speech = ids.get("<|nospeech|>", ids.get("<|nocaptions|>"))
    return {
        "eot": ids["<|endoftext|>"],
        "sot": ids["<|startoftranscript|>"],
        "en": ids["<|en|>"],
        "transcribe": ids["<|transcribe|>"],
        "no_speech": no_speech,
        "no_timestamps": ids["<|notimestamps|>"],
    }


def greedy(model, encoded, prompt, suppressed, eot):
    """WhisperPipeline.GreedyDecode: the prompt in one pass, then one token at a time on the KV cache."""
    decoder = model.model.decoder
    mask = torch.zeros(model.config.vocab_size, dtype=torch.bool)
    mask[suppressed] = True
    tokens, margins, runners_up = [], [], []
    cache = None
    step_ids = torch.tensor([prompt])
    for _ in range(MAX_NEW_TOKENS + 1):
        out = decoder(input_ids=step_ids, encoder_hidden_states=encoded, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        logits = model.proj_out(out.last_hidden_state[:, -1])[0].masked_fill(mask, float("-inf"))
        top = torch.topk(logits, 2)
        token = int(top.indices[0])
        if len(tokens) == MAX_NEW_TOKENS and token != eot:
            break
        margins.append(float(top.values[0] - top.values[1]))
        runners_up.append(int(top.indices[1]))
        if token == eot:
            break
        tokens.append(token)
        step_ids = torch.tensor([[token]])
    return tokens, margins, runners_up


def run_tokens(args):
    torch.set_num_threads(args.threads)
    clips = read_clips(args.dir)
    for spec in args.model:
        repo, model_dir = spec.split("=", 1)
        model = WhisperForConditionalGeneration.from_pretrained(model_dir, dtype=torch.float32).eval()
        ids = special_ids(model_dir)
        multilingual = model.config.vocab_size >= MULTILINGUAL_VOCAB
        n_mels = model.config.num_mel_bins
        extractor = WhisperFeatureExtractor(feature_size=n_mels)
        suppressed = sorted(set(range(ids["sot"], ids["transcribe"] + 1)) | {ids["no_speech"], ids["no_timestamps"]})
        out_dir = os.path.join(args.dir, "hf", repo.replace("/", "_"))
        os.makedirs(out_dir, exist_ok=True)
        for name, audio in clips:
            if args.clips and name not in args.clips:
                continue
            default, _ = features(audio, extractor)
            with torch.no_grad():
                encoded = model.model.encoder(torch.from_numpy(default)[None]).last_hidden_state
                for timestamps in (False, True):
                    prompt = [ids["sot"]] + ([ids["en"], ids["transcribe"]] if multilingual else [])
                    if not timestamps:
                        prompt.append(ids["no_timestamps"])
                    start = time.time()
                    tokens, margins, runners_up = greedy(model, encoded, prompt, suppressed, ids["eot"])
                    mode = "timestamps" if timestamps else "notimestamps"
                    record = {
                        "model": repo, "clip": name, "timestamps": timestamps, "prompt": prompt,
                        "suppressed": suppressed, "eot": ids["eot"], "tokens": tokens, "margins": margins,
                        "runners_up": runners_up, "n_mels": n_mels, "transformers": transformers.__version__,
                        "torch": torch.__version__,
                    }
                    with open(os.path.join(out_dir, f"{name}.{mode}.json"), "w", encoding="utf-8") as f:
                        json.dump(record, f)
                    smallest = min(margins) if margins else float("nan")
                    print(f"{repo} {name} {mode}: {len(tokens)} tokens in {time.time() - start:.1f}s, "
                          f"smallest margin {smallest:.4f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    feat = sub.add_parser("features")
    feat.add_argument("--dir", required=True)
    tok = sub.add_parser("tokens")
    tok.add_argument("--dir", required=True)
    tok.add_argument("--model", action="append", required=True, help="repo id=local directory")
    tok.add_argument("--clips", nargs="*", help="only these clips")
    tok.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if torch.cuda.is_available():
        sys.exit("run CPU-only: set CUDA_VISIBLE_DEVICES=\"\"")
    run_features(args) if args.command == "features" else run_tokens(args)


if __name__ == "__main__":
    main()
