"""Teacher-forced IndexTTS-2.5 GPT decode dump: per-step logits and per-layer hidden states from upstream.

Runs the unmodified index-tts `IndexTTS2` (infer_v2_5.py) on CPU in F32 up to the GPT. For each line it records the
inputs `inference_speech` receives (CAM++ style, emotion vector, text ids, language id), lets upstream decode greedily
(num_beams 1, do_sample False, repetition penalty 10), then replays that code sequence through
`GPT2InferenceModel` exactly as HF `generate()` calls it (prefill over the fake input ids with the attention mask,
then one cached step per code, the mask growing by one each step) and records, for every step:

    step{s}_block{l}   output of GPT block l at the step's position      [1280]
    step{s}_lnf        after the backbone's ln_f                          [1280]
    step{s}_logits     mel-head logits before any logits processor        [8194]

plus the inputs (`style`, `spk_proj`, `emovec`, `text_ids`, `lang_id`, `prefix_embeds`, the `[pad][cond][text]` rows
upstream caches), the greedy `codes`, and the reference front end: `audio22k` (librosa load), `audio16k` (torchaudio
22.05 kHz → 16 kHz, the speaker path), `emo_audio16k` (librosa load at 16 kHz, the emotion path), `spk_cond_emb` and
`emo_cond_emb` (normalized w2v-bert `hidden_states[17]`). Ids are stored as F32 (exact below 2^24).

    PYTHONHASHSEED=0 idx25-venv/bin/python tools/parity/indextts25_gpt_decode_dump.py <index-tts checkout> \
        <checkpoint dir with config.yaml and hf_cache/> <reference.wav> <out dir>

Writes `<out dir>/indextts25_gpt_decode_l{i}.safetensors`, one per line. The checkpoint dir is IndexTeam/IndexTTS-2.5
plus its config.yaml and the aux models upstream puts under hf_cache/ (w2v-bert-2.0, campplus_cn_common.bin, bigvgan).
"""
import hashlib
import os
import platform
import sys
import types

LINES = ["Yes.", "Okay, see you."]


class _Stop(Exception):
    pass


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


def teacher_force(gpt, codes, attention_mask, input_ids, torch):
    """Replays `codes` through gpt.inference_model the way HF generate() does, hooking every block."""
    model = gpt.inference_model
    taps = {}
    hooks = [blk.register_forward_hook(lambda m, i, o, l=l: taps.__setitem__(l, o[0][0, -1].detach().clone()))
             for l, blk in enumerate(model.transformer.h)]
    hooks.append(model.transformer.ln_f.register_forward_hook(
        lambda m, i, o: taps.__setitem__("lnf", o[0, -1].detach().clone())))
    out = {}
    try:
        past = None
        ids = input_ids
        mask = attention_mask
        for s in range(len(codes)):
            if s > 0:
                ids = torch.cat([ids, torch.tensor([[codes[s - 1]]], dtype=ids.dtype)], dim=1)
                mask = torch.cat([mask, mask.new_ones((1, 1))], dim=1)
            inputs = model.prepare_inputs_for_generation(ids, past_key_values=past, attention_mask=mask, use_cache=True)
            res = model(**inputs, return_dict=True)
            past = res.past_key_values
            for l in range(len(model.transformer.h)):
                out[f"step{s}_block{l}"] = taps[l]
            out[f"step{s}_lnf"] = taps["lnf"]
            out[f"step{s}_logits"] = res.logits[0, -1].detach().clone()
    finally:
        for h in hooks:
            h.remove()
    return out


def main():
    repo, ckpt, ref_wav, out_dir = sys.argv[1:5]
    sys.path.insert(0, repo)
    install_whisper_stub()
    import torch
    import torchaudio
    from safetensors.torch import save_file
    from indextts.infer_v2_5 import IndexTTS2

    torch.backends.mkldnn.allow_tf32 = False
    platform.system = lambda: "Darwin"  # TextNormalizer.load(): wetext branch (WeTextProcessing is Linux-only)
    tts = IndexTTS2(cfg_path=os.path.join(ckpt, "config.yaml"), model_dir=ckpt, device="cpu", use_qwen_emo=False)
    gpt = tts.gpt
    original = gpt.inference_speech
    captured = {}

    def capture(*args, **kw):
        codes, latent = original(*args, **kw)
        captured.update(spk_cond_emb=args[0], emo_cond_emb=args[3],
                        style=kw["campplus_embedding"], emovec=kw["emo_vec"], text=args[1], lang=args[2],
                        codes=codes[0].tolist(), spk_proj=latent)
        raise _Stop()

    gpt.inference_speech = capture
    load_audio = tts._load_and_cut_audio
    clips = {}

    def capture_audio(path, seconds, verbose=False, sr=None):
        audio, rate = load_audio(path, seconds, verbose, sr)
        clips["emo_audio16k" if sr else "audio22k"] = audio[0].clone()
        return audio, rate

    tts._load_and_cut_audio = capture_audio
    os.makedirs(out_dir, exist_ok=True)
    with open(ref_wav, "rb") as f:
        ref_sha = hashlib.sha256(f.read()).hexdigest()
    for i, text in enumerate(LINES):
        captured.clear()
        torch.manual_seed(0)
        try:
            tts.infer(spk_audio_prompt=ref_wav, text=text, output_path=None, lang="en", num_beams=1, do_sample=False)
        except _Stop:
            pass
        codes = captured["codes"]
        with torch.no_grad():
            emo = captured["emovec"]
            conds = torch.cat((captured["spk_proj"] + emo.unsqueeze(1), torch.zeros(1, 2, captured["spk_proj"].size(2))), 1)
            input_ids, inputs_embeds, attention_mask = gpt.prepare_gpt_inputs(conds, captured["text"], captured["lang"])
            gpt.inference_model.store_mel_emb(inputs_embeds)
            dump = teacher_force(gpt, codes, attention_mask, input_ids, torch)
        f32 = lambda x: torch.as_tensor(x, dtype=torch.float32).contiguous()
        dump.update(style=f32(captured["style"]), spk_proj=f32(captured["spk_proj"]), emovec=f32(emo),
                    text_ids=f32(captured["text"][0]), lang_id=f32(captured["lang"]), prefix_embeds=f32(inputs_embeds),
                    attention_mask=f32(attention_mask), codes=f32(codes),
                    audio22k=f32(clips["audio22k"]), emo_audio16k=f32(clips["emo_audio16k"]),
                    audio16k=f32(torchaudio.transforms.Resample(22050, 16000)(clips["audio22k"].unsqueeze(0))[0]),
                    spk_cond_emb=f32(captured["spk_cond_emb"]), emo_cond_emb=f32(captured["emo_cond_emb"]))
        dump = {k: v.float().contiguous() for k, v in dump.items()}
        path = os.path.join(out_dir, f"indextts25_gpt_decode_l{i}.safetensors")
        save_file(dump, path, metadata={"text": text, "reference_wav_sha256": ref_sha})
        print(text, len(codes), codes, "->", path, flush=True)


if __name__ == "__main__":
    main()
