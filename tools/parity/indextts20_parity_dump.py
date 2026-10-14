"""IndexTTS-2.0 stage-by-stage reference dump for IndexTts2V20PythonParityTests.

Runs the unmodified index-tts `IndexTTS2` (infer_v2.py) on CPU in F32 for one line and records every intermediate of
`infer_generator` through forward hooks and wrappers, without changing what upstream computes:

  case1_b3.safetensors  (upstream defaults: 3-beam sampling, seed 0)
    audio22k, audio16k            librosa load (22.05 kHz) and the torchaudio 22.05 kHz -> 16 kHz speaker signal
    emo_audio16k                  librosa load at 16 kHz (the default emotion clip is the speaker clip)
    input_features                SeamlessM4T stacked features of audio16k                  [T, 160]
    spk_cond_emb, emo_cond_emb    normalized w2v-bert hidden_states[17]                     [T, 1024]
    style                         CAM++ style                                               [192]
    ref_mel                       S2Mel reference mel                                       [80, Tref]
    codes_ref, s_ref              semantic_codec.quantize(spk_cond_emb)                     [T], [T, 1024]
    prompt_condition              length_regulator(S_ref, Tref)                             [Tref, 512]
    spk_latent                    Conformer+Perceiver speaker latents                       [32, 1280]
    emovec_base                   gpt.get_emovec(spk_cond_emb)                              [1280]
    emovec                        merge_emovec(spk, emo, alpha 1) as inference_speech got it [1280]
    seg0_text_ids, seg0_codes     segment 0 text ids (no start/stop) and its codes
    seg0_latent                   second GPT pass after gpt_layer                           [n, 1024]
    seg0_vq2emb                   semantic_codec.quantizer.vq2emb(codes)                    [n, 1024]
    seg0_cond                     length_regulator(S_infer, n * 1.72)                        [m, 512]
    est{1,13,25}_{x,prompt_x,mu,style,t,out}   the CFM estimator's stacked inputs and output at those Euler steps
    seg0_vc_target, seg0_wav      the generated mel (prompt removed) and BigVGAN's waveform
  greedy1.safetensors
    emovec, codes_greedy_rp1, codes_greedy_rp10   inference_speech greedy (1 beam), repetition penalty 1 and 10,
                                  400-token cap, codes cut at the stop token
  decode1.safetensors
    codes, step{s}_block{l}, step{s}_lnf, step{s}_logits   the first TEACHER_STEPS greedy rp1 codes teacher-forced
                                  through GPT2InferenceModel as HF generate() steps it (see indextts25_gpt_decode_dump)
  cfm1.safetensors
    cat, z, out                   the CFM solve's condition [T, 512], its initial noise [80, T] and its output [80, T]

    PYTHONHASHSEED=0 idx25-venv/bin/python tools/parity/indextts20_parity_dump.py <index-tts checkout> \
        <checkpoint dir> <semantic codec model.safetensors> <reference.wav> <out dir>

<checkpoint dir> holds IndexTeam/IndexTTS-2 (bpe.model, gpt.pth, s2mel.pth, wav2vec2bert_stats.pt, feat1/feat2.pt and
its config.yaml) plus hf_cache/ with w2v-bert-2.0/, campplus_cn_common.bin and bigvgan/ (config.json and
bigvgan_generator.pt of nvidia/bigvgan_v2_22khz_80band_256x).
"""
import hashlib
import os
import platform
import sys

TEXT = "Hello there, how are you today?"
TEACHER_STEPS = 24


def main():
    repo, ckpt, codec_path, ref_wav, out_dir = sys.argv[1:6]
    sys.path.insert(0, repo)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from indextts25_gpt_decode_dump import install_whisper_stub, teacher_force
    install_whisper_stub()
    import torch
    import torchaudio
    from safetensors.torch import save_file
    from indextts.infer_v2 import IndexTTS2

    torch.set_num_threads(16)
    torch.backends.mkldnn.allow_tf32 = False
    platform.system = lambda: "Darwin"  # TextNormalizer.load(): wetext branch (WeTextProcessing is Linux-only)
    aux = {"w2v_bert": os.path.join(ckpt, "hf_cache", "w2v-bert-2.0"), "semantic_codec": codec_path,
           "campplus": os.path.join(ckpt, "hf_cache", "campplus_cn_common.bin"), "bigvgan": os.path.join(ckpt, "hf_cache", "bigvgan")}
    tts = IndexTTS2(cfg_path=os.path.join(ckpt, "config.yaml"), model_dir=ckpt, device="cpu", use_qwen_emo=False, aux_paths=aux)
    gpt, s2mel = tts.gpt, tts.s2mel.models
    d = {}
    f32 = lambda x: torch.as_tensor(x).detach().to(torch.float32).contiguous().clone()

    load_audio = tts._load_and_cut_audio

    def capture_audio(path, seconds, verbose=False, sr=None):
        audio, rate = load_audio(path, seconds, verbose, sr)
        d["emo_audio16k" if sr else "audio22k"] = f32(audio[0])
        return audio, rate
    tts._load_and_cut_audio = capture_audio

    extract = tts.extract_features

    def capture_features(audio, **kw):
        out = extract(audio, **kw)
        d.setdefault("input_features", f32(out["input_features"][0]))
        return out
    tts.extract_features = capture_features

    get_emb = tts.get_emb

    def capture_emb(features, mask):
        out = get_emb(features, mask)
        d["emo_cond_emb" if "spk_cond_emb" in d else "spk_cond_emb"] = f32(out[0])
        return out
    tts.get_emb = capture_emb

    quantize = tts.semantic_codec.quantize

    def capture_quantize(x):
        codes, s_ref = quantize(x)
        d["codes_ref"], d["s_ref"] = f32(codes.reshape(-1)), f32(s_ref[0])
        return codes, s_ref
    tts.semantic_codec.quantize = capture_quantize

    mel_fn = tts.mel_fn

    def capture_mel(x):
        out = mel_fn(x)
        d["ref_mel"] = f32(out[0])
        return out
    tts.mel_fn = capture_mel

    tts.campplus_model.register_forward_hook(lambda m, i, o: d.__setitem__("style", f32(o.reshape(-1))))
    regulator_calls = []
    s2mel["length_regulator"].register_forward_hook(lambda m, i, o: regulator_calls.append(f32(o[0][0])))
    s2mel["gpt_layer"].register_forward_hook(lambda m, i, o: d.__setitem__("seg0_latent", f32(o[0])))
    vq2emb = tts.semantic_codec.quantizer.vq2emb

    def capture_vq2emb(codes, *a, **kw):
        out = vq2emb(codes, *a, **kw)
        d["seg0_vq2emb"], d["seg0_codes"] = f32(out[0].transpose(0, 1)), f32(codes.reshape(-1))
        return out
    tts.semantic_codec.quantizer.vq2emb = capture_vq2emb

    inference_speech = gpt.inference_speech
    seg = {}

    def capture_inference(*args, **kw):
        codes, latent = inference_speech(*args, **kw)
        seg.update(spk=args[0], text=args[1], emo=args[2], kw=kw)
        d["spk_latent"], d["emovec"] = f32(latent[0]), f32(kw["emo_vec"].reshape(-1))
        d["seg0_text_ids"] = f32(args[1][0])
        return codes, latent
    gpt.inference_speech = capture_inference

    est_step = [0]

    def capture_estimator(m, args, out):
        est_step[0] += 1
        s = est_step[0]
        if s in (1, 13, 25):
            x, prompt_x, _, t, style, mu = args[:6]
            d.update({f"est{s}_x": f32(x), f"est{s}_prompt_x": f32(prompt_x), f"est{s}_mu": f32(mu),
                      f"est{s}_style": f32(style), f"est{s}_t": f32(t.reshape(-1)[:1]), f"est{s}_out": f32(out)})
    s2mel["cfm"].estimator.register_forward_hook(capture_estimator)
    cfm = {}
    solve_euler = s2mel["cfm"].solve_euler

    def capture_solve(x, x_lens, prompt, mu, style, f0, t_span, inference_cfg_rate=0.5):
        cfm["cat"], cfm["z"] = f32(mu[0]), f32(x[0])
        out = solve_euler(x, x_lens, prompt, mu, style, f0, t_span, inference_cfg_rate)
        cfm["out"] = f32(out[0])
        return out
    s2mel["cfm"].solve_euler = capture_solve

    def capture_bigvgan(m, args, out):
        d["seg0_vc_target"], d["seg0_wav"] = f32(args[0][0]), f32(out.reshape(-1))
    tts.bigvgan.register_forward_hook(capture_bigvgan)

    torch.manual_seed(0)
    tts.infer(spk_audio_prompt=ref_wav, text=TEXT, output_path=None)
    d["audio16k"] = f32(torchaudio.transforms.Resample(22050, 16000)(d["audio22k"].unsqueeze(0))[0])
    d["prompt_condition"], d["seg0_cond"] = regulator_calls[0], regulator_calls[1]
    stop = tts.stop_mel_token
    with torch.no_grad():
        spk, emo = seg["spk"], seg["emo"]
        d["emovec_base"] = f32(gpt.get_emovec(spk, torch.tensor([spk.shape[-1]])).reshape(-1))
        greedy = {"emovec": d["emovec"]}
        for rp in (1, 10):
            kw = dict(seg["kw"], do_sample=False, num_beams=1, repetition_penalty=float(rp), max_generate_length=400)
            for k in ("top_p", "top_k", "temperature", "length_penalty"):
                kw.pop(k, None)
            codes, _ = inference_speech(spk, seg["text"], emo, **kw)
            codes = codes[0].tolist()
            greedy[f"codes_greedy_rp{rp}"] = f32(codes[:codes.index(stop)] if stop in codes else codes)
        # inference_speech's own prefix (speaker latents + emovec, speed_emb(1), speed_emb(0)), then the first
        # TEACHER_STEPS greedy rp1 codes replayed one cached step at a time.
        latent = gpt.get_conditioning(spk.transpose(1, 2), torch.tensor([spk.shape[-1]]))
        assert torch.equal(f32(latent[0]), d["spk_latent"]), "speaker latents differ from inference_speech's"
        speed = gpt.speed_emb(torch.tensor([1, 0]))
        conds = torch.cat((latent + seg["kw"]["emo_vec"].unsqueeze(1), speed.unsqueeze(0)), 1)
        input_ids, inputs_embeds, attention_mask = gpt.prepare_gpt_inputs(conds, seg["text"])
        gpt.inference_model.store_mel_emb(inputs_embeds)
        rp1 = [int(c) for c in greedy["codes_greedy_rp1"].tolist()][:TEACHER_STEPS]
        decode = {k: f32(v) for k, v in teacher_force(gpt, rp1, attention_mask, input_ids, torch).items()}
        decode["codes"] = f32(rp1)
    n = d["seg0_codes"].numel()
    os.makedirs(out_dir, exist_ok=True)
    with open(ref_wav, "rb") as f:
        meta = {"text": TEXT, "reference_wav_sha256": hashlib.sha256(f.read()).hexdigest()}
    save_file({k: v.contiguous() for k, v in d.items()}, os.path.join(out_dir, "case1_b3.safetensors"), metadata=meta)
    save_file(greedy, os.path.join(out_dir, "greedy1.safetensors"), metadata=meta)
    save_file(cfm, os.path.join(out_dir, "cfm1.safetensors"), metadata=meta)
    save_file(decode, os.path.join(out_dir, "decode1.safetensors"), metadata=meta)
    print(f"{n} codes; greedy rp1 {len(greedy['codes_greedy_rp1'])}, rp10 {len(greedy['codes_greedy_rp10'])} -> {out_dir}")


if __name__ == "__main__":
    main()
