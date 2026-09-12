"""Dumps YuE2 parity fixtures from the official `yue2_infer` wheel (m-a-p/YuE2-3B).

Deps: the pinned wheel in its own venv (torch 2.10, transformers 4.57.6, tiktoken, soundfile).
    python -m venv ~/yue2-ref/.venv && ~/yue2-ref/.venv/bin/pip install yue2_infer-0.1.5-py3-none-any.whl

Checkpoints: the ORIGINAL m-a-p weights, not the Comfy repack — gate A0 proves the repack is a
bit-identical re-layout, so the originals are the cleaner oracle. Every stage dumps its own inputs
alongside its outputs so the C# side shares actual tensors rather than only matching seeds.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

from yue2.protocol import (EOD, ABC_START, ABC_END, MUSIC_START, MUSIC_END, CODEC_OFFSET,
                           INSTRUCTIONS, GenerationConfig, Sampling, SongRequest,
                           token_prefixes, negative_prefix, chunk_ranges)
from yue2.tokenization_yue2 import YuE2TextTokenizer

STYLE = "uplifting synth pop, female vocal, bright, 120 bpm"
LYRICS = "[verse]\nmorning light across the wire\n[chorus]\nwe are awake tonight\n"
SEED = 831001


def dump_protocol(out, model_dir):
    """Gate A1/A2: tokenizer ids and the exact prefixes each cot mode builds."""
    tok = YuE2TextTokenizer(Path(model_dir) / "qwen.tiktoken")
    data = {"style": STYLE, "lyrics": LYRICS, "seed": SEED,
            "specials": {"EOD": EOD, "ABC_START": ABC_START, "ABC_END": ABC_END,
                         "MUSIC_START": MUSIC_START, "MUSIC_END": MUSIC_END,
                         "CODEC_OFFSET": CODEC_OFFSET},
            "instructions": INSTRUCTIONS, "modes": {}}
    sample_abc = "X:1\nL:1/8\nM:4/4\nK:C\n|:C2 E2 G2 c2|d2 c2 B2 A2:|\n"
    for cot in ("off", "melody", "full"):
        req = SongRequest(style=STYLE, lyrics=LYRICS, cot=cot, seed=SEED)
        entry = {
            "prompt_text": req.text(),
            "prompt_ids": tok.encode(req.text()),
            "instruction_ids": tok.encode(INSTRUCTIONS[cot]),
            "guidance": req.guidance,
            "prefix_planning": token_prefixes(req, tok),
        }
        if cot != "off":
            abc_ids = tok.encode(sample_abc)
            entry["abc_text"] = sample_abc
            entry["abc_ids"] = abc_ids
            entry["prefix_with_abc"] = token_prefixes(req, tok, abc_ids)
            entry["negative_with_abc"] = negative_prefix(req, tok, abc_ids)
        else:
            entry["negative"] = negative_prefix(req, tok)
        data["modes"][cot] = entry
    data["chunk_ranges_1000f_300p"] = chunk_ranges(1000, 300)
    (out / "protocol.json").write_text(json.dumps(data, indent=1))
    print(f"  protocol.json: prompt_ids={len(data['modes']['off']['prompt_ids'])}")
    return tok


def dump_ar(out, model, tok, device):
    """Gate A3/A5: greedy AR logits (no RNG in the loop) and the per-layer KV a NAR chunk consumes."""
    from yue2.modeling_yue2 import StaticKVCache
    req = SongRequest(style=STYLE, lyrics=LYRICS, cot="off", seed=SEED)
    prefix = token_prefixes(req, tok)
    steps = 8
    cache = StaticKVCache(num_layers=model.config.num_hidden_layers, batch_size=1,
                          num_kv_heads=model.config.num_key_value_heads,
                          max_seq_len=len(prefix) + steps + 1,
                          head_dim=model.config.head_dim, dtype=torch.bfloat16, device=device)
    with torch.inference_mode():
        outputs = model(torch.tensor([prefix], device=device), past_key_values=cache,
                        use_cache=True, logits_to_keep=1)
        logits = [outputs.logits[:, -1, :].float().cpu().numpy()]
        tokens = []
        current = outputs.past_key_values
        for _ in range(steps):
            # Greedy over the codec span only, mirroring distribution()'s hard mask at temperature 0.
            row = logits[-1][0].copy()
            row[:CODEC_OFFSET] = -np.inf
            row[CODEC_OFFSET + 32768:] = -np.inf
            nxt = int(row.argmax())
            tokens.append(nxt)
            step = model(torch.tensor([[nxt]], device=device), past_key_values=current,
                         use_cache=True, logits_to_keep=1)
            logits.append(step.logits[:, -1, :].float().cpu().numpy())
            current = step.past_key_values
    np.savez(out / "ar_greedy.npz", prefix=np.asarray(prefix, np.int32),
             logits=np.concatenate(logits, 0).astype(np.float32), tokens=np.asarray(tokens, np.int32))
    print(f"  ar_greedy.npz: prefix={len(prefix)} steps={steps} first_tokens={tokens[:4]}")

    # Gate A5: the post-RoPE K/V the NAR stack attends over, from nar.CachedNAR's own prefill.
    from yue2.nar import CachedNAR, Chunk
    codec = [int(t) - CODEC_OFFSET for t in tokens]
    noise = torch.randn((len(codec), 64), dtype=torch.float32, device="cpu",
                        generator=torch.Generator(device="cpu").manual_seed(SEED))
    chunk = Chunk(prefix + [v + CODEC_OFFSET for v in codec] + [MUSIC_END], noise)
    engine = CachedNAR(model, chunk)
    keys = np.stack([k.float().cpu().numpy() for k, _ in engine.cache])
    values = np.stack([v.float().cpu().numpy() for _, v in engine.cache])
    np.savez(out / "ar_kv.npz", ar_tokens=np.asarray(chunk.ar_tokens, np.int32),
             keys=keys.astype(np.float32), values=values.astype(np.float32))
    print(f"  ar_kv.npz: keys={keys.shape} values={values.shape}")

    # Gate B1/B2: one velocity evaluation, then the full 32-step midpoint solve on the same noise.
    t0 = 1.0
    raw = torch.logit(torch.tensor(t0, dtype=torch.float64)).clamp(-20, 20).item()
    state = noise.to(device=engine.device, dtype=engine.dtype)
    velocity = engine.velocity(state, raw).float().cpu().numpy()
    latents = engine.solve(steps=GenerationConfig().ode_steps).numpy()
    engine.close()
    np.savez(out / "nar.npz", noise=noise.numpy().astype(np.float32), t=np.float32(t0), raw_t=np.float32(raw),
             velocity=velocity.astype(np.float32), latents=latents.astype(np.float32),
             ode_steps=np.int32(GenerationConfig().ode_steps))
    print(f"  nar.npz: velocity={velocity.shape} latents={latents.shape}")
    return latents


def dump_vae(out, vae_dir, latents, device):
    """Gate C1: latents -> 48 kHz stereo, untiled so the comparison has no halo seam in it."""
    from yue2.modeling_vae import YuE2VAE
    model = YuE2VAE.from_pretrained(vae_dir, decoder_only=True, device=device, local_files_only=True)
    z = torch.as_tensor(latents, dtype=torch.float32).T.unsqueeze(0).to(device)
    with torch.inference_mode():
        audio = model.decode(z).cpu()[0].float().clamp(-1, 1).T.contiguous().numpy()
    np.savez(out / "vae.npz", latents=latents.astype(np.float32), audio=audio.astype(np.float32),
             sample_rate=np.int32(48000))
    print(f"  vae.npz: audio={audio.shape} ({len(audio)/48000:.2f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="../../Models/audio/music/yue2/reference/YuE2-3B")
    ap.add_argument("--vae", default="../../Models/audio/music/yue2/reference/YuE2-Vae")
    ap.add_argument("--out", default="yue2_reference")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--stage", default="all", choices=("all", "protocol", "ar", "vae", "song"))
    ap.add_argument("--seconds", type=float, default=30.0)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    out = (root / args.out); out.mkdir(parents=True, exist_ok=True)
    model_dir, vae_dir = (root / args.model).resolve(), (root / args.vae).resolve()
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    tok = dump_protocol(out, model_dir)
    if args.stage == "protocol":
        return 0

    if args.stage == "song":
        from yue2.pipeline import YuE2Pipeline
        pipe = YuE2Pipeline(model_dir, vae_dir, device=str(device), verify_hashes=False)
        cfg = GenerationConfig(semantic=Sampling(max_tokens=int(args.seconds * 25)))
        pipe.generation_config = cfg
        result = pipe(style=STYLE, lyrics=LYRICS, cot="off", seed=SEED)
        result.save_artifacts(out / "song")
        print(f"  song: {len(result.audio)/48000:.2f}s -> {out/'song'}")
        return 0

    from yue2.modeling_yue2 import YuE2ForCausalLM
    model = YuE2ForCausalLM.from_pretrained(model_dir, local_files_only=True,
                                            dtype=torch.bfloat16, low_cpu_mem_usage=True).eval().to(device)
    latents = dump_ar(out, model, tok, device)
    model.to("cpu"); torch.cuda.empty_cache()
    if args.stage in ("all", "vae"):
        dump_vae(out, vae_dir, latents, device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
