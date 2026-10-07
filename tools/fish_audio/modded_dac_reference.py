"""Writes a tiny random ModifiedDAC checkpoint (decoder side, in the released key layout) plus the audio the official
fish-speech implementation decodes from fixed codes, for ModifiedDacDecoderParityTests.

Usage: python modded_dac_reference.py <fish-speech checkout> <output dir>
Needs torch, safetensors, einops, loguru and the `descript-audio-codec` package (installed --no-deps). The `audiotools`
dependency only supplies a base class, so it is stubbed. Rope tables are rebuilt in float32 (upstream uses bfloat16).

With a third argument (path to the real codec.pth) it instead decodes pseudo-random codes through the real S2 codec and
writes <output dir>/s2_codec_reference.safetensors holding the codes and the audio.
"""
import site, sys, types, json
from pathlib import Path

import torch, torch.nn as nn
from safetensors.torch import save_file

repo, out = Path(sys.argv[1]), Path(sys.argv[2])
sp = [p for p in site.getsitepackages() if (Path(p) / "dac").exists()][0]
pkg = types.ModuleType("dac"); pkg.__path__ = [str(Path(sp) / "dac")]; sys.modules["dac"] = pkg
for sub in ("dac.nn", "dac.model"):
    m = types.ModuleType(sub); m.__path__ = [str(Path(sp) / sub.replace(".", "/"))]; sys.modules[sub] = m
base = types.ModuleType("dac.model.base")
class CodecMixin:
    def get_delay(self): return 0
base.CodecMixin = CodecMixin; sys.modules["dac.model.base"] = base
at = types.ModuleType("audiotools"); at.AudioSignal = object
ml = types.ModuleType("audiotools.ml"); ml.BaseModel = nn.Module; at.ml = ml
sys.modules["audiotools"] = at; sys.modules["audiotools.ml"] = ml
sys.path.insert(0, str(repo))
import fish_speech.models.dac.modded_dac as md
import fish_speech.models.dac.rvq as rvq


def fp32_rope(module, head_dim, base_theta):
    module.freqs_cis = md.precompute_freqs_cis(4096, head_dim, base_theta, dtype=torch.float32)


def build(latent, cb_dim, n_res, sem_size, res_size, dec_dim, dec_rates, layers, heads, head_dim, inter, window, real=False):
    cfg = lambda **kw: md.ModelArgs(block_size=2048, n_layer=layers, n_head=heads, dim=latent, intermediate_size=inter,
                                    n_local_heads=-1, head_dim=head_dim, rope_base=10000, norm_eps=1e-5, **kw)
    mk = lambda: md.WindowLimitedTransformer(causal=True, window_size=window, input_dim=latent, config=cfg(channels_first=True))
    q = rvq.DownsampleResidualVectorQuantize(
        input_dim=latent, n_codebooks=n_res, codebook_size=res_size, codebook_dim=cb_dim, quantizer_dropout=0.0,
        downsample_factor=[2, 2], pre_module=mk(), post_module=mk(), semantic_codebook_size=sem_size)
    enc_dim = latent // 16
    if real:   # configs/modded_dac_vq.yaml
        import functools
        general = functools.partial(md.ModelArgs, block_size=8192, n_local_heads=-1, head_dim=64, rope_base=10000,
                                    norm_eps=1e-5, dropout_rate=0.1, attn_dropout_rate=0.1, channels_first=True)
        return md.DAC(encoder_dim=64, encoder_rates=[2, 4, 8, 8], decoder_dim=dec_dim, decoder_rates=dec_rates,
                      quantizer=q, sample_rate=44100, causal=True, encoder_transformer_layers=[0, 0, 0, 4],
                      decoder_transformer_layers=[4, 0, 0, 0], transformer_general_config=general)
    import functools
    general = functools.partial(md.ModelArgs, block_size=8192, n_local_heads=-1, head_dim=64, rope_base=10000,
                                norm_eps=1e-5, dropout_rate=0.0, attn_dropout_rate=0.0, channels_first=True)
    return md.DAC(encoder_dim=enc_dim, encoder_rates=[2, 2, 2, 2], decoder_dim=dec_dim, decoder_rates=dec_rates,
                  quantizer=q, sample_rate=44100, causal=True, encoder_transformer_layers=[0, 0, 0, 2],
                  transformer_general_config=general)


if len(sys.argv) > 3:   # real S2 codec
    model = build(1024, 8, 9, 4096, 1024, 1536, [8, 8, 4, 2], 8, 16, 64, 3072, 128, real=True)
    sd = torch.load(sys.argv[3], map_location="cpu", weights_only=True)
    print(model.load_state_dict(sd, strict=False))
    model.eval()
    fp32_rope(model.quantizer.post_module, 64, 10000)
    g = torch.Generator().manual_seed(3)
    T = 24
    codes = torch.stack([torch.randint(0, 4096, (T,), generator=g)] + [torch.randint(0, 1024, (T,), generator=g) for _ in range(9)])[None]
    with torch.no_grad():
        z = model.quantizer.decode(codes.clone())
        audio = model.decoder(z)
    ref = {"codes": codes[0].float().contiguous(), "audio": audio[0, 0].contiguous()}
    if len(sys.argv) > 4:   # also encode a reference clip (44.1 kHz mono 16-bit WAV)
        import wave, numpy as np
        wf = wave.open(sys.argv[4]); pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768
        clip = torch.from_numpy(pcm)[None]
        with torch.no_grad():
            enc_codes, _ = model.encode(clip)
        ref["enc_audio"] = clip[0].contiguous(); ref["enc_codes"] = enc_codes[0].float().contiguous()
    save_file(ref, str(out / "s2_codec_reference.safetensors"))
    print("wrote real reference", audio.shape)
    sys.exit(0)

torch.manual_seed(11)
T = 12
model = build(64, 8, 3, 32, 16, 32, [2, 2, 2, 2], 2, 4, 16, 96, 8).eval()
with torch.no_grad():
    for name, p in model.named_parameters():
        if name.endswith("original0") or name.endswith("weight_g"):
            p.copy_(torch.rand_like(p) * 0.5 + 0.3)
        elif name.endswith("alpha"):
            p.copy_(torch.rand_like(p) * 0.3 + 0.1)
        elif name.endswith("gamma"):
            p.copy_(torch.randn_like(p) * 0.5)
        elif "norm" in name and name.endswith("weight"):
            p.copy_(1.0 + 0.2 * torch.randn_like(p))
        else:
            p.copy_(torch.randn_like(p) * 0.05)
fp32_rope(model.quantizer.post_module, 16, 10000)
fp32_rope(model.quantizer.pre_module, 16, 10000)
fp32_rope(model.encoder.block[4].block[5], 64, 10000)
codes = torch.stack([torch.randint(0, 32, (T,))] + [torch.randint(0, 16, (T,)) for _ in range(3)])[None]
taps = {}
with torch.no_grad():
    q = model.quantizer
    c = codes.clone()
    zs = q.semantic_quantizer.from_codes(c[:, :1])[0] + q.quantizer.from_codes(c[:, 1:])[0]
    taps["sum"] = zs[0].T.contiguous()                     # [T, D] like the host layout
    zp = q.post_module(zs)
    taps["post"] = zp[0].T.contiguous()
    zu = zp
    for i, blk in enumerate(q.upsample):
        zu = blk[0](zu); taps[f"up{i}_conv"] = zu[0].contiguous()
        zu = blk[1](zu); taps[f"up{i}"] = zu[0].contiguous()
    x = zu
    dec = model.decoder.model
    x = dec[0](x); taps["stem"] = x[0].contiguous()
    for i in range(len(dec) - 4):
        blk = dec[i + 1].block
        x = blk[1](blk[0](x)); taps[f"stage{i}_up"] = x[0].contiguous()
        for j in range(2, 5): x = blk[j](x)
        taps[f"stage{i}"] = x[0].contiguous()
    audio = model.decoder(zu)
# encode side: reference clip -> codes through the official encoder + quantizer
g = torch.Generator().manual_seed(5)
audio_in = (torch.randn(1, 1, 640, generator=g) * 0.3).clamp(-1, 1)
with torch.no_grad():
    enc = model.encoder.block
    x = enc[0](audio_in)
    for i in range(1, 5):
        x = enc[i](x); taps[f"enc{i-1}"] = x[0].contiguous()
    x = enc[6](enc[5](x)); taps["latent"] = x[0].contiguous()
    q = model.quantizer
    zd = q.downsample(x)
    zpre = q.pre_module(zd)
    taps["pre"] = zpre[0].T.contiguous()
    enc_codes = q(x).codes[0]
ck = {k: v.contiguous().float() for k, v in model.state_dict().items() if k.startswith(("quantizer.", "decoder.", "encoder."))
      and not k.endswith(("causal_mask", "freqs_cis"))}
ck["ref.enc_audio"] = audio_in[0, 0].contiguous()
ck["ref.enc_codes"] = enc_codes.float().contiguous()
ck["ref.audio"] = audio[0, 0].contiguous()
for k, v in taps.items(): ck[f"tap.{k}"] = v.float().contiguous()
ck["ref.codes"] = codes[0].float().contiguous()
out.mkdir(parents=True, exist_ok=True)
save_file(ck, str(out / "modded_dac_tiny.safetensors"))
print("wrote", out, audio.shape)
