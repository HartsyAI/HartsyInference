# AuK architecture (Tencent, MIT)

Sources: `tencent/AuK`, `tencent/AuK-Flash`, `github.com/Tencent-Hunyuan/AuK` (`src/auk/infer/infer_auk.py`, `model/cfm_edit.py`, `model/flux2_edit.py`, `model/modules.py`, `model/vae/bigvgan_flow_vae.py`).

- **Files:** `auk_base.safetensors` / `auk_flash.safetensors` (F32, 420 tensors: `transformer.*`, `layer_weights [36]`, `layer_scale [1]`), shared `vae.safetensors` (F32, 1137 tensors).
- **DiT:** dim 1536, 24 heads, 10 double-stream then 20 single-stream blocks, SwiGLU FFN (ff_mult 2), AdaLN-Zero, per-head QK RMSNorm, fused `to_qkv`. RoPE `inv_freq` is a checkpoint buffer (bf16-rounded) and must be loaded, not recomputed. One shared `audio_embed` (Linear 64→1536 + conv-pos) is applied to the ref and noisy latents. Layout `[text | ref | noisy]`; double blocks use per-stream RoPE positions, single blocks continuous positions.
- **Conditioning:** Qwen2.5-Omni-3B thinker only (36 layers, hidden 2048, 16/2 heads, 1D RoPE for text + audio-only input) plus its audio tower. Hidden states 1..36 (the last is post-final-norm) are layer-normed (no affine, eps 1e-5), softmax(`layer_weights`)-weighted, scaled by `layer_scale`, then `txt_proj` 2048→1536 + RMSNorm. The reference clip is used twice: as 16 kHz mel through the audio tower and as 24 kHz VAE latents.
- **Sampling:** flow matching, Euler t 0→1. Flash: grid `[0, 0.0761205, 0.2928932, 0.6173166, 1]`, cfg 0. Base: 32 steps, cfg 2.0 (`v_c + (v_c − v_u)·cfg`, uncond zeroes the text conditioning after `txt_norm` and the ref latent before `audio_embed`), sway −1.
- **VAE:** 24 kHz, hop 480, 64 latent channels; decoder causal with anti-aliased SnakeBeta AMP blocks, encoder symmetric and stochastic (variance-style normalization). The `flow.*` weights are training-only.
- **Licenses:** AuK MIT; Qwen2.5-Omni-3B Qwen Research License (not re-hosted).
- **Budget:** about 30 s of source plus target; sequential residency peaks near 9–10 GB in bf16.
