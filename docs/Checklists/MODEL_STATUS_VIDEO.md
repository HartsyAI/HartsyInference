# Video Models — status

Concise status for every video-generation (T2V / I2V) model. Build detail lives in
[PHASE_9_VIDEO.md](PHASE_9_VIDEO.md). Parity evidence lives in
[PARITY_VERIFICATION.md](PARITY_VERIFICATION.md). Legend: [MODEL_STATUS.md](MODEL_STATUS.md).

## Verified end-to-end (✅)

None yet. No video model has a real-weight, output-confirmed end-to-end run.

## Built, validation-pending (🔧)

All built end-to-end (transformer + VAE + pipeline + converter), structural tests pass; numeric parity
against a Python reference is pending for every one.

| Model | Notes |
|---|---|
| **Wan 2.2 TI2V-5B** (T2V / I2V) | umT5 entry + Wan2.2 3D causal VAE incl. encoder (RGB-input I2V works); TI2V VAE == decoder. |
| **LTX-Video** | DiT + base VAE decoder + pipeline; non-causal symmetric-replicate CausalConv3d mode added. |
| **LTX-2** (22B) | Dual-stream audio+video; Gemma 49-layer wiring; SwarmUI loader wired (blocked on engine NuGet republish). |
| **WanAnimate / WanS2V / WanVace** | Wan-lineage variants on the shared backbone. |
| **Lance (ByteDance) video** | Shared Lance backbone + Wan2.2 3D causal VAE; brought up the reusable 3D-video foundation. |
| **Kandinsky-5 video** | Built on the Kandinsky-5 backbone. |

## Not started (❌)

| Model | Notes |
|---|---|
| **Cosmos-Predict1 V2W (5B / 13B)** | NVIDIA AR video-continuation. FSQ tokenizer + AR transformer substrate (reusable for AR-token world models). `.pt` pickle weights, T5-11B encoder. |

## Notes

The reusable 3D-video foundation (CausalConv3d, streaming Wan VAE, frame encoders) was brought up by the
Lance and Wan builds and is shared across video + world models. The fastest path to the first ✅ here is a
single Python layer-diff pass on Wan 2.2 (the most complete) once weights are downloaded.
