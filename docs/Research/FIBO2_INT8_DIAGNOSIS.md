# Fibo 2.0 int8 dev: is the 23.5 dB gap quantization or an engine fault?

Status: **not established.** The GPU runs below separate the gap into parts, but one measured pair (the engine's int8 path against its own dequantized weights) is confounded by the export, and the decisive pass criterion cannot be applied as written. Nothing here is a parity claim.

## Setup

- Prompt: "A red bicycle leaning against a white wall in a sunlit courtyard, photorealistic." 1024×1024, 50 steps, CFG 5, seed 42, one image per run, RTX 4090 (24 GB).
- Quantized checkpoint: `diffusion_models/fibo2_dev_int8_convrot.safetensors` (sha256 `063dd397…`), int8_tensorwise with ConvRot 256, 215 int8 linears, per-row F32 scales.
- Dequantized reference (export): the int8 weights widened to BF16 with the ConvRot inverse, written by `Fibo2Int8DequantExportTests`. Export sha256 `efbb7082…`. 674 tensors: 634 BF16 (names match `fibo2_dev_bf16.safetensors`, 0 shape or dtype mismatches) and 40 extra U8 `comfy_attention.config` hints.
- Engine head for the runs: not recorded in the handover; the GPU job logs hold it (see Evidence).
- Encoder: the int8 encoder is `text_encoders/fibo2/comfy-org/qwen3vl_4b_int8_convrot.safetensors` (sha256 `32de113c…`). The bf16 encoder is the one the bf16 build loads. Which encoder the 2026-11-26 bf16 run (AV) used is taken from the peer's job notes, not from its log, and is marked as such below.

## Runs

| Run | DiT weights | Encoder | Result |
|---|---|---|---|
| AV int8 (2026-11-26) | int8 kernels on the int8 checkpoint | int8 | 117 s, engine peak 12,902 MiB |
| AV bf16 (2026-11-26) | bf16 `fibo2-dev` | bf16 (per peer notes; not in the AV log) | 211 s, engine peak 20,394 MiB |
| E2 (2026-11-27) | the export, BF16, through the bf16 path | int8 (named in the log) | rc 0, 114 s, peak 21,246 MiB; PNG `08e65a49…` |
| E3 (2026-11-27) | `fibo2_dev_bf16.safetensors`, real bf16 | int8 (named) | rc 0, 192 s, peak 21,242 MiB; PNG `20080d8e…` (same hash as the earlier E3 run) |

E2 and E3 both run the bf16 path, so they differ only in the DiT weights.

## PSNR (average over channels, dB)

| Pair | dB | What it differs by |
|---|---|---|
| A. E2 vs AV int8 | 29.30 | DiT weights (export against int8 kernels); encoder equal (both int8) |
| B. E2 vs E3 | 25.99 | DiT weights only: export against released bf16; encoder equal (both int8); path equal (both bf16) |
| C. E3 vs AV bf16 | 23.25 | Encoder only, if AV bf16 used the bf16 encoder (peer notes); DiT equal (both bf16) |
| Ref. AV int8 vs AV bf16 | 23.53 | DiT and encoder both (the original gap) |

## What each pair isolates

- **B (DiT weights, same path).** The cost of the int8 DiT representation as seen through the bf16 path, with the encoder fixed. It is a clean isolation of the weights, apart from the confounds below.
- **C (encoder).** The cost of the int8 encoder alone, with the bf16 DiT fixed, under the assumption that AV bf16 used the bf16 encoder. If that assumption is wrong, C is not an encoder-only pair.
- **A (kernel path against its own weights).** The engine's int8 kernels against the same weights widened to BF16 in the export. This is the pair the diagnosis plan calls decisive. It mixes the kernel path with the export's own rounding.
- **Ref.** The original 23.53 dB is close to C (23.25 dB), which suggests that the encoder accounts for most of it. That is a consistency statement, not a decomposition: two PSNRs do not add.

## Confounds

1. **Truncated export.** `Int8ConvRotCodec.DequantToBf16` casts with `TensorCasts.F32ToBf16Bits`, which keeps the top 16 bits and so truncates toward zero. The export inherits that: each weight is off by up to one BF16 ulp (up to 2^-7 of its value), in one direction. Pair A is measured against that export, so A includes a systematic weight bias that a correct dequant would not have. The independent check (`tools/fibo2-int8-diagnosis/independent_dequant.py`, job BM: exit 0, worst error 7.618e-03 of the tensor max against a truncation bound of 7.828e-03) confirms the widening is within truncation of the definition. It does not measure what truncation does to the image.
2. **Attention hints.** The 40 `comfy_attention.config` U8 tensors are present in the E2 file and absent in the bf16 file. The engine validates them, logs a warning per backend, and drops them (`Fibo2AttentionHints.Requested`, `Fibo2WeightPreparation.HalfPrecisionWeights`). Nothing reads the backend name, and attention stays float unless `numerics.int8AttnV2` is set (default false). So they should not change the run. The E2 log must still show the `comfy_kitchen_int8` warning and no int8-attention line. The GPU handover does not state `numerics.int8AttnV2` for E2, E3 or AV int8. Each must be checked against its log before pairs A and B are read.
3. **Kernel arithmetic.** The engine's int8 path does not cast the weight to BF16 before the GEMM, so pair A compares different arithmetic as well as different weight values.
4. **AV bf16 encoder.** Taken from the peer's job notes, not from the AV log. If AV bf16 used the int8 encoder, C is not an encoder pair and the encoder reading above is void.
5. **Single image.** One prompt, one seed, one PSNR per pair. The PSNR says how far two images are apart, not which one is better, and the 25 dB floor is a proposed threshold, not a quality bar.
6. **CLI head.** The handover does not give the engine head for E2 and E3. AV int8 and AV bf16 were `1ce13303a`. Pair B is only clean if E2 and E3 ran the same head as each other, and pair C only if E3 and AV bf16 did.

## Read-out against the plan

The plan (`rmbg-fibo2-int8-diagnosis`, Test A) has three bands for int8 against its dequantized reference: at least 40 dB means the path matches its weights; below 30 dB means an engine fault to find; in between, report and go to Test C.

Pair A is 29.30 dB, in the below-30 band. That band assumes the dequantized reference is exact. It is truncated (confound 1) and the paths differ in arithmetic (confound 3), so the band does not yet identify an engine fault. Test A is therefore **inconclusive**, not failed.

## Conclusion

Established by these runs:

- Within this pipeline, the int8 DiT representation alone costs about 26 dB (pair B), with the encoder and path fixed.
- The int8 encoder alone costs about 23 dB (pair C), if the AV bf16 encoder assumption holds. That is close to the original 23.5 dB gap, so the encoder is the most likely main contributor.
- The engine's int8 path does not match its own dequantized weights to the 40 dB the plan required. It is 29.3 dB, and that figure is confounded by the truncated export and the kernel arithmetic.

Not established:

- Whether the engine's int8 kernels are correct. Pair A cannot separate a kernel fault from the export's truncation.
- Whether the 23.5 dB is expected from quantization in ComfyUI. Test B (ComfyUI on the same file) has not run.
- Where the image first diverges. Test C (per-step latents) needs a dump hook that does not exist yet.
- Image quality. These are PSNR numbers. Visual judgement is not recorded here.

## Next experiments

1. **Round-to-nearest export.** Re-export the dequant with round-to-nearest and rerun pair A. If A moves toward 40 dB, the truncation accounts for most of the 29.3 dB; if it stays near 29 dB, look at the kernel arithmetic.
2. **Check hints and attention in each log.** Confirm `numerics.int8AttnV2` was off in E2, E3, AV int8 and AV bf16, and that E2 shows the hint warning and nothing else.
3. **Confirm the AV bf16 encoder** from its log, or rerun it with the bf16 encoder named.
4. **Test B and Test C,** as the plan describes.

## Evidence

- E2, E3: `/home/hartsy/evidence/peer-job-bj-2026-11-27/` (peer job BJ; export sha256 `efbb7082…`, checked by the peer).
- Independent dequant check: job BM, `/home/hartsy/evidence/peer-job-bm-2026-11-27/` (exit 0; worst 7.618e-03 at `layers.32.feed_forward.w13.weight`, bound 7.828e-03).
- AV int8 and AV bf16: `/home/hartsy/evidence/peer-job-av-2026-11-26/` (reported by the peer; not re-verified here).
- Plan: `rmbg-fibo2-int8-diagnosis.md` in the side-session notes.
