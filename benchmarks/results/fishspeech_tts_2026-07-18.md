# Fish-Speech 1.5 (fishaudio) — end-to-end verification, 2026-07-18

**Verdict: FishSpeech 1.5 works end-to-end, word-perfect, RTF ~1.05 (44.1 kHz). The firefly-gan-vq codec —
the last unverified piece per the handoff — decodes correctly on real weights.**

## What was checked
`<scratchpad>/fsbench/`: loads `model.pth` (DualAR: Llama-style "slow" backbone + "fast"/depth transformer
over 8×1024 codebooks) + `firefly-gan-vq-fsq-8x1024-21hz-generator.pth` (the FSQ codec) via
`PytorchPickleLoader`, `tokenizer.tiktoken` (+ `special_tokens.json` sibling). Upstream v1.5 chat template
(`<|im_start|>system … <|im_end|><|im_start|>user … <|im_end|><|im_start|>assistant\n<|voice|>`), stop token
`<|im_end|>` (id 100004). `FishSpeechPipeline.Synthesize` → DualAR frames → `FireflyDecoder.Decode` → 44.1 kHz.

## Results (RTX 3060, device 1) — whisper (medium.en) word-perfect
| Clip | Prompt tok | Audio | Wall | RTF | Transcript |
|------|-----------:|------:|-----:|----:|------------|
| short | 35 | 3.44 s | 3.56 s | **1.04** | "Hello there, this is a test of the FISH speech synthesizer." |
| long  | 44 | 6.55 s | 6.86 s | **1.05** | "The five boxing wizards jump quickly. Pack my box with five dozen liquor jugs, and then we can rest." |

Both verbatim. The firefly-gan-vq decoder (grouped-residual FSQ → ConvNeXt/SiLU HiFi-GAN, 21 Hz frames →
44.1 kHz) produces clean, intelligible speech — codec confirmed correct on real weights.

## Prior state
DualAR LM was already real-weight parity-verified (slow bit-exact corr 1.0, fast corr 0.9999 — see
`PARITY_VERIFICATION`); the firefly codec had only a synthetic-forward unit test
(`FishSpeechTests.FireflyDecoder_SyntheticForward_CodesToFiniteAudio`) and a gated parity test needing a
Python ref dump. This is the first full text→44.1 kHz e2e confirmation on the real checkpoint.

## Perf note
RTF ~1.05 (near real-time), dominated by the DualAR decode (Qwen2-shape slow + 4-layer fast per frame at
21 Hz). No perf work needed — already near real-time and word-perfect. If ever squeezed, the DualAR decode is
the lever (same GPU-resident-decode pattern as the other Qwen2/Llama audio LMs).
