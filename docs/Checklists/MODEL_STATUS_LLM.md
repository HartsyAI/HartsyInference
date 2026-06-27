# LLM + Text Encoders + VLMs — status

Concise status for native text generation, text/vision encoders, and vision-language models. Goal:
support every architecture Ollama / llama.cpp runs via the config-driven `GenericTransformer` spine, so a
new model is *a preset + key-mapper + (rarely) one engine knob*, never a new transformer class. Full
phased plan + per-model bring-up notes live in [LLM_MODEL_COVERAGE.md](LLM_MODEL_COVERAGE.md) and
[PHASE_12_LANGUAGE.md](PHASE_12_LANGUAGE.md). Legend: [MODEL_STATUS.md](MODEL_STATUS.md).

Two bars apply: **Runnable@3060** (fits 12 GB at Q4/Q8, verified coherent e2e on this box) and
**build-defer** (arch + key-map + slice tests land, e2e marked pending-hardware for >12 GB models).

## LLM — verified end-to-end (✅, Runnable@3060)

| Family | Verified | Notes |
|---|---|---|
| **Llama 1/2/3.x** | Llama-3.2-1B Q8 | Interleaved (NORM) RoPE; Llama3 `rope_freqs` divisor fix. |
| **Mistral / TinyLlama / SmolLM / Yi** | Mistral-7B-v0.3, SmolLM2-1.7B, TinyLlama-1.1B, Yi-1.5-6B | All run via the llama-family path unchanged. |
| **Qwen2 / Qwen2.5** | 0.5B Q4_K_M + Q8 | — |
| **Qwen3** | 0.6B (incl. `<think>`) | — |
| **Gemma 2 / Gemma 3** (text) | Gemma-3-1B, Gemma-2-2B Q4_K_M | GeGLU + √d embed scale + sandwich norm + dual-RoPE; attn + final logit soft-cap. SPM-from-GGUF tokenizer. |
| **Phi-3 / Phi-3.5-mini / Phi-4-mini** | all three | Fused QKV split, LongRope, partial rotary, non-pow2 head-dim fix, gpt-4o tokenizer. |
| **StableLM-2** | 1.6B | Partial rotary + QKV bias. |
| **Granite-3** | 3.1-2B | Embedding/attention/residual/logit scalar multipliers. |
| **Cohere Command-R** (cohere2) | Command-R7B | LayerNorm + parallel residual + interleaved RoPE + NoPE global layers + logit-scale. |
| **OLMoE** | 1B-7B-0924 | MoE wired from GGUF; whole-vector Q/K norm. |
| **Granite-MoE** | granite-3.0-1b-a400m | Scalars + MoE combined. |

Qwen-MoE shared-expert path is unit-test-verified against an HF reference (no 14 GB GGUF needed).

## LLM — build-defer (🔧, wired but >12 GB)

| Model | Notes |
|---|---|
| **Mixtral 8x7B** (47B) | `llama` arch + experts, interleaved RoPE, renorm; config + mapper + stacked-expert split wired. |
| **Qwen3-MoE 30B-A3B / 235B** | `qwen3moe`, per-head Q/K norm, no shared expert; wired. |
| **DeepSeek-V2-Lite** | MLA + DeepSeek-MoE built + `MlaTests` pass; loads but OOMs the 3060 at preload. |
| **DeepSeek-V3 671B / Kimi-K2 1T** | MLA + MoE done; V3 sigmoid / group-routing + q-LoRA still TODO. Architecturally a bigger config of the above. |

## VLMs (vision-language) — verified end-to-end (✅)

| Model | Notes |
|---|---|
| **Gemma-3-4B-vision** | Vision tower numerically correct (corr 1.0 vs reference); e2e coherent image-grounded text. 2 bugs fixed (swapped SigLIP MLP names; relabel-not-transpose). |
| **SmolVLM2-2.2B** | SigLIP tower corr 1.0; e2e correct answers. idefics3 pixel-shuffle projector. |

Next VLM targets (Qwen2.5-VL-3B own ViT, LLaVA) not started; Llama-3.2-11B-Vision / Qwen2.5-VL-7B+ are
build-defer.

## Text / vision encoders

| Encoder | Status | Notes |
|---|---|---|
| **GenericTransformer** (Qwen2/Qwen3/Llama-3) | ✅ | Parity tests; backs the text encoders + native LLM. Llama-3 RoPE NTK-by-parts validated. |
| **T5 / UMT5 / Pile-T5 (AuraFlow) / BERT / SigLIP / Qwen3-VL vision tower** | 🔬 | Diff tests present (`T5EncoderDiff`, `BertModel`, `Siglip`, `Qwen3VlVisionTower`); Pile-T5 == UMT5 (per-layer rel-attn-bias). |
| **Native LLM package** (Phase 12) | 🚧 | One config-driven transformer backing both LLM + text encoders; CUDA-first quant; GPU-resident decode is the gating blocker. |

## Not yet covered

Embedding models (nomic-embed / bge / MiniLM, encoder mode + pooling), GPT-OSS attention sinks, more quant
formats (Q4_0/Q5_0, Q2_K/Q3_K, IQ-quants), and the longer-context / batch>1 / speculative-decode slices.
See [LLM_MODEL_COVERAGE.md § Phase 5](LLM_MODEL_COVERAGE.md).
