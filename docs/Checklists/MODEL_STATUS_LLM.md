# LLM, text encoders, VLMs, and embeddings

[Legend](MODEL_STATUS.md) · [Numerical evidence](PARITY_VERIFICATION.md) ·
[Performance](../../benchmarks/scoreboards/LLM.md) · [Open work](#remaining-work).
Verification is checkpoint/path-specific; historical 12 GB hardware limits are not architectural limits.

## CLI catalog-path verification (2026-07-22)

The real CLI pass verified coherent/correct outputs for qwen3 (including thinking/no-thinking), llama3,
mistral, gemma, phi, granite3, olmoe, granite-moe, gpt2, starcoder2, mamba, qwen25-vl, llava15, and smolvlm2.
Stablelm2 and gemma4 passed factual prompts; stablelm2 arithmetic quality and gemma4 short-run timing
were not conclusive engine diagnostics. These are recorded runs, not a fresh fleet verification.

Partial paths: qwen35 used ChatML fallback (real-template tool filter gap); RWKV-6-World was not
instruction-tuned for that prompt format; llama32-vision had variant-specific limitations; command-r
was host-OOM-killed (observed peak about 3.1× file size, above the loader's 2.5× estimate).

Outstanding findings from that pass:

| Path | Evidence and remaining gap |
|---|---|
| GLM-4 Q4_K | Partial interleaved RoPE fixed; coherent prose recovered. Precise retrieval still failed on the same unsloth GLM-4-9B-0414-Q4_K_M GGUF that llama.cpp answered correctly. CPU F32 synthetic logits matched HF to maxAbs 1.3e-6 across 16 positions. This narrows the lead to quantized/shape-dependent behavior; it does not prove the exact kernel cause. |
| Gemma-3 vision | Vision tower cosine approximately 1.0. Removing erroneous sqrt(hidden) scaling of image embeddings changed output but did not resolve real-photo hallucinations. Mixed causal/bidirectional image-block attention remains a lead requiring verification. |
| DeepSeek-V2-Lite | Real checkpoint loaded but produced incoherent output; MLA cause unconfirmed. |
| GPT-OSS | MXFP4 type-39 codec gap fixed and byte-layout tested; this pass did not verify full-model generation. |
| Gemma-4 MoE | Fused gate/up expert splitting fixed. Per-expert down-projection scales were an unresolved numerics gap; consult the newer variant details below before generalizing. |

Resolved integration traps: catalog directories passed where a GGUF file was required; literal RWKV
chat-template sentinel; Jinja boolean/slice support; VLM repetition defaults; image-embedding scaling;
Qwen2.5-VL host RoPE; pre-scaled graph embedding tables for non-unit EmbeddingScale.
Granite graph embedding-table math was tested, not a full eager-vs-graph generation comparison.

Follow-up measurements: verify graph decode on eligible starcoder2/phi3/stablelm2; profile unsupported
fused quant variants and BERT rotary/MoE host work. T5 generation recomputes its growing decoder prefix
without a KV cache and is not wired into TextService. SSM recurrence performance needs dedicated kernel
work, not a claim that the family is unimplemented. Current throughput belongs in the scoreboard.

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
| **Gemma-4** (E2B/E4B mobile) | Gemma-4-E2B-it Q4_K_M | Ported cold from llama.cpp's `src/models/gemma4.cpp` (no local reference model existed) — per-layer embeddings (PLE, Gemma-3n-lineage), per-layer head dim (global 512 vs local/SWA 256 on E2B — the Q/K/V/O weight shapes themselves differ, not just … ([details](#gemma-4)) |

Qwen-MoE shared-expert path is unit-test-verified against an HF reference (no 14 GB GGUF needed).

## LLM — build-defer (🔧, wired but >12 GB)

| Model | Notes |
|---|---|
| **Mixtral 8x7B** (47B) | `llama` arch + experts, interleaved RoPE, renorm; config + mapper + stacked-expert split wired. |
| **Qwen3-MoE 30B-A3B / 235B** | `qwen3moe`, per-head Q/K norm, no shared expert; wired. |
| **DeepSeek-V2-Lite** | MLA + DeepSeek-MoE built + `MlaTests` pass; loads but OOMs the 3060 at preload. |
| **DeepSeek-V3 671B / Kimi-K2 1T** | MLA + MoE + **V3 node-limited routing (sigmoid + e_score bias + group top-k + routed_scaling) + q-LoRA query** all built & **slice-verified** (`MoeTests` group-routing vs HF `noaux_tc`, `MlaTests` q-LoRA block vs host ref). e2e >12 GB. |
| **GPT-OSS 20B / 120B** | Per-head **attention sinks** built (CPU+CUDA, PTX recompiled) & **slice-verified** (`FlashAttentionTests.Flash_Sink_*`); `gpt-oss` arch/mapper/config wired. MoE + o200k tokenizer reused. MXFP4 tensor-type decode added 2026-07-22 (`DType.MXFP4`/`Codec_MXFP4`) — the load-time crash every public checkpoint hit is fixed. e2e 20B+ still deferred (VRAM, not an engine gap anymore). |
| **Llama-3.2-Vision-11B (mllama)** | ✅ **VERIFIED e2e on the 3060** (leafspark Q4_K_M + mmproj-F16, low-VRAM): red circle→"red", blue square→"a blue square with a white outline…". ([details](#llama-32-vision-11b-mllama)) |
| **Gemma-4 31B-dense / 26B-A4B-MoE** | Same `gemma4` arch/config/forward path as the verified E2B row above — the 26B-A4B variant additionally exercises `ParallelDenseMoeBranch` (routed-expert FFN running IN PARALLEL WITH the dense/"shared" branch, each own pre/post norm, summed — a genuinely different pattern from every other MoE arch here, which fuses a shared expert into the router output instead) and `ComputeRouterLogits`'s separately-normalized router input. ([details](#gemma-4-31b-dense--26b-a4b-moe)) |

## VLMs (vision-language) — mixed verification

| Model | Notes |
|---|---|
| **Gemma-3-4B-vision** | ⚠️ Tower parity and simple synthetic-image generation; real-photo hallucinations remain unresolved. SigLIP + avg-pool/RMSNorm/Linear projector. ([details](#gemma-3-4b-vision)) |
| **SmolVLM2-2.2B** | SigLIP + idefics3 pixel-shuffle projector. Tower corr 1.0; e2e correct. CLI-reverified 2026-07-22 against a real photo (not just a synthetic shape) — still correct. |
| **LLaVA-1.5-7B** | CLIP ViT (CLS token, pre-LN, quick-GELU, penultimate layer) + MLP projector. Tower corr 1.0; e2e "a red circle … a Japanese flag". CLI-reverified 2026-07-22 against a real photo — still correct. |
| **LLaVA-NeXT (1.6) Vicuna-7B** | `LlavaNextEncoder` (new): reuses `SiglipVlmEncoder`'s CLIP tower + `mm.0`/`mm.2` projector unchanged (identical GGUF tensor shapes to LLaVA-1.5) per-tile via composition, adding `LlavaNextImagePreprocessor` (anyres tiling: `select_best_resolution`/`get_patch_output_size`/pad+`divide_to_patches`, base tile + best-fit grid) and `LlavaNextFeatureMerger` (`pack_image_features` port: reshape/permute → unpad → `image_newline` insert → base-first concat). ([details](#llava-next-16-vicuna-7b)) |
| **Qwen2.5-VL-3B** | Own ViT — Conv3D patch embed, 2D-RoPE, window attention (full at 7/15/23/31), SwiGLU, 2×2 merger. All stages corr 1.0; e2e correct. |
| **Qwen2.5-VL-7B** | Same `Qwen25VlEncoder` + qwen2 text as the 3B. ([details](#qwen25-vl-7b)) |
| **Llama-3.2-Vision-11B (mllama)** | See build-defer table above for the original engine-internal verification (red circle→"red" etc). ([details](#llama-32-vision-11b-mllama-2)) |

Shared `SiglipVlmEncoder` (SigLIP + CLIP towers, 3 projectors auto-detected) + dedicated `Qwen25VlEncoder` +
`MllamaVisionEncoder`, behind `IVlmImageEncoder` / the mllama cross-attention path. Production wiring done: real
`SamplerChain`, reusable `VlmImagePreprocessor`, real PNG decode path. All four small VLM families + the two
larger ones (Qwen2.5-VL-7B, Llama-3.2-Vision-11B) are now verified e2e on the 3060.

## Embeddings — verified end-to-end (✅)

| Model | Notes |
|---|---|
| **bge-small-en-v1.5** | `BertEmbeddingModel` (BERT encoder), CLS pooling. **cosine = 1.000000** vs HF transformers. |
| **all-MiniLM-L6-v2** | `BertEmbeddingModel`, mean pooling. **cosine = 1.000000** vs HF. |
| **nomic-embed-text-v1.5** | `BertEmbeddingModel` config-driven path: rotary position + fused QKV + SwiGLU + no biases. **cosine = 1.000000** vs HF. |
| **Qwen3-Embedding-0.6B** | `DecoderEmbeddingModel` (reuses the qwen3 decoder + last-token pooling). **cosine = 1.000000** vs HF. Covers gte-Qwen2 / e5-mistral. |
| **bge-reranker-v2-m3** (reranker) | `BertEmbeddingModel.Score` — xlm-roberta encoder + cross-encoder head (cls→tanh→out_proj). Logit matches HF (relevant 4.63, irrelevant −11.04). |

Bidirectional post-norm BERT + CLS/mean pooling + L2-normalize; `bert`/`nomic-bert` registered as passthrough archs.
E2E via GGUF-vocab `BertWordPieceTokenizer`: cos(cat,kitten) 0.91 > cos(cat,car) 0.78. Quant decode verified
(Q8_0/Q5_K/Q4_K/Q3_K all >0.99 vs F32; codecs for all K-quants + legacy + IQ4_NL).

## Non-transformer architectures

| Family | Status | Notes |
|---|---|---|
| **Mamba-1 (SSM)** | ✅ verified | `MambaModel` — selective state-space scan + causal Conv1d, no attention. mamba-130m next-token logits **cosine = 1.000000** + argmax match vs HF. GGUF `ssm_a` is pre-baked `−exp(A_log)`. Mamba-2 / Falcon-Mamba reuse the path. |
| **RWKV-6** | ✅ verified | `RwkvModel` — WKV6 recurrence + data-dependent token-shift LoRA + GroupNorm. C# runs at **cosine 1.0** (argmax 281) vs the validated Python ref (= official `rwkv` package). No-copy `Reshape` relabel views to fit the 1.6B model in host RAM. RWKV-7 = near-variant. |
| **Hybrids** (Jamba/Zamba2/Granite-4) | ⬜ planned | Mamba + attention + MoE interleave (7c). |
| **Qwen3.5 dense** (Gated DeltaNet hybrid) | ✅ verified (0.8B) | `Qwen35Model` (new `HartsyInference.LLM.Ssm.ISsmModel`, not `GenericTransformer` — mixes TWO attention mechanisms per model, not pure-recurrent like the rows above). ([details](#qwen35-dense)) |
| **Encoder-decoder** (T5/FLAN-T5) | ✅ verified | `T5Model` — full seq2seq (rel-pos bias, no 1/√d scaling, cross-attn, GeGLU). flan-t5-small encoder + decoder **cosine = 1.0** vs HF; e2e "Das Haus ist schön." BART is a near-variant. |

## Text / vision encoders

| Encoder | Status | Notes |
|---|---|---|
| **GenericTransformer** (Qwen2/Qwen3/Llama-3) | ✅ | Parity tests; backs the text encoders + native LLM. Llama-3 RoPE NTK-by-parts validated. |
| **T5 / UMT5 / Pile-T5 (AuraFlow) / BERT / SigLIP / Qwen3-VL vision tower** | 🔬 | Diff tests present (`T5EncoderDiff`, `BertModel`, `Siglip`, `Qwen3VlVisionTower`); Pile-T5 == UMT5 (per-layer rel-attn-bias). |

## Remaining work

Distilled from the retired PHASE_12_LANGUAGE / LLM_MODEL_COVERAGE / LLM_CLI_CATALOG_HANDOFF /
VPS_GPU_HANDOFF / QWEN35_GPU_VERIFICATION plans. The "Remaining for FULL LLM support" (Phases 6-9) section
above already tracks the hybrid families, build-defer e2e, and serving items; this section captures the rest.
Items now ✅ above (Llama-3.x / Mistral / Qwen coverage, flash-attention, the small VLMs, gpt2, starcoder2,
`--thinking`/`--no-thinking`, `--image`) are omitted.
See [ROADMAP.md](ROADMAP.md) for cross-cutting infra (multi-GPU, kernel perf, quant, serving).

### Architecture coverage frontier
- [ ] Nemotron-H (Mamba-proper hybrid — beyond the Jamba / Zamba2 / Granite-4 hybrids already in Phase 7).

### CLI catalog
- [ ] Reconcile remaining catalog variants/assets against the recorded CLI pass; rerun only unverified or changed paths.
- [ ] Wire the T5 / seq2seq generation loop in `TextService` (T5 is not reachable via `hartsy text` today).
- [ ] `glm4` still FAIL — root-cause the quantized (Q4_K) precise-retrieval bug (the F32 path is proven correct).

### Qwen3.5 GPU verification (OOM'd the dev box — needs a big-VRAM GPU)
- [ ] `moe-text` (35B-A3B), `moe-vl`, `vl9b`, `kat` tiers unrun.
- [ ] Flip `qwen35moe` to Verified once green on a big-VRAM GPU.

### Testing
- [ ] Refresh checkpoint-specific parity/long-context evidence where it is absent; do not treat old phase counts as a gate.

## Details

Verification evidence, bugs found, and caveats for the rows above. Moved out of the status
tables on 2026-08-06 so the tables stay scannable — no content was dropped.

### Gemma-4

Ported cold from llama.cpp's `src/models/gemma4.cpp` (no local reference model existed) — per-layer embeddings (PLE, Gemma-3n-lineage), per-layer head dim (global 512 vs local/SWA 256 on E2B — the Q/K/V/O weight shapes themselves differ, not just RoPE theta), cross-layer KV-cache sharing (donor-layer formula from llama.cpp's kv-cache `reuse` callback), weightless V RMSNorm, optional per-layer output scale, per-layer FFN width (derived from the loaded weight's own shape, not a config constant — some layers are 2× wider). Found 3 real bugs along the way, none Gemma-4-specific: our GGUF parser silently discarded BOOL-typed arrays (`sliding_window_pattern` is a genuine per-layer array, not a broadcast period — every layer was reading as "global"); our `Linear`'s `Tensor.Shape[0]`=outDim/`Shape[1]`=inDim convention got reversed in my first pass; `tokenizer.ggml.model="gemma4"` wasn't routed to the SentencePiece tokenizer (fell through to generic BPE, dropping the `▁`→space substitution — coherent but "Thecapitalof..."). Also fixed 4 general Jinja chat-template bugs surfaced by Gemma-4's tool-calling template (unary minus, block-form `{% set %}...{% endset %}`, `range()`, `is sequence`). Verified live: coherent, factually correct, properly-spaced multi-sentence generation.

### Llama-3.2-Vision-11B (mllama)

✅ **VERIFIED e2e on the 3060** (leafspark Q4_K_M + mmproj-F16, low-VRAM): red circle→"red", blue square→"a blue square with a white outline…". The only splice-free VLM — vision feeds gated cross-attention layers `[3,8,13,18,23,28,33,38]`. `MllamaVisionEncoder` (560px ViT, class embd, pre/post-tile + dual-gated position embeds, 32 local + 8 gated-global, intermediate-concat `[3,7,15,23,30]`→7680→`mm.0`→4096) **reference-validated cos=1.000000 on every stage** (`dump_mllama_vision_ref.py`). Key finding: Ollama's converter **pre-tanh's all gates** (and `1−tanh` for `position_embd.gate`, splitting HF's single gate into two) so the forward multiplies gates directly; MLP names clip-swapped; q/k permute is a no-op (no vision RoPE). `MllamaCrossAttentionLayer` slice-verified (`MllamaCrossAttentionTests`); `MllamaGenerator` (no token splice, `crossStates` threaded through `ForwardEmbeds` every step). Covers the whole mllama family.

### Gemma-4 31B-dense / 26B-A4B-MoE

Same `gemma4` arch/config/forward path as the verified E2B row above — the 26B-A4B variant additionally exercises `ParallelDenseMoeBranch` (routed-expert FFN running IN PARALLEL WITH the dense/"shared" branch, each own pre/post norm, summed — a genuinely different pattern from every other MoE arch here, which fuses a shared expert into the router output instead) and `ComputeRouterLogits`'s separately-normalized router input. Built and compiles; not e2e verified (>12 GB, exceeds the 3060 — per user directive, no local load attempted). Real checkpoint's fused `ffn_gate_up_exps` MoE tensor now handled correctly (2026-07-22, was a `KeyNotFoundException`) — `ffn_down_exps.scale` still unsupported, see the follow-up pass section above.

### Gemma-3-4B-vision

SigLIP + avg-pool/RMSNorm/Linear projector. Tower corr 1.0 vs reference; e2e coherent on simple synthetic shapes. 2 bugs fixed (swapped SigLIP MLP names; relabel-not-transpose). ⚠️ **CLI pass 2026-07-22 (real photo, not a synthetic shape) found a real, still-unresolved FAIL**: `hartsy text -m gemma3-vision -i <bus photo>` consistently hallucinates an unrelated scene despite the vision-tower math independently re-verified as numerically correct (fresh PyTorch parity replay, cosine ≈1.0 every stage). **Follow-up same day**: found + fixed a real bug (image embeddings were erroneously scaled by the √hidden normalizer that should only apply to text tokens, confirmed against HF `transformers` source — ~50.6x overamplification) — re-verified live, hallucination persists with different fabricated content, so a SECOND real gap was identified (this engine's LLM decode attention is unconditionally causal; real Gemma-3 gives image tokens bidirectional attention within their block via a mask this engine has no mechanism for) but not fixed (genuine new engine capability, out of scope). See MODEL_STATUS_LLM.md's "CLI verification evidence" section. Simple-shape tests (red circle, blue square) are apparently not sufficient to catch this; a real-photo regression test is worth adding.

### LLaVA-NeXT (1.6) Vicuna-7B

`LlavaNextEncoder` (new): reuses `SiglipVlmEncoder`'s CLIP tower + `mm.0`/`mm.2` projector unchanged (identical GGUF tensor shapes to LLaVA-1.5) per-tile via composition, adding `LlavaNextImagePreprocessor` (anyres tiling: `select_best_resolution`/`get_patch_output_size`/pad+`divide_to_patches`, base tile + best-fit grid) and `LlavaNextFeatureMerger` (`pack_image_features` port: reshape/permute → unpad → `image_newline` insert → base-first concat). Both new pieces ported from the REAL installed `transformers` source (not memory) since llama.cpp's own LLaVA-NeXT merge is a known-buggy reference (base tile last, no unpad, unused `image_newline`; ggml-org/llama.cpp#8457) — HF's own docstrings in `modeling_llava_next.py` even contradict each other on (H,W) vs (W,H), resolved empirically against `select_best_resolution`'s actual body. Merge/tower numerically validated by feeding identical Python-computed pixels through the C# tower+merge: corr ≥0.99994 on both `unpad_image` branches (portrait bus.png → 2352 tokens/crop-width branch; the same photo rotated 90° → 2340 tokens/crop-height branch — closes the H/W-swap-prone conditional both ways). C#'s own tiling (bilinear, not HF's bicubic — same approximation already accepted for every other family's resize) checked structurally: corr 0.996 vs HF's real image processor, tile count/dims exact, pad-then-normalize order confirmed (`-mean/std` in padded regions, not 0). CLI e2e 2026-07-24 on a real photo: correctly read on-image text ("Cero Emisiones"), bus color, two people crossing. Catalog id `llava16` (`cjpais/llava-v1.6-vicuna-7b-gguf`). ⚠️ **VRAM finding**: the anyres tile grid can push image-token count to 4x+ LLaVA-1.5's (2352 vs 576 for a 2×2 grid), and `FixedKvCache` sizes to `seqLen+maxTokens` with no paging — this OOM'd repeatedly on the 3060's 12GB (recovered via allocator retry, but corrupted the timing); the RTX 3060 comfortably fits LLaVA-1.5 but LLaVA-NeXT needs more headroom (verified clean on the 4090). **Perf pass vs llama.cpp** (same 4090, `llama-cpp-python` 0.3.34 CUDA, `Llava15ChatHandler` — the only handler llama.cpp has for any LLaVA variant): llama.cpp decode 129.35 tok/s / ttft 642ms vs this engine's 100.82 tok/s / 2047ms prefill — BUT llama.cpp's log shows it only encodes ONE 576-token image slice (`clip_image_batch_encode: output embedding shape [4096, 576, 1]`), i.e. its documented merge bug appears to make it skip the anyres grid entirely and process only the base/overview tile, not the 4x-richer 2352-token sequence this engine's (transformers-verified-correct) pipeline produces — so the decode-speed gap partly reflects llama.cpp doing structurally less visual work, not a clean apples-to-apples comparison. Not chased further (would require patching llama.cpp itself to force full anyres, out of scope).

### Qwen2.5-VL-7B

Same `Qwen25VlEncoder` + qwen2 text as the 3B. **Verified e2e on the 3060** (unsloth Q4_K_M + mmproj-F16, low-VRAM): blue→"Blue.", red→"Red." in ~5s. Bring-up fixes: metadata-based Qwen mmproj detection (`clip.projector_type`, not filename) + a CUDA int-overflow in the cast byte-size math that OOM'd the 152k-vocab Q6_K lm_head (`count * SizeInBytes` widened to 64-bit — affected any large-vocab quantized head). CLI-reverified 2026-07-22 on the 4090 against a real photo — best result of the whole VLM CLI pass, correctly read on-image text ("cero emisiones").

### Llama-3.2-Vision-11B (mllama) (2)

See build-defer table above for the original engine-internal verification (red circle→"red" etc). CLI-reverified 2026-07-22 against a real photo: PARTIAL — correctly identifies the broad scene (bus, street, people, trees) but gets the bus color wrong and doesn't reliably stop at content end (free-runs into a hallucinated follow-up turn past a normal token budget; confirmed the eos token IS registered in StopIds, so this is model behavior, not a stop-token wiring bug — a lower `--max-tokens` truncates cleanly).

### Qwen3.5 dense

`Qwen35Model` (new `HartsyInference.LLM.Ssm.ISsmModel`, not `GenericTransformer` — mixes TWO attention mechanisms per model, not pure-recurrent like the rows above). Every 4th layer (`full_attention_interval`) is regular GQA + partial RoPE + KV cache (Qwen3-style QK-norm, plus a fused query+gate projection unique to this arch — query and a sigmoid gate share one projection, de-interleaved per head); the rest are Gated DeltaNet: causal Conv1d over a fused QKV projection, per-head L2-norm on Q/K, then a sequential delta-rule recurrence (`S_t=α_t·S_{t-1}+β_t(v_t−S_{t-1}k_t)k_t^T`, `o_t=S_t·q_t`) → gated RMSNorm(o, silu(z)) → out_proj. Ported cold from llama.cpp's `src/models/{qwen35.cpp, delta-net-base.cpp}` (no local reference model existed). Text-only M-RoPE degenerates to ordinary partial-rotary RoPE (every section gets the same position when there's no multimodal input) — no M-RoPE machinery built. **Real bug found and fixed via live testing**: missed the `q *= 1/√S_k` scale llama.cpp applies right before the recurrence (present in the reference, easy to miss reading it once) — produced word-salad, not a crash; fixing it flipped straight to coherent, factually correct, grammatically clean output. Verified live: 0.8B, ~100 tokens, stable. 2B/4B/9B share the identical code path, untested (same-tier VRAM as tested, likely fine, just not run). `qwen35moe` (the 35B-A3B/122B-A10B/397B-A17B MoE tier) **is now implemented (2026-07-27)**: same hybrid GDN/full-attn trunk, every trunk layer gets a `MoeFeedForward` (256 experts, top-8, softmax + top-k renorm) + a sigmoid-gated shared expert; stacked `ffn_*_exps` stay quantized and are split into per-expert views; MTP/NextN block skipped; device-step + CUDA-graph decode disabled for MoE (`GraphDecodeReady=false`). Compiles + dense/VL regressions pass; 35B real-weight run deferred (OOMs the dev box). **Vision (2026-07-27): Qwen3.5/3.6-VL now works** — `Qwen3VlEncoder` (`v.blk.N.attn_qkv` fused-split, LayerNorm, GELU, full attention, 2D-RoPE, learned `v.position_embd` bilinear-interpolated, `mm.0/mm.2` merger) + `Qwen35VlGenerator` (embeds-in prefill via new `Qwen35Model.ForwardEmbedsLastLogits`/`EmbedLookup`) drove an accurate OCR/VQA caption of `bus.png` on Qwen3.5-0.8B-Q4_K_M + its `mmproj-F16.gguf`. Loader fix: `clip`→`PassthroughKeyMapper` (fused `attn_qkv` was making `PhiKeyMapper` steal the mmproj). Deepstack-free (this mmproj has no deepstack tensors); text-side spatial M-RoPE not yet applied (degenerate scalar, as with Qwen2.5-VL).
