# DeepSeek-V4.1-Flash — Architecture Research Notes

> Source snapshot: 2026-09-30. This date does not establish current build or verification status.

> Reference notes. Support is tracked row by row in [DEEPSEEK_V41_SUPPORT_MATRIX.json](../Checklists/DEEPSEEK_V41_SUPPORT_MATRIX.json);
> engine status stays in [MODEL_STATUS_LLM.md](../Checklists/MODEL_STATUS_LLM.md) until a row there carries evidence.

Facts below were read from the pinned upstream files, the checkpoint index and shard headers, not from parameter estimates. Where a
reference is silent or two references disagree, the gap is listed in [Reference gaps](#reference-gaps) and stays open until a named test closes it.

## Pinned references

| Reference | Pin | Role |
|---|---|---|
| `deepseek-ai/DeepSeek-V4.1-Flash` | `dba1be0a40aa45a94ad051997016db3960a90277` | Official code, config, tokenizer, encoding, tech report. Weights are unchanged since `df42c109`; `dba1be0a` fixed tool-namespace encoding only. SHA-256 of every non-weight file: `tests/python-reference/deepseek_v41/upstream.sha256` |
| Tech report | arXiv 2609.19969 (`DeepSeek_V41_Tech_Report.pdf` in the repo) | §2.2 CED, §2.3 CSA2, §2.4.1 mHC, §2.4.2 Engram, §2.4.3 DSpark, §2.4.4 FP4 KV, §3.2 persistent KV and SWA bounded replay |
| DSpark paper | arXiv 2607.05147 §3.1–3.2, Alg. 1, §5.2 | The generation loop the official repo omits |
| `deepseek-ai/deepseek-recipe` | `8cadfede7063c896b944e7bae05daa3549ae97ea` | Maintained protocol toolkit named in the model README: `deepseek-recipe-encoding/src/v4/dsv41.rs` (`66260bb5`), `deepseek-recipe/src/stream/state_machine.rs` (`e4bb0f49`), `docs/streaming.md` (`337064a6`). Reference for the streamed parse |
| vLLM | `834ad456d314f801696b259e426c6946e50ee42d` (main, 2026-09-30) | `vllm/models/deepseek_v41/` (`compressor.py` `54dc9c4a`, `nvidia/model_state.py` `30b287db`, `common/engram.py` `8cd93547`), `vllm/v1/worker/gpu/spec_decode/dspark/speculator.py` (`b86f58a4`), DSML parser `rust/src/parser/src/tool/deepseek_dsml/deepseek_v41.rs` (`be68b6d2`) and `vllm/parser/deepseek_v41.py` (`84216005`). Secondary reference for chunking, replay and streaming |
| SGLang | `6cc9352dfe6c5c013750e72b39c127870ef5b54f` (merge of PR 30261) | `speculative/dspark_components/dspark_verify.py` (`43099c9f`), `kernels/dspark_accept.py` (`560c1707`), `benchmark/dspark_sts_fit.py` (`ec8f0943`). Secondary reference for verify/accept |
| `nvidia/DeepSeek-V4.1-Flash-NVFP4` | `3431dde3` | Derivative |
| `amd/DeepSeek-V4.1-Flash-Quark-MXFP4` | `0d56db21` | Derivative |
| `antirez/deepseek-v4.1-flash-gguf` | `dd8a266f` | Derivative, component-stripped |
| `sfxnz/DeepSeek-V4.1-Flash-EXL3` | branch `2.0bpw-mcg-viterbi-lmhead-mxfp8` @ `982b7045` | Derivative; `main` carries no weights |
| `mlx-community/DeepSeek-V4.1-Flash-MLX-4bit` | `100694c1` | Derivative, incomplete draft component |

Short derivative pins are the revisions inspected; resolve them to full hashes before a download or certification run.

## Architecture

- **Sizes.** Vocab 129280, dim 5120, 40 backbone layers plus 3 DSpark draft layers (`mtp.0-2`); `compress_ratios` has 43 entries: layers 0–1
  ratio 0, layers 2–19 ratio 2, layers 20–39 ratio 1, drafts ratio 0.
- **mHC.** The residual is `hc_mult=4` copies. `hc_mixes` = RMS-normalized flattened `[4·5120]` stream · `hc_*_fn [24, 20480]` (fp32) →
  `hc_split_sinkhorn(scale[3], base[24], 20 iterations, eps 1e-6)` → `pre[4]` (sigmoid+eps), `post[4]` (2·sigmoid), `comb[4,4]` (row softmax + eps, then
  alternating column/row normalization). **Single-pass shift**: a sub-block consumes the `pre` produced by the previous sub-block (`Block.forward`); the
  initial `pre` is a one-hot on copy 0.
- **Attention (all layers).** `wq_a [1280,5120]` → RMSNorm → `wq_b [32768,1280]` → 64 heads × 512 (448 nope + 64 rope). One latent `wkv [512,5120]` →
  RMSNorm(512) → RoPE on the last 64; the latent is both K and V (`sparse_attn(q, kv, sink, idxs)`), with a per-head fp32 `attn_sink` in the softmax
  denominator. The output is un-rotated (`apply_rotary_emb(..., inverse=True)`), then grouped `wo_a` (8 groups, 4096 → 1024 each, block-diagonal einsum) and
  `wo_b [5120, 8192]`. Softmax scale 512^-0.5. An all-invalid row yields zeros.
- **Two KV sources per call.** (a) a per-layer SWA ring of 128 fp8-quantized latents including the RoPE tail (`get_window_topk_idxs`, `-1` = empty slot);
  (b) when `compress_ratio > 0`, `index_topk = 512` compressed positions.
- **CSA2 modes.** KV source layers 2, 8, 14 (ratio 2) and 20 (ratio 1, the CED global KV from the layer-20 input). Index source layers 2, 8, 14, 20, 24, 28,
  32, 36; every other compressed layer reuses `shared_attn.topk_idxs`. Candidate source is layer 20 (`select_candidate_blocks`: 2048 blocks × 8 = 16384
  candidates, newest block pinned); 24/28/32/36 re-rank within the candidates. Window 128.
- **Compressor.** Ratio 1: `norm(wkv(x))` in bf16. Ratio 2: fp32 `wkv` + `wgate`, softmax-gated pooling over groups of 2 with **partial-group state carried across
  decode steps**; the latent is emitted only when a group completes; RoPE position of group j is `j·ratio`.
- **RoPE.** Layers with `compress_ratio > 0` use ONE table for q, window K and compressed latent: theta 160000 with YaRN (factor 16, beta 32/1, original 65536).
  Ratio-0 layers use theta 10000 without YaRN. **Rotation is interleaved**: `apply_rotary_emb` calls `view_as_complex` on adjacent pairs, so it is not the
  split-half rotation `MlaForward` uses. This is a candidate cause for the un-root-caused DeepSeek-V2-Lite incoherence and is not evidence about it.
- **Indexer.** `wq_b [4096,1280]` from `qr` (32 heads × 128), RoPE tail 64, FP4 e8m0/32 quantization of Q and K; K = `k_norm(wk(latent))`, written to `k_cache`
  by KV-source layers only. Score = ReLU(q·k) · `weights_proj(x)` · (128^-0.5 · 32^-0.5) summed over heads; causal mask by `compress_lens`; top-k re-sorted by
  position; indices offset by the window length.
- **KV cache encodings.** Main compressed KV: FP4 E2M1 with one E4M3 scale per 16 channels, quantized after RoPE. SWA KV: FP8 e4m3 with ue8m0 per 32.
- **MoE.** `gate.weight [384,5120]` bf16, computed in fp32. `sqrtsoftplus` = `sqrt(softplus(logits))`. Selection = top-6 of `scores + bias` (`bias_vl` for image-span
  tokens); weights = raw scores gathered, normalized with `+1e-20`, ×1.5. SwiGLU clamps up to ±10 and gate to ≤10. One FP8 shared expert is always added.
  Routed experts are MXFP4 (I8-packed E2M1, `[out, in/2]`, e8m0 scale per 32 along K) with FP8 e4m3/ue8m0-per-32 activations. Ties: `torch.topk` documents no order, so
  parity compares the selected set and weights and fixtures reject k-boundary gaps below 1e-6.
- **Dense linears.** FP8 e4m3 weight with 32×32 block e8m0 scales (`weight_scale_inv` renamed `scale`), dynamic FP8 activations. `convert.py` dequantizes `wo_a` to bf16.
- **Engram (layers 1 and 14).** Compressed token map (NFKC → NFD → strip accents → lower → whitespace collapse), `engram_compressed_vocab_size = 99092`. Hash = XOR of
  `compressed_id × multiplier` over look-backs, modulo a prime per (n-gram order 2..4, head 0..7), plus a cumulative offset. Primes come from a `nextprime` chain
  starting at 16,000,000−1; multipliers from `np.random.default_rng(10007·layer)`. DEAD (image) tokens and the sequence start block look-back → pad id. Tables have
  `[384,006,168 | 384,016,682]` rows × 256 FP8 with `[rows, 8]` e8m0 scales. `wkv [25600, 6144]` FP8 → 4 keys (one per hc copy) + 1 value; gate =
  `sigmoid(copysign(sqrt(clamp(|dot|, 1e-6)), dot))` with per-copy RMS normalization.
- **DSpark (`mtp.0-2`).** Block 5, noise token 128799. `main_proj [5120, 15360]` + norm over the concatenated **inputs** of target layers 37, 38, 39 (mean over hc copies).
  Draft attention is SWA-only over target-derived `main_x` KV plus the block's own KV. `markov_head` embed/head `[129280,256]` adds a bias per sampled predecessor;
  `confidence_head [1, 5376]` on `[hidden; markov_embed]`; `forward_head` samples left to right at temperature.
- **Vision.** ViT 32 layers, dim 1024, 16 heads, 2D RoPE (32 = 16 per axis), biases on `wqkv`/`wo`, fused gate/up `w1 [5632,1024]`, RMSNorm eps 1e-6 in fp32,
  `patch_embed [1024, 588]`. Aligner: 3×3 unfold (pad to multiples of 3) → `w1 [5120, 9216]` → GELU → `w2 [5120, 5120]`. Preprocess: minimum 295936 px, pad to 14-multiples with
  gray 127, normalize (x−0.5)/0.5, at most 1024 tokens. Span layout `IMAGE_START + (IMAGE×w + NEWLINE)×h + IMAGE_END`; every span position carries id 129264; learned
  `image_start/end/newline` embeddings; image tokens use `bias_vl` routing and are DEAD for Engram.
- **Tokenizer.** HF byte-level BPE (128000 + 1283 added tokens); a three-stage `Split` pre-tokenizer (`\p{N}{1,3}`, CJK run, DeepSeek-V3 regex) + ByteLevel. `add_bos_token=false`
  (the encoder writes BOS as text). BOS 0, EOS 1, pad 2, image 129264. `HfTokenizerJson` builds a `PreTokenizerPipeline` that runs every `Split` stage in order (behavior and invert honoured) before ByteLevel; ids match HF `tokenizers` on the committed stress corpus (PR 10).
- **Conversation protocol** (`encoding/encoding.py`). Role markers `<｜System｜>`, `<｜User｜>`, `<｜Assistant｜>`, `<｜latest_reminder｜>`; `<think>`/`</think>` (chat mode writes
  `</think>` immediately); `Reasoning Effort: N` (1..100; low = 50, high = 75, max = 100) rendered only at index 0 in thinking mode; tools rendered as JSON schemas with the
  `<｜DSML｜ calls>` / `<｜DSML｜ invoke name=..>` / `<｜DSML｜ parameter name=.. string="true|false">` template (**leading space in tag names**, `namespace::tool` names); tool results are
  merged into the following user turn, sorted by call order; `drop_thinking` strips prior-turn reasoning unless tools are present; a mid-conversation `system` counts as a user turn
  for the generation header. `encoding.py` parses well-formed output only; streamed parsing is checked against the deepseek-recipe and vLLM parsers above.

## Storage

96,085 tensors, `total_size` 510,286,023,000 B (475.2 GiB) across 48 shards (`model.safetensors.index.json` SHA-256 in the manifest).

| Shards | Content | Size |
|---|---|---|
| 1 | vision + aligner | 0.97 GB |
| 2 | embed | 1.3 GB |
| 3–42 | one backbone layer each (experts 384 × 3 × (5.9 MB + 0.37 MB scale)) | ~7.39 GB |
| 43 | head + norm | 1.3 GB |
| 44–46 | `mtp.0-2` (128 experts each) | 2.6 GB |
| 47–48 | Engram tables (`[rows,256]` F8_E4M3 + `[rows,8]` F8_E8M0) | 101.5 GB each |

Dense weights per layer are about 170 MB (6.8 GB for 40 layers). One expert is about 17.7 MB (whole-slot upload ~18.8 MB with scales); the per-token routed working set is about 6 × 17.7 MB × 40 ≈ 4.2 GB.

## Quantization inventory

"All quant versions" means this versioned list of real encodings. Component-stripped derivatives are labelled separately, never folded into the official row.

| Family | Container | Encoding present | Other tensors | Native kernel |
|---|---|---|---|---|
| Official | safetensors ×48 | dense FP8 e4m3 + e8m0 32×32; experts MXFP4 (I8-packed) + e8m0/32; Engram FP8 + e8m0/32 | hc/gate/sink F32; norms/embed/head/vision BF16 | FP8-act × FP8-w block GEMM; FP8-act × FP4-w GEMM |
| NVIDIA NVFP4 | safetensors ×48 | experts U8-packed E2M1 + `weight_scale` E4M3/16 + `weight_scale_2` F32 + `input_scale` F32 | attention/shared/vision/Engram/mtp = official | W4A4 native on Blackwell only; elsewhere W4A16 (weights-only dequant, `input_scale` unused), a different numeric recipe with its own tolerance and label |
| AMD Quark | safetensors ×48 | experts + shared: U8-packed E2M1 + `weight_scale` U8 (e8m0)/32; attention FP8 + e8m0 32×32 renamed `weight_scale` | vision/gate/hc/Engram/mtp original | MXFP4 W4A4, MXFP8 |
| EXL3 | safetensors ×48 (branch) | experts `trellis` I16 + `suh/svh` F16 + `mcg` I32 (2 bpw); lm_head FP8 + row `weight_scale` U8 | others official | trellis decode (dequant) |
| MLX | safetensors ×79, odd naming | affine 4-bit gs64: `weight` U32, `scales`/`biases` F32, Engram included (lossy re-quantization) | gate/hc/norms | group-int4 dequant |
| DwarfStar GGUF | one GGUF + separate vision GGUF | Q2: IQ2_XXS gate/up + Q2_K down; Q4: Q4_K; Q8_0 attention/shared/head; F16 hc_fn/indexer/compressor/engram_kv; Engram I8 `[264, rows]` (256 fp8 + 8 e8m0) | **no `mtp`**; tokenizer `gpt2`/`joyai-llm` | existing GGUF codecs |

**Component-stripped or incomplete:** DwarfStar GGUF has no draft module and vision in a separate GGUF; its Q4 is two raw byte-parts to concatenate; it targets DwarfStar's
`deepseek41` architecture, not llama.cpp. MLX `mtp.2` lacks 13 experts (w1/w2/w3 for experts 9 and 88–99), so DSpark must be refused with the expert list.

**Format traps.**

- The `convert.py` `FP4_TABLE` is [0, 0.5, 1, 1.5, 2, 3, 4, 6] then negatives, with the low nibble as the even element. Milestone M1 (shard-3 dequant parity) must still confirm nibble order and that the
  scale is a multiplier (`w = q·2^(e−127)`) on real tensors before either is treated as fact.
- The engine's `ApplyFp8ScaledDequant` folds only scalar `.weight_scale`; a rank-2 scale is dropped silently and `weight_scale_inv` survives unused, so block-FP8 would load and run at scale 1.0.
- The reference `act_quant` uses an amax floor of 1e-4 and, with `scale_fmt`, a power-of-two ceiling taken from the fp32 exponent bits. `fp4_act_quant` floors amax at 6·2^-126 on the e8m0 path and 6·2^-9 on the e4m3 path (scale = e4m3(amax/6)).
- Engram constants (multipliers, primes, compressed token map) come from NumPy PCG64 and the Rust `unicode_normalization` pipeline. They are pinned as Python-dumped fixtures; .NET normalization is not assumed identical.
- The `tokenizer.json` CJK split regex uses classes from the HF regex crate; the .NET translation is proved against the dumped HF id list (`encoder_reference.json` stress corpus), and astral code points go through a same-category BMP proxy because .NET regex works on UTF-16.

## Reference gaps

Each gap stays open until its acceptance test exists and passes.

| Gap | What the official code gives | Resolve against | Acceptance |
|---|---|---|---|
| Chunked prefill | `model.py` handles only `start_pos == 0` full prefill or one-token decode; an unaligned chunk boundary is undefined for `Compressor` partial groups and `get_window_topk_idxs` | vLLM `compressor.py` (ring slots, fused save/norm) and `nvidia/model_state.py` (look-back, replay batch); SGLang dsv4 backend | one-shot vs chunked (aligned and unaligned) prefill give identical logits and state |
| SWA bounded replay and prefix restore | tech report §3.2.2 prose only | vLLM replay window in `nvidia/model_state.py`; define replay of the last 128 tokens under truncated SWA | restore + replay equals the documented approximation; measured drift recorded, not hidden |
| Images mid-sequence, multiple chunks | the reference asserts image spans live in the first prefill chunk | keep "a span is never split across chunks"; images at any position inside one chunk | multi-image fixtures at different positions pass; an explicit error otherwise |
| DSpark verify / accept / rollback | `forward_spec` only, no loop | DSpark paper §3.1–3.2, Alg. 1; SGLang `dspark_verify.py`, `dspark_accept.py`; vLLM `spec_decode/dspark/speculator.py` | T=0 speculation on/off byte-identical; T=1 acceptance statistics match the analytical rate on a synthetic model |
| STS calibration | not shipped in any checkpoint; SGLang fits offline | optional per-deployment file; default raw confidences; offline fitter | scheduler consumes calibrated or raw survival products; documented as an approximation |
| SPS(B) throughput profile | engine-specific | measure at load (decode step time vs verify batch size) | table exists and is logged |
| `fp4_gemm`, tilelang kernels | need Blackwell (`fp4_gemm`) or SM ≥ 8.9 (`fp8_gemm`) | local reference runs use bf16 dense and no fp4 experts with the pure-torch ports in `kernel_ports.py` | tilelang-vs-port agreement recorded from the rented run |
| Multi-image ordering, tool namespaces | `encoding.py` @ `dba1be0a` | five shipped fixtures plus our own multi-image, mid-system and task cases | exact token ids against HF `tokenizers` |
| DwarfStar numerics | own quant recipe (imatrix), not llama.cpp | DwarfStar `gguf-tools/deepseek41_*.py` layout; our codecs | tensor-wise dequant agreement with official dequantized weights within the format's error |
| EXL3 trellis decode | exllamav3 format (MCG codebook, K=2 trellis, `suh/svh`) | exllamav3 `quant/exl3_lib` as the format spec | lossless bounded-memory decode vs exllamav3's own dequant on a sampled expert |
| `weight_scale_inv` semantics, nibble order | inferred from `convert.py` | M1 on shard 3 | bit-exact 64-row dequant windows |

## Reference harness

`tests/python-reference/deepseek_v41/` — README there has the commands.

- `kernel_ports.py` replaces the tilelang `kernel.py` via `sys.modules['kernel']` so the **unmodified** upstream `model.py` runs on CPU. It ports `act_quant`, `fp4_act_quant`, `fp8_gemm`, `fp4_gemm`,
  `sparse_attn` (64-index blocks, running max seeded at −1e30, bf16 probabilities for P@V, sink in the denominator) and `hc_split_sinkhorn`. `test_kernel_ports.py` checks them against independent scalar or float64 implementations.
- `run_small_config.py` dumps intermediates for two small configs. `default` is upstream's small `ModelArgs` shape; `modes` keeps the real mode pattern in miniature (SWA-only, ratio-2 sources, ratio-1 candidate source with re-index and reuse layers, Engram on two layers, three draft layers).
  Weights are seeded random because upstream allocates with `torch.empty`; temperature 0. Runs are bitwise repeatable on one machine.
- `dump_engram_constants.py` dumps the token map, multipliers and primes with SHA-256 from the unmodified upstream `engram.py` and tokenizer. It asserts the compressed vocab is 99092 and per-layer prime sums equal `engram_num_embeddings`.
- Output is raw little-endian `.bin` plus `manifest.tsv` in `deepseek_v41_ref/` (gitignored), the same convention as the other reference dumps.

## Open questions

- Whether vLLM's chunk-boundary handling and the reference `Compressor` agree byte for byte is unread; PR 8 decides it against saved intermediates.
- Cold RAID latency for Engram row gathers decides whether a host-resident hot set is required (measured in PR 13).
- The checkpoint is not staged. The official 475 GiB download needs an explicit go-ahead; check `/mnt/model-storage/Models/llm/deepseek-v4.1-flash/` first.
