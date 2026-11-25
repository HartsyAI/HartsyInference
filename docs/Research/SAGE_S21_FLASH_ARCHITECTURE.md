# Sage S2.1 Flash (`glm5_next`)

Snapshot 2026-11-24. Huge MoE: no real-weight run is planned; the bar is tiny-config CPU parity and header-only loader tests.

## Pins and licence

| Item | Value |
|---|---|
| BF16 repo | `internlm/Sage-S2.1-Flash` at `eacc61b34a1af2ecb0c764e099152c46e076169b` (120 shards, `total_size` 642,653,338,488 B) |
| FP8 repo | `internlm/Sage-S2.1-Flash-FP8` at `5717cf2fb5c2a54e9c123334f78029d5eadc09cd` (62 shards, 328,333,456,248 B; E4M3, 128x128 blocks, F32 `weight_scale_inv`) |
| Paper | arXiv 2611.21072. Title "Sage S2.1 Flash Technical Report" in v1 and v2; the v1 abstract says "Sage S1.5 Flash" (typo), v2 says S2.1 |
| Reference code | transformers `src/transformers/models/glm5_next/` at `52bac5d0e0da56e26289c072f306877ff5870262` (modeling, configuration, `conversion_mapping.py` L763-808) |
| Licence | Card: "The code and model weights in this repository are released under the **MIT License**." No `LICENSE` file in the repo. The projector repo `internlm/Sage-SciPerception` says "License metadata has not yet been specified for this release" |
| Size | 320B total, 18B active; deployment guide: "uses the GLM-5.3-Flash architecture" |

## Decoder

45 layers + MTP layer 45, hidden 4096, vocab 154,880, RMS eps 1e-5. NoPE everywhere (`qk_rope_head_dim` 0).
- Layers 3,7,...,43 are DSA (MLA, 64 heads x 256, q rank 1536, kv rank 512, expanded form, no sink, no window); the other 34 are Kimi Delta Attention (64 heads x 128, conv 4, gate bound -5, low-rank forget and output gates, `A_log` per head). Config spells DSA `deepseek_sparse_attention` (transformers 5.16-5.18) or `indexed_attention` (5.19+).
- Indexer: 32 heads x 128, `index_topk` 2048, `index_kpool` 4. Keys are pooled by a softmax over the pool's gate scores plus `ape`; whole pools are scored (`relu(q.k)` weighted per head), the top `topk/kpool` pools expand to raw tokens, and the query's incomplete pool is appended as raw tokens. A pool is selectable only when its last token is visible.
- MLP: layers 0-2 dense (12288); 3-44 MoE (288 experts, top 8, width 2048, 1 shared, sigmoid + `e_score_correction_bias`, scaling 2.5, normalized top-k). `swiglu_limit` 10 clamps gate (max) and up (both sides) in dense, shared and routed paths.
- mHC (4 streams, 20 Sinkhorn rounds, `hc_eps` 1e-6) around every attention and MLP; the mix norm uses `rms_norm_eps`; the read-out is an unweighted mean over streams, then `model.norm`.
- MTP layer 45: `enorm`, `hnorm`, `eh_proj`, `shared_head.norm`, DSA + indexer, MoE, no hyper-connections. The reference ignores it, so there is no reference maths for it.
- Vision: 24 blocks, width 1024, 16 heads, patch 14, temporal patch 2, merge 2, out 4096, merger intermediate 10240.

## Checkpoint layout (traps)

- Per-expert `gate_proj`/`up_proj`/`down_proj`; the reference fuses `gate_up_proj` with gate first, concatenating on dim 1.
- Separate `q/k/v_conv1d.weight` [8192,1,4] F32; the reference concatenates them q, k, v. `f_a_proj`, `f_b_proj`, `dt_bias`, `A_log` sit directly under `self_attn` (the reference nests them under `forget_gate`).
- `hc_{attn,ffn}_{fn,base,scale}` at the layer root (`fn` BF16 [24,16384], `base` F32 [24], `scale` F32 [3]).
- FP8 repo: quantized are dense/shared/routed FFN weights and the DSA `q_a_proj`, `q_b_proj`, `kv_a_proj_with_mqa`, `o_proj`. KDA projections, `kv_b_proj`, indexer, router, embedding, LM head, vision and hyper-connections stay BF16/F32.
- Layer 45 owns 288 experts: 43 MoE layers x 288 = 12,384 keys per projection.
- Tokenizer: control-token spellings inside message content must encode as ordinary text (`tokenization_glm5.py`); only template-written controls keep their ids.

## Scientific inputs

The scientific encoders are not part of the checkpoint. `Sage-SciPerception` ships trained projectors (feature rows in, 4096-wide rows out) and says encoder weights and preprocessing are separate dependencies. The Flash repo carries 8 projectors under `projectors/sage-s2.1-flash/` (CIF/CHGNet 64-d, DNA readout 3-d, ECG 768-d, EEG/REVE 512-d, protein sequence/ESM-2 1280-d, protein structure/ProteinMPNN 384-d, RNA 640-d, spectra patches 50-d). All are `AtomProjector` MLPs (`modeling_projector.py` at SciPerception `7c228fc03394027fed19bda1ff26388394b0031a`). Output rows replace the placeholder span of the templated prompt embeddings.

## Evidence

`Glm5NextWeightCoverageTests` proves every key of both real index key sets (38,770 BF16, 76,108 FP8; fixtures under `tests/HartsyInference.LLM.Tests/Fixtures/Glm5Next`) is named by `Glm5NextWeightNames`, and that FP8 differs only by the scale companions `IsFp8Weight` predicts. Shapes come from header reads of shards 1, 11 and 120 (BF16) and shards 1 and 6 (FP8) and are checked by the loader, not here.
