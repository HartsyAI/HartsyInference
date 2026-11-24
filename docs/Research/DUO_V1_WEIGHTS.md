# Duo v1 weights: pins, formats, layout traps, licence

inclusionAI Duo v1 is MiniMax-H3's Ref2VA video+audio DiT plus a rank-128 LoRA. This page keeps the weight-side facts the code cannot recover; the code is `ModelAssets/Dcp` and `ModelAssets/Duo`. Source snapshot 2026-11-23, upstream code `inclusionAI/Duo` @ `068e357ede2bdea233b408645301e522f560da73`.

## Pins and Duo-specific files

Each repo ships the same backbone, Intent base, codecs and ASR; only the realization LoRA differs. The LFS hashes are the repos' own, read from the HF API listing at the pinned revision.

| Repo @ revision | `realization/dcp/__0_0.distcp` sha256 (2,784,751,539 B) | `.metadata` sha256 |
|---|---|---|
| `inclusionAI/Duo-1.0-causal-base` @ `345945e9` | `36a14c91142a15321d0d1e40257afb8ab4ddf1c5eddbb8723eca18d83d0620e1` | `e0a1e8802adb2fde9ddb6231056fca479a683014cad387ca39f8fc4f8f6cfd0e` (206,259 B) |
| `inclusionAI/Duo-1.0-distill` @ `2ac57adb` | `27b0188a15d7d52872db0005022c9737c3b128fdde9df360237aa0798a165bd4` | `4ca08a328ea6ecc19490bd16605164d07dd9d8dcd7d3b0774c4d9358349be10a` (207,733 B) |
| `inclusionAI/Duo-1.2-causal-base` @ `d65767bc` | `05332963e83056c3b3352329fdd55e84fb46881170f7889ddebf7ce7eede88a7` | same as 1.0 causal |

Stage defaults: causal `grid(51, shift 1.0)`, distill `grid(5, shift 5.0)`.

## Distributed checkpoint format

`.metadata` is a pickle of dataclasses (`Metadata`, `TensorStorageMetadata`, `ChunkStorageMetadata`, `MetadataIndex`, `_StorageInfo`); upstream's README warns it is unpickled before any check. `DcpMetadataUnpickler` interprets it with an allow-list and never calls anything.
`__0_0.distcp` is one standalone `torch.save` ZIP per tensor, found at `(offset, length)` in `storage_data`. The released file has 635 tensors and 695,937,536 elements (**[P]** from the plan's earlier parse; the reader pins it in a guarded test, `HARTSY_DUO_DCP_DIR`).
Keys are `models.<family>.<module>.lora_{A,B}.default.weight` with the diffusers module names, plus an unused `role_embeddings.weight`. Upstream resolves the family from the manifest (`generator`); the first research pass read `student` for the base. `DuoLoraSource.FromDcp` takes the one prefix that carries LoRA tensors.

## Layout traps

- **Fused qkv.** Backbone shards store `blocks.N.attn.qkv_proj.weight [21504, 5376]` per head (`[head0: q k v, head1: q k v, ...]`, each 128 rows). The engine reads `[Q; K; V]` blocks, as upstream does after `reorder_interleaved_qkv`. Feeding the raw shards to `MiniMaxH3CheckpointConverter` loads without error and gives wrong output; `DuoQkvLayout` and `DuoRepacker.MergeBackbone` reorder.
- **fc1.** Backbone `mlp.fc1` is `[gate; value]`; the LoRA's `ff.net.0.proj` B is `[value; gate]` (diffusers `SwiGLU`), so B's halves swap on merge.
- **Merge numerics.** Upstream merges `local.add_((B @ A * scale).to(local.dtype))`: delta in F32, rounded to the weight dtype, added in that dtype. Two BF16 roundings, nearest even. The engine's `Tensor.CastTo(BF16)` truncates, so the merge has its own rounding. Scale is `alpha / rank` per module (released: 128 / 128).
- **Module map.** `transformer_blocks.N.attn.to_{q,k,v}` to rows of `blocks.N.attn.qkv_proj`; `attn.to_out.0` to `attn.out_proj`; `ff.net.0.proj` to `mlp.fc1`; `ff.net.2` to `mlp.fc2`; `token_refiner.refiner_blocks.N.*` to `token_refiner.blocks.N.*`; `proj_in`, `audio_proj_in`, `context_embedder`, `proj_out`, `audio_proj_out` to `video_patch_proj`, `audio_patch_proj`, `condition_proj`, `final_layer.video_out`, `final_layer.audio_out`. Weights that are F32 in the backbone (patch projections, heads) merge without rounding.

## AdaLN tables

Modulation depends on the scalar time and the row's modality only (`index = time_index * 3 + tag`, tags video 0, text 1, audio 2). Upstream feeds unscaled `t` in `[0, 1]`, a cos-first sinusoid, the time embedder in F32, then `silu` and the BF16 `adaln_proj`. For a fixed schedule the 13.0B-parameter per-block `adaln_proj` layers reduce to `blocks.N.adaln_proj.table [times * 3, 6 * hidden]`; 5 times (distill) are about 97 MB, 51 (causal) about 1 GB in F32. `DuoAdalnPrecision.Bf16Reference` follows upstream's rounding, `F32` the engine's.
The schedule formula in `DuoSchedule` (`sigma' = shift * sigma / (1 + (shift - 1) * sigma)`, `sigma = 1 - t`) reproduces the distinct distill times the research derived; it is not yet compared with upstream's `grid()`.

## Licence

All four upstream repos ship the MiniMax H3 Community License (`license_name: minimax-h3-community-license`, `LICENSE` 17,604 B, `NOTICE` 121 B). Binding points from the research pass, §II to §V and Exhibit A:

- Rights only outside the EU, UK, Republic of Korea and the United States, for use and for distribution (§II, §V.4, Exhibit A.1). Separate licences for those territories: `https://platform.minimax.io/h3-license`.
- Redistribution only to third parties inside the territory, with the licence and `NOTICE` text beside each file, a prominent notice on modified files, and no added terms. A repack or merge of Duo is a modified Model Derivative (§I.11).
- Recipients bound to the §V restrictions, including no use of the Works or their Outputs to improve another AI model (§V.3).
- Above USD 20M yearly revenue, written authorisation; a commercial product displays "MiniMax H3" on its UI (§IV). Public machine-generated content is disclosed as such (Exhibit A.12).

**Hosting terms: not recorded.** The owner holds a separate MiniMax licence that allows hosting a repack; its terms have not been given, so the defaults above stand and nothing is uploaded. Record the licence's territory, redistribution, notice and recipient terms here before any upload, and name the repo and files for each upload.

## Open

- The real `.metadata` has not been read; the unpickler is tested on synthetic metadata that mirrors torch's dataclass pickling.
- Upstream `grid()` and its dtype (schedule check above).
- Whether the Comfy-Org H3 builds are `[Q; K; V]`-ordered is inferred from their verified runs, not from a file.
