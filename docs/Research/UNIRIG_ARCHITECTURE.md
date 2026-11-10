# UniRig — architecture spike

Source-backed map of upstream UniRig onto HartsyInference. Nothing here is checked against real weights yet; every
"verified" below means "read in the upstream source at the pinned commit".

## Pins

| Item | Value |
|---|---|
| Code | `VAST-AI-Research/UniRig` @ `6793c6640ff01c8fb389f3993434124bb43d2933` (2026-06-04, "Update README.md") |
| Weights | `VAST-AI/UniRig` @ `36842e2b5947e9e60f89275b83208c8e74071c63` (lastModified 2025-08-01), ungated |
| Skeleton ckpt | `skeleton/articulation-xl_quantization_256/model.ckpt`, 1,439,617,174 B, sha256 `d8cf9b42d56e7bc316d293597ecf8c4c39a8631cdd44a261ecf388242fabd0f4` |
| Skin ckpt | `skin/articulation-xl/model.ckpt`, 4,375,464,854 B, sha256 `9d40cf42fb9d4c10d8b373d9f4f557c6fbbc5ebacae7ae6b3dce5a0d8a18bc33` |
| Not used | `skeleton/rignet/model.ckpt` (621,660,032 B, RigNet-trained variant, different tokenizer config) |

Total for the two checkpoints: 5,815,082,028 B. Hashes are the Hugging Face LFS oids from the tree API.

## Licenses (read from the files)

- Code `LICENSE` and HF model card: **MIT** (both checked). Model card lists dataset `Seed3D/Articulation-XL2.0`, whose card says **CC-BY-4.0** (tag `license:cc-by-4.0`); attribution needed for the data, which only matters for the weights' provenance.
- `src/model/pointcept/LICENSE`: MIT (Pointcept).
- **`src/model/michelangelo/*.py` carry a GPL-3.0 header** ("derived from NeuralCarver/Michelangelo") and the folder ships a GPL-3.0 `LICENSE`. The repo-level MIT does not cover that folder. Consequence for this port: the C# Michelangelo encoder is written from the architecture (standard Perceiver cross-attention + residual blocks, equations below), not translated line by line from those files, and the PR says so. Whether weights trained with a GPL-derived module are affected is not something this port can settle; flagged for the maintainer.
- Upstream dependencies that the port replaces: Blender `bpy` (GPL), `fast_simplification`, `trimesh`, `torch_cluster`, `spconv`, `flash_attn`, `pyrender`/`open3d`.

## Upstream inference pipeline (as shipped)

1. `extract`: load mesh through Blender, apply `matrix_world`, triangulate polygons, `trimesh.Trimesh(process=True)` (vertex merge), `fast_simplification` to **50,000 faces** if larger (`--faces_target_count 50000`), vertex normals and unit face normals from trimesh.
2. Skeleton (`quick_inference_skeleton_articulationxl_ar_256`): normalize vertices into [-1,1], sample **65,536 points** (`mix`: 8,192 random mesh vertices + 65,536−8,192 area-weighted surface points; normals = vertex normal for vertex samples, face normal for surface samples), run AR model, detokenize, write `predict_skeleton.npz`.
3. Skin (`quick_inference_unirig_skin`): re-extract, normalize, sample **32,768 points** (8,192 vertices + surface), compute the `voxel_skin` prior on the mesh, run `UniRigSkin`, `reskin` back onto the real vertices, export FBX through Blender.
4. `merge`: transfers the result onto the original textured file.

### Corrections to the briefing numbers

- Latent count into the decoder is **1,024**, not 512. `query_method: False`, so the encoder takes 4×`token_num` = 4,096 random points (numpy `default_rng(seed=0)` in eval), runs torch_cluster FPS (`ratio=1/4`, `random_start=False` in eval) down to **1,024** query points, and those cross-attend to all 65,536 input points. `num_latents: 512` is unused on this path.
- The Michelangelo block here is the encoder only (`ShapeAsLatentPerceiverEncoder`, `no_query=True`); there is no KL head, decoder or geometry decoder.
- Decoder is OPT config with `hidden_size 1024`, `word_embed_proj_dim 1024` (so **no** `project_in/out`), `do_layer_norm_before True` (pre-norm; stock opt-350m is post-norm), other fields come from `facebook/opt-350m`'s config via `AutoConfig` (24 layers, 16 heads, ffn 4096, ReLU, biases on) — these are inherited, not stated in the repo, and must be confirmed from the checkpoint tensor shapes.
- Skin model needs a **geometric prior** (`voxel_skin`, grid 196) as an input feature; without it the network cannot run. This is the largest hidden dependency.

## Stage 1 — skeleton generation

### Conditioning (Michelangelo encoder, config `unirig_ar_350m_1024_81920_float32`)

- Input: `pc [N,3]`, `feats = normals [N,3]`, N = 65,536, both in the normalized cube.
- `FourierEmbedder(num_freqs=8, include_pi=False)`: frequencies `2^0..2^7`, output `[x, sin(x·f), cos(x·f)]` with per-axis grouping `x[...,None]*f → view(...,24)`, so dim = 3·(8·2+1) = 51; plus 3 normal channels → 54. `input_proj: Linear(54→512)`.
- Cross attention block (`ResidualCrossAttentionBlock`): `x + attn(ln_1(x), ln_2(data))`, `x + mlp(ln_3(x))`; width 512, 8 heads, `qkv_bias False`, `c_q`, `c_kv`, `c_proj` (bias on `c_proj`), MLP `c_fc 512→2048`, exact GELU, `c_proj 2048→512`. Queries = input_proj of the 1,024 FPS points; keys/values = input_proj of all 65,536 points.
- Self attention: 16 `ResidualAttentionBlock` (pre-LN, fused `c_qkv`, `qkv_bias False`, `c_proj`, MLP 4×, exact GELU), `ln_post` (`use_ln_post True`).
- `output_proj: Linear(512→1024)` yields the 1,024 conditioning embeddings.
- Weights are fp32 (config `dtype: float32`); attention scale `1/sqrt(head_dim)` (applied as sqrt on both sides upstream).
- **Not shared with Hunyuan3D**: Hunyuan's ShapeVAE uses a decoder-side cross-attention with different width/latents and a KL bottleneck; the two have not been compared weight-wise and this port does not reuse any of it. The `FourierEmbedder` here has `include_pi=False` and 8 freqs; Hunyuan's has 51-dim output with different freq set — different contract.

### Decoder (OPT)

- Sequence = `[cond(1024 rows) | embed(bos) | embed(cls)]` then tokens; `generate(inputs_embeds=...)`, positions 0..L−1 (+2 OPT offset).
- vocab = 267: `0..255` coordinate bins, `256 branch`, `257 bos`, `258 eos`, `259 pad`, `260 spring`, `261 body`, `262 hand`, `263 cls_none`, `264 vroid`, `265 mixamo`, `266 articulationxl` (computed in `TokenizerPart.__init__`). `assign_cls: articulationxl` ⇒ start tokens `[257, 266]`.
- Max positions 3,076 (`n_positions`), `max_new_tokens 2048`.

### Tokenizer / grammar (`tokenizer_part.py`)

- `discretize`: `clip(round((t+1)/2·256), 0, 255)`; `undiscretize`: `(t+0.5)/256·2−1`.
- Bone token layout: optional part token before bone `i` (`spring` or `body/hand`), then for a branch bone `branch, parent xyz(3), joint xyz(3)`, else `joint xyz(3)`. Sequence `bos, cls, ..., eos`.
- Detokenize: first bone parent = itself; a non-branch bone's parent position = previous joint; branch bone carries parent position; parent id = nearest earlier joint (`_get_parents`, reverse scan, strict `<`); tails: previous bone's tail = this joint when consecutive; leaf/branch tails extruded by `make_skeleton` (`extrude_scale 0.5`).
- Grammar (`next_posible_token`): state machine `expect_bos → expect_cls_or_part_or_joint → … expect_joint_2 → expect_joint_3 → expect_branch_or_part_or_joint`; EOS only allowed after a complete bone. Implemented as a hard −inf mask; unit-testable without weights.
- Bone names for `articulationxl`: `order.make_names` finds no part lists for that class (only `vroid`, `mixamo` have YAML files), so names are **`bone_0 … bone_{n-1}`**. (VRoid/Mixamo name lists exist but are unreachable with the shipped checkpoint's class token.)

### Generation parameters (`ar_inference_articulationxl`)

`max_new_tokens 2048, num_beams 15, do_sample True (beam sample), top_k 5, top_p 0.95, repetition_penalty 3.0, temperature 1.5, num_return_sequences 1`, seed 12345 in the launch scripts. Transformers pinned to 4.51.3. Beam-sample applies `log_softmax` first, then repetition penalty (over generated ids only), the grammar mask, temperature, top-k, top-p, then samples `2·beams` candidates from `softmax(scores + beam_scores)`. This is stochastic upstream, so parity with a Python run is a distribution claim, never a token-for-token one.

## Stage 2 — skinning

### Input features

- Mesh vertices normalized to [-1,1], 32,768 sampled points (+normals), skeleton joints/parents from stage 1 (the model uses only `joints`, `parents`, `num_bones`; tails are not read).
- `voxel_skin [N_sample, J]`: for every sample point, an inverse-distance-style prior from the shortest path on a graph made of mesh edges + interior voxel centers (grid **196**, `alpha 0.5`, `link_dis 1e-5`, `grid_query 7`, `vertex_query 1`, `grid_weight 3.0`, mode `square`: `(1/((1−α)d+αd²))²`, normalized over joints). The interior mask comes from six orthographic depth renders (`pyrender`, alternatively `open3d`) with "2-of-3 axes" voting; Dijkstra from each joint's nearest graph node. Needs a CPU rasterizer and a KD-tree/grid neighbor search in C#.

### Network (`unirig_skin.yaml`)

- `ptv3obj` encoder: `in_channels 9` (xyz, normal, 3 zeros), `grid_size 0.005`, depths `[3,3,3,6,16]`, channels `[32,64,128,256,384]`, heads `[2,4,8,16,24]`, patch 1024, `stride=()` (**no pooling**; stages are joined by `Linear`), `enable_qknorm True` (non-affine LayerNorm over head dim on q and k), `layer_norm False` (BatchNorm1d in the stem only), `res_linear True`, pre-norm, no RPE, no RoPE. Embedding = SubMConv3d k=5 (no bias) + BN + GELU + `Linear(9→32)` residual. Blocks = SubMConv3d k=3 CPE + Linear + LN → pre-norm attention → pre-norm MLP(×4, GELU). Serialization orders `z, z-trans, hilbert, hilbert-trans` per block `i % 4`, **randomly permuted at inference** (`shuffle_orders True`).
- `feat_map`: `Linear(384→768)+LN+GELU`.
- `global_encoder`: the same Michelangelo encoder as stage 1, but fed 32,768 points → 1,024 latents, `out_proj Linear(512→768)+LN+GELU`.
- `BoneEncoder`: `FrequencyPositionalEmbedding` (6 freqs, include π) on `joint − min_coord` → MLP(→1024 →4096 →768, LN+GELU) → 8 `ResidualCrossAttn` (post-norm: `norm1(q+attn)`, `norm2(x+ffn)`, 16 heads on 768, flash_attn `MHA(cross_attn=True)`, so `Wq`, `Wkv`, `out_proj`, default biases) over `[bone tokens | global latents]`.
- 16 `mesh_bone_attn` `ResidualCrossAttn`: mesh features query `[bone feats | global latents]`.
- Head: `qmesh`/`kmesh` `Linear(768→768·16)`; per-head `softmax(q·kᵀ/√768)` over bones → `attn_skin_norm` LN(16); `voxel_skin_embed Linear(1→16)+LN`; concat(32) → `downscale Linear(32→16)+LN+GELU`; `SkinweightPred` MLP(16→512 ×4 layers LN+GELU →1); `softmax` over bones.
- Post: `skin_mask` = `voxel_skin` with child mass added to parents; `skin = softmax_out · mask^3.0`, renormalized.
- `reskin` back to real vertices: 7 nearest sample points, **median**, one diffusion step along mesh edges (hot→cool, `alpha 2.0`, parent-summed), values below 0.03 zeroed if another channel is above, renormalize. FBX export keeps top-4 influences (`group_per_vertex 4`) and renormalizes.

## Mapping to existing C#

| Upstream stage | Existing C# | Gap |
|---|---|---|
| Mesh load | `SceneMeshReader` (obj/glb), `ObjReader`, `GlbReader` | no vertex welding, no degenerate-face removal, no decimation (PR 721 has `MeshDecimator`, not merged; not assumed) |
| Normalize to [-1,1] | — | trivial, new |
| Surface sampling | `SurfaceSampler` (only FPS) | area-weighted surface sampling + barycentric interpolation missing; FPS exists but is O(N·k) host code, deterministic from index 0 (matches `random_start=False`) |
| Point choice for FPS (numpy `default_rng(0).choice`) | `NumpyNormalGenerator` (normal only) | PCG64 `choice(replace=False)` missing; a different RNG gives a different but statistically equivalent subset |
| Michelangelo encoder | none (Hunyuan ShapeVAE is a different contract) | new module, `IBackend` ops for Linear/LN/GELU/SDPA exist |
| OPT decoder | `GenericTransformer` (`UseLayerNorm`, biases, `Relu`, `AbsolutePositionEmbeddings`, `ForwardEmbeds`), `PrefillChunk.Embeds` | key remap (`model.decoder.layers.N.self_attn.*` → host layout, `embed_positions.weight` needs the first 2 rows dropped for OPT's offset), per-layer LN bias on both pre-norms, `out_proj` bias, `lm_head` tied to embeddings? — all **unverified until the checkpoint is read** |
| Grammar | `ISamplerStep` / `SamplerChain` (steps: repetition penalty, temperature, top-k, top-p exist) | new `ISamplerStep` for the skeleton grammar; chain order differs from HF (HF applies `log_softmax` first under beam search); `SamplerChain` has no beam sampling |
| Beam sample (15 beams) | none | 15 beams × 24 layers × ≤3,076 positions of KV is ~9 GB in fp32 for the full config; plan starts with single-sequence sampling and states the deviation |
| PTv3 skin encoder | `UtoniaEncoder` | different: Utonia = Linear embedding, RoPE attention, pooling stages; UniRig = SubMConv k5 stem + BN, QK-norm, no pooling, no RoPE. Reusable pieces: `UtoniaSerialization` (z/hilbert codes), `UtoniaPatchPlan`, `SparseOps.SubmanifoldConv3dSparse` (needs k=5), `UtoniaLinear/LayerNorm/RowChunks`. New: QK-norm attention, BN fold, stage linears |
| Skin cross-attn stack, head | none | new |
| `voxel_skin` prior | `DualGridVoxelizer` (different purpose) | new: 6-view depth rasterizer, interior voxels, grid graph, Dijkstra, KD/spatial hash |
| `reskin` | none | new (KNN median + edge diffusion) |
| Rigged GLB | `HairCardGlbWriter` (HairDeck-prefixed records: skins, joints, IBMs, animation) | write a separate skin exporter; reuse of HairDeck's writer is not attempted because byte-identical HairDeck output cannot be proven without refactoring its record types |
| Checkpoints | `PytorchPickleLoader` (zip `.ckpt`), `PickleCheckpointRepacker` | both `.ckpt` files are Lightning checkpoints; nested `state_dict`, possible optimizer state in the 4.38 GB skin file; **not yet opened** |

## Checked against the real skeleton checkpoint (2026-11-09)

Read with the safe-subset loader: 585 tensors, all F32, root `state_dict.model.*`, no optimizer state, no quantized storage ("quantization" in the path is the 256-bin coordinate grid). Verified: OPT hidden 1024, 24 layers, 16 heads, FFN 4096, vocab 267, `embed_positions` `[3078, 1024]` (OPT's +2 offset; dropped at load), no `project_in/out`, separate `lm_head` `[267, 1024]`, per-layer `self_attn_layer_norm` + `final_layer_norm` (pre-norm layout), `q/k/v/out_proj` with biases; encoder `input_proj` `[512, 54]`, fused `c_qkv` `[1536, 512]` without bias, `c_kv` `[1024, 512]`, 16 resblocks, `ln_post`, `output_proj` `[1024, 512]`. The `GenericTransformer` host works with a key remap and no engine change (`AbsolutePositionEmbeddings`, `UseLayerNorm`, `Relu`, `FfnBias`, untied head).

Open items 1–3 below are answered for the skeleton file; the skin file is still unread.

## Open questions (need the real files)

1. Exact tensor names, shapes and dtypes in both `.ckpt` files (OPT dims, whether `lm_head` exists or is tied, BN running stats in the skin PTv3, `flash_attn` MHA parameter names, Michelangelo key prefixes).
2. Whether the pickle contains only plain tensors, or Lightning/OmegaConf objects the safe subset loader refuses. "Quantization" in the folder name is coordinate quantization (256 bins); weight storage dtype is read from the pickle.
3. Whether `SubMConv3d` weights are stored `[out,k,k,k,in]` (spconv) as `UtoniaBlock` expects.
4. Whether any Python run is feasible to dump references (none exist; no parity is claimed).

## Plan

1. PR 1: mesh prep (weld, normalize, sampling), Michelangelo encoder, tokenizer + grammar + detokenizer, OPT host, single-sequence generation, skeleton result type. Gated key-map test against the real skeleton checkpoint.
2. PR 2: PTv3 object encoder, skin heads, `voxel_skin`, `reskin`.
3. PR 3: rigged GLB exporter, `MeshModelIds`/`MeshService` dispatch, `hartsy 3d -m unirig`, catalog entry, docs.
