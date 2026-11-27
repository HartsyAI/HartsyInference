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
`__0_0.distcp` is one standalone `torch.save` ZIP per tensor, found at `(offset, length)` in `storage_data`. The released file has 635 tensors and 695,937,536 elements (read with this reader from the real `.metadata` of the 1.0 causal and 1.0 distill repos on 2026-11-24, both 635 tensors and 317 LoRA modules, all mapped; the guarded tests `HARTSY_DUO_DCP_DIR` repeat it).
Keys are `models.<family>.<module>.lora_{A,B}.default.weight` with the diffusers module names, plus an unused `role_embeddings.weight`. Upstream resolves the family from the manifest (`generator`); the first research pass read `student` for the base. `DuoLoraSource.FromDcp` takes the one prefix that carries LoRA tensors.

## Layout traps

- **Fused qkv.** Backbone shards store `blocks.N.attn.qkv_proj.weight [21504, 5376]` per head (`[head0: q k v, head1: q k v, ...]`, each 128 rows). The engine reads `[Q; K; V]` blocks, as upstream does after `reorder_interleaved_qkv`. Feeding the raw shards to `MiniMaxH3CheckpointConverter` loads without error and gives wrong output; `DuoQkvLayout` and `DuoRepacker.MergeBackbone` reorder.
- **Token refiner.** Upstream's `load_original` applies the qkv reorder to every key ending in `.attn.qkv_proj.weight`, so `token_refiner.blocks.N` is reordered the same way as `blocks.N` (`duo/models/realization/modules/weights.py` @068e357). The repack does the same, and a test covers it. A wrong `--heads`/`--head-dim` pair is refused before any shard is written, by checking the qkv rows and each block's `q_norm` length. A fused qkv bias in per-head order is refused, since only weights are reordered.
- **fc1.** Backbone `mlp.fc1` is `[gate; value]`; the LoRA's `ff.net.0.proj` B is `[value; gate]` (diffusers `SwiGLU`), so B's halves swap on merge.
- **Merge numerics.** Upstream merges `local.add_((B @ A * scale).to(local.dtype))`: delta in F32, rounded to the weight dtype, added in that dtype. Two BF16 roundings, nearest even. The engine's `Tensor.CastTo(BF16)` truncates, so the merge has its own rounding. Scale is `alpha / rank` per module (released: 128 / 128).
- **Module map.** `transformer_blocks.N.attn.to_{q,k,v}` to rows of `blocks.N.attn.qkv_proj`; `attn.to_out.0` to `attn.out_proj`; `ff.net.0.proj` to `mlp.fc1`; `ff.net.2` to `mlp.fc2`; `token_refiner.refiner_blocks.N.*` to `token_refiner.blocks.N.*`; `proj_in`, `audio_proj_in`, `context_embedder`, `proj_out`, `audio_proj_out` to `video_patch_proj`, `audio_patch_proj`, `condition_proj`, `final_layer.video_out`, `final_layer.audio_out`. Weights that are F32 in the backbone (patch projections, heads) merge without rounding.

## AdaLN tables

Modulation depends on the scalar time and the row's modality only (`index = time_index * 3 + tag`, tags video 0, text 1, audio 2). Upstream feeds unscaled `t` in `[0, 1]`, a cos-first sinusoid, the time embedder in F32, then `silu` and the BF16 `adaln_proj`. For a fixed schedule the 13.0B-parameter per-block `adaln_proj` layers reduce to `blocks.N.adaln_proj.table [times * 3, 6 * hidden]`; 5 times (distill) are about 97 MB, 51 (causal) about 1 GB in F32. `DuoAdalnPrecision.Bf16Reference` follows upstream's rounding, `F32` the engine's.
`DuoSchedule.Grid` is upstream's `grid` (`schedulers/flow.py` @068e357): `t = linspace(0, 1, points)` then `1 - shift * (1 - t) / (1 + (shift - 1) * (1 - t))`, in torch's default float32. Noisy blocks take `grid[step]` as their time and committed chunks 1.0 (`pipelines/inference.py`, `streaming.py`); the training layout uses a 0.999 placeholder for clean rows, which inference does not. AdaLN rows are `time_id * 3 + tag` into the distinct-times vector (clean slot first), so a consumer addresses a table row by step index, never by comparing floats.

## Engine recipe

`DuoRecipe` reads a release folder in the upstream bundle layout: `realization/dcp` (catalogued), `realization/backbone` (the 13 Ref2VA shards), `codec/video_vae/model.safetensors` and `codec/audio_vae/diffusion_pytorch_model.safetensors`. Only the LoRA is a catalog asset, so the other files are placed by hand, and the command line refuses a run whose release folder has no `realization/backbone` before offering any download; the distilled release ships its LoRA alone and takes the causal release's backbone and codec. A `realization/complete.json` naming the other stage (`causal` or `dmd`), a LoRA without that stage's keys (`models.student.` or `models.generator.`), a single-file backbone and a `DuoRepack merge` output are refused.

A generation takes `VideoRequest.Extra["DuoRunFolder"]`: `manifest.json` (`steps`, `shift`, `sharedText`, `referenceVideo`, and per block `bodyText`, `noiseVideo`, `noiseAudio`) naming little-endian float32 files. The distilled model (`duo-v1-distill`, 4 steps at shift 5) also needs per block `renoiseVideo` and `renoiseAudio`, each `steps - 1` draws in step order laid out like the initial noise; a causal run refuses them. Intent row counts come from the file lengths, the geometry from the request, and the frame count must be 17 times the block count. The prompt must be empty and the seed is unused. From the command line the two keys are `--duo-run-folder` and `--duo-window-frames`, with no prompt; `-m` is required:

    hartsy video -m duo-v1-causal --model-path <release> --duo-run-folder <run> --duo-window-frames 34 --frames 34 -o <out>

Over HTTP (`/v1/native/video/plan` and `/stream`) the run folder must exist under the models root, and a relative one is taken under it. A history larger than the device can still hand out is refused before the merge, naming the window key.

`Extra["DuoWindowFrames"]` sets the history window (default 144, the default of upstream's `--sliding-window-size` option, `data/inference_options.py`: seven clean blocks behind the current one; upstream's demo runtime passes 238, thirteen; -1 keeps all). Upstream facts the recipe follows (@068e357): the demo runtime sets the `new` RoPE policy (`inference_runtime/engine.py`, which also sets an `adaln_timestep` flag that no Python file there reads); each audio block is decoded alone and cut to `ceil(stop * 32000 / 24) - ceil(start * 32000 / 24)` samples (`serving/common.py::clock_block`). The latent mean and std of `video_vae.yaml` and `audio_vae.yaml` equal the engine's H3 defaults to F32 precision (compared 2026-11-26).

Memory: the load-time merge makes about 40 GB of host copies beside the mapped BF16 backbone (computed; the measured figure is below). The history is held in F32, about 2.9 MB per row, so the default window at 576x768 (about 16,000 rows) needs about 46 GB on the backend (computed, not measured).

## First run on released weights (2026-11-27)

JOB BL: `duo-v1-causal` on the `Duo-1.0-causal-base` release, `main` at `981965f36`, RTX 4090, host with 62 GiB of RAM (56 GiB available), the run's cgroup capped at 54 GiB, `vram.noAutopromote=true`, 320x384, 2 steps (the release runs 50), history window 34 frames. **The Intent rows, reference latent and noise were random draws, so the pictures mean nothing. This shows loading, merging, shapes, memory and time. It is not parity.**

| | A: 1 block | B: 2 blocks |
|---|---|---|
| Exit | 0 | 0 |
| Output | 17 PNG frames, 22,667 audio samples per channel | 34 PNG frames, 45,334 samples in each of 2 channels, mp4 muxed |
| Wall | 660 s | 942 s (`time -v` 15:42.66), budget 1800 s |
| Step intervals printed by the CLI | [2/2] 93,795 ms | [2/4] 93,652 ms, [3/4] 188,486 ms, [4/4] 93,652 ms |
| Peak host RSS | 56,564,256 kB (53.9 GiB) | 56,744,172 kB (54.1 GiB) |
| GPU-wide memory peak | 16,697 MiB, about 436 MiB of it SwarmUI's | 16,898 MiB |

Both merged 317 LoRA modules into 213 weights, loaded 560 video VAE and 1087 audio VAE tensors, held 842 history rows and produced no nonfinite latent. A logged the DiT as streamed (68.06 GB with its reserve against 13.99 GB free). The CLI prints the time since the previous step, so the first step has none, and B's [3/4] spans block 0's commit and decoder calls as well as block 1's first step (from the code). Load, merge and decode times are not logged. A first attempt at B, on a 754 s budget, was killed after 757 s in step 4 of 4, before any output was written. Evidence on the server: `/home/hartsy/evidence/peer-job-bl-2026-11-27/` (B in `outB/`).

Host memory, sampled every 3 s from start to exit of B: `RssAnon` peaked at 40,394,248 kB (38.5 GiB) and `RssFile` at 27,962,684 kB (26.7 GiB), not at the same moment; the cgroup peaked at its cap while page cache was reclaimed. A sample over only the last 61 s of the killed attempt read `RssAnon` 38.2 GiB, `RssFile` 9.0 GiB, cgroup anonymous 38.2 GiB and cgroup file 15.6 GiB. The anonymous peak (41.4 GB) agrees with the computed 40 GB of merge copies. Peak RSS is higher than that because it also counts the mapped checkpoint pages resident at the time, which the kernel can drop. B's peak reads 0.1 GiB over the cap; mapped pages that another cgroup had cached would count in RSS and not against the cap (a possible cause, not checked).

## Licence

All four upstream repos ship the MiniMax H3 Community License (`license_name: minimax-h3-community-license`, `LICENSE` 17,604 B, `NOTICE` 121 B). Binding points from the research pass, §II to §V and Exhibit A:

- Rights only outside the EU, UK, Republic of Korea and the United States, for use and for distribution (§II, §V.4, Exhibit A.1). Separate licences for those territories: `https://platform.minimax.io/h3-license`.
- Redistribution only to third parties inside the territory, with the licence and `NOTICE` text beside each file, a prominent notice on modified files, and no added terms. A repack or merge of Duo is a modified Model Derivative (§I.11).
- Recipients bound to the §V restrictions, including no use of the Works or their Outputs to improve another AI model (§V.3).
- Above USD 20M yearly revenue, written authorisation; a commercial product displays "MiniMax H3" on its UI (§IV). Public machine-generated content is disclosed as such (Exhibit A.12).

**Hosting terms: not recorded.** The owner holds a separate MiniMax licence that allows hosting a repack; its terms have not been given, so the defaults above stand and nothing is uploaded. Record the licence's territory, redistribution, notice and recipient terms here before any upload, and name the repo and files for each upload.

## Open

- The qkv order of the Comfy-Org H3 builds the engine runs (the `minimax-h3` catalog entry) is inferred from those runs working, not read from a file.
- A real-weight LoRA merge against upstream's merged tensors (plan level L2) needs the 2.78 GB LoRA and backbone shards; not run.
- Host memory, two options, neither built, figures computed and not measured:
  1. Precomputed AdaLN tables in place of the 26 GB of `adaln_proj` weights that every forward reads: about 1 GB in F32 for the causal schedule. `DuoRepacker.WriteAdalnTables` writes them; the transformer cannot read them. Merge copies plus tables would be about 41 GB, with almost nothing read from disk per forward.
  2. Merged weights on disk (a `DuoRepack merge` output, or a scratch file) in place of about 37 GiB of heap copies. Every forward would then read from disk unless about 66 GB of RAM were free for the page cache.
- Auto-promotion during a run: weights a run did not preload are freed after it (`471f605f5`), but a streamed run still promotes them while it runs, until about 1.5 GB of device memory is free (read from the code, not run). Real-weight runs pass `vram.noAutopromote=true`.
- The distilled and 1.2 releases have not run on released weights.
