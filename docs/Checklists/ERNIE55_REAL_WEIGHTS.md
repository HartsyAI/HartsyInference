# ERNIE-5.5-Flash-Base real-weight run

Checkpoint: `baidu/ERNIE-5.5-Flash-Base` at revision `028544b2ffceea189b2d8768ed2b9d09b06f2343`. FP8 e4m3 with 128x128 F32
`weight_scale_inv` grids, 128 shards, 288,434,400,632 bytes. Base model: no chat template in the repo, so the harness test below feeds
token ids and the Text path sends a plain completion prompt.

## What is verified without the weights

- Config passes `Ernie5Config.UnimplementedFeatures` (the loader gate).
- The real `model.safetensors.index.json` maps every one of its 134,231 keys (`Real_IndexMapsEveryKey`).
- One expert and one MLP scale tensor are F32 powers of two (2^-12 and up), so the stored scales are UE8M0-valued and
  multiply the E4M3 weights (the convention `weight_scale_inv` names).
- Multimax matches `Ernie5_5ParallelLMHead.multimax` in vLLM-ERNIE term for term.
- The indexer's weight scale `head_dim^-0.5 * n_head^-0.5` matches the reference.
- Real dimensions, FP8 storage, BF16 routers/LM head/compressor/indexer projections, gated and VHA attention, and split-feature
  routing: CUDA and Vulkan match the CPU (`Cuda_/Vulkan_RealDimensionsFp8_LogitsMatchCpu`).

## Real-weight runs (CUDA)

An earlier run compared prefill numerics: every decoder layer's residual stream matched a CPU torch reference at cosine >= 0.999 for a
5-token prefill (`tools/ernie5-reference`). It is not repeated here.

Decode parity (2026-11-16, RTX 4090, `580cd2a7d`, reference precision): the 5-token prompt `1,2,3,4,5`, then 8 fixed decode tokens
`2551,7479,2811,1230,2810,4357,1298,36326`, one forward each through the KV caches, against the CPU torch reference on the same 13
tokens. All 43 layers reach cosine >= 0.999 at every position (printed 1.00000000 everywhere; relative error 1.6e-6 to 9.1e-6), and the
argmax after the last position is 1232 on both sides. That covers the HCA and DSA KV caches, RoPE at later positions and the one-token MoE
path. Commands and the full result are in `tools/ernie5-reference/README.md` (Decode parity).

The Text-service run below used the CUDA tests directly, not `scripts/ernie55_pod_run.sh`, so the runbook's size and SHA check and its
`Real_PrefillAndDecode_ProduceFiniteLogits` timing line were not run. RunPod EU-RO-1, image `kalebbroo/hartsy-gpu-runner:cuda13.0-ssh`,
RTX 4090 (24 GB, sm_89), driver 595.91.07, CUDA 13.2, 251 GB host RAM, checkpoint on the network volume, repo at `1779786fb`. Host RSS
was not measured.

- Tokenizer: `Real_CompletionPrompt_MatchesTheHfTokenizer` matches the Hugging Face `tokenizers` ids on all 12 cases from
  `tests/python-reference/dump_ernie55_tokenizer_cases.py`.
- GPU parity: all 8 `Ernie5ModelGpuParityTests` CUDA tests pass (F32 and FP8), including `Cuda_TextService_Fp8_LoadsOnTheDeviceAndMatchesCpu`.
- Generation: `Real_TextService_GeneratesFromAPlainPrompt` (greedy, 16 tokens, stop reason Length) continues "Large language models are"
  (5 prompt ids) with " a type of artificial intelligence that can generate human-like text. They are trained on".

Timings before the routed-expert prefetch (#926); the next section has the comparison.

| Stage | Time (before #926) |
| --- | --- |
| Load (dense preload 7644 MB of a 20025 MB budget, 48.5 s of it) | 140.5 s |
| Prefill, 5 tokens | 174.5 s |
| Decode, 15 steps | 208.8 s (13.9 s per token, 5.5 to 17.8 s) |
| Total with load (measured; includes about 1 s outside the stages above) | 525 s |

Peak VRAM was about 22.5 GB; the device weight total grew from 9216 MB to over 20 GB during decode. The Text service attaches no expert
cache, so this is the CUDA backend's auto-promotion: a routed expert uploaded a second time is kept on the device while free memory stays
above `vram.autopromoteHeadroomMb` (default 1536 MB, which matches the 22.5 GB peak), and after that it uploads on every use. The growth
stops at that floor, not at the preload budget. Promoted weights are not evicted under memory pressure, so a longer prompt whose
activations exceed the floor can still run out of device memory.

### Decode profile and the expert prefetch

`Real_PrefillAndDecode_ProduceFiniteLogits` with `ERNIE_REAL_RESIDENT=1 ERNIE_REAL_DECODE_STEPS=16 ERNIE_REAL_PROFILE=<dir>` (reference
precision, tokens `1,2,3,4,5`), same pod type, before (`564d824da`) and after (`f1b226734`) the routed-expert prefetch of #926. The pod's
container memory is capped at 46 GB, so the page cache cannot hold the experts between steps. The volume reads about 800 MB/s in one
sequential `dd` stream but about 95 MB/s from one thread in 4 MiB reads; 8 to 30 parallel 4 MiB readers measured 550 to 790 MB/s while
another run shared the volume.

ERNIE takes the host-routed per-expert MoE loop (split-feature routing and SiTU keep it off the device-routed path). Before the prefetch,
routed experts were 168.5 of 171 s of the 16 decode steps, almost all of it 9,576 serial expert uploads at 17.2 ms each; attention, the
router, the shared expert and the latent projections were under 1 s together. A warm rerun with the same tokens was only 8% faster.

| Cold run | Before | With the prefetch |
| --- | --- | --- |
| Prefill, 5 tokens | 172.7 s | 36.2 s |
| Decode, 16 steps (layer time) | 171.2 s | 49.3 s |
| Decode steps 1 to 8 | 22.7, 20.4, 16.4, 10.1, 10.6, 17.7, 11.2, 8.1 s | 5.9, 5.3, 4.6, 3.1, 3.1, 4.4, 3.2, 2.6 s |
| Decode steps 15 and 16 | 2.9, 2.7 s | 1.3, 1.2 s |
| Expert upload (`H2D_BIG` average) | 17.2 ms | 0.52 ms |

The greedy tokens are identical. The tokens repeat from step 9, so the late steps reuse experts already on the device or in the page
cache; compare position by position, not the 16-step mean. With the prefetch, `moe.prefetch` is 38.6 of the 49.3 s: about 37 GiB read at
about 1 GB/s with 16 readers, the highest rate seen on this volume (the separate reader tests ran while another run shared it, and some
late repeated-token reads hit the page cache), so decode is now bound by its read speed. In #926 the prefetch ran before each layer's
expert loop, so its reads and the uploads and GEMMs did not overlap.

#928 lets each expert run as soon as its own weights are read. A third pod (`339aa8517`, same harness) had a slower
volume, 490 MB/s in one `dd` stream against 750 to 800 MB/s before, so its times are not comparable with the table above; the arms
within it are:

| Unprofiled run, 16 decode steps | Prefill | Decode, sum of the 16 step wall times | Steps 1 to 8 |
| --- | --- | --- | --- |
| 16 readers (first run on the pod) | 36.6 s | 68.5 s | 7.4, 8.6, 6.2, 4.1, 4.0, 6.0, 4.2, 3.5 s |
| 32 readers | 32.9 s | 61.8 s | 6.9, 6.5, 5.2, 3.7, 3.7, 5.3, 3.9, 3.3 s |
| 16 readers again | 36.1 s | 62.2 s | 7.1, 6.4, 5.3, 3.8, 3.9, 5.3, 4.1, 3.3 s |

32 readers match 16 on decode; the first run's extra time is the pod warming up, not the reader count. Prefill is about
9% faster with 32, likely because its 5 tokens select more experts per layer. The greedy tokens match in every run. A fourth, profiled run (16 readers, after the three above;
the stage timer's syncs add time, so it is not a table arm) puts 41.5 of 65 s of layer time in `moe.prefetch.wait`, so decode stays
bound by the volume's read speed. How much the overlap itself saves is within the volume's run-to-run variation and was not isolated;
running the commits before and after #928 back to back on one pod, after a warm-up run, would isolate it.

## Long-input parity and production precision (real weights)

2026-11-16, RTX 4090 (24 GB), driver 595.91.07, CUDA 13.2, `main` at `2317bb2e3`; the commands are in
`tools/ernie5-reference/README.md` (Long-input parity). Prompt: 2,300 ids, plus the 8-token decode feed (2,308 positions), so every HCA
layer completes 18 compressor groups and every DSA query from position 2,176 selects 2,048 keys.

- Engine (CUDA, reference precision): load 138.7 s, prefill of 2,300 tokens 512.1 s, decode steps 4.5 to 10.1 s.
- Reference (CPU): 1,621 s on 48 vCPUs. It was rerun at `f58204e00` on a 16-vCPU CPU pod (3,746 s) to record the top-k gap files.
- Layers 0 to 6, the compressor included, reach cosine 1.00000000 at every position. The argmax after the last position is 1232 on
  both sides. The whole-stream cosine is at least 0.99996 in every layer.
- From layer 7 on, 22 of the 2,308 positions fall below cosine 0.999 in some layer. The lowest is 0.924, at position 2062 in layer 42,
  the last layer, whose activations are largest. `compare.py --explain` traces them to one MoE routing near tie: at layer 7, position
  1919's last routed and first unrouted experts are 1.3e-6 of the routing-score spread apart (the smallest 0.1% of that layer).
- Median relative error per position band, engine against reference (a subset of the 9 bands and 14 layers computed):

  | Layer | 0-127 | 512-1023 | 1536-1791 | 1792-1919 | 1920-2047 | 2048-2175 | 2176-2307 |
  |---|---|---|---|---|---|---|---|
  | 0 | 4.5e-6 | 4.3e-6 | 4.8e-6 | 4.6e-6 | 4.3e-6 | 4.6e-6 | 4.8e-6 |
  | 7 | 5.1e-6 | 5.3e-6 | 6.7e-6 | 6.8e-6 | 6.7e-6 | 6.8e-6 | 7.3e-6 |
  | 8 | 5.1e-6 | 5.1e-6 | 6.7e-6 | 6.7e-6 | 4.7e-4 | 9.6e-5 | 8.6e-5 |
  | 22 | 7.6e-6 | 6.8e-6 | 7.3e-6 | 7.2e-6 | 1.5e-3 | 1.3e-3 | 4.0e-4 |
  | 28 | 6.0e-6 | 5.8e-6 | 7.2e-6 | 9.8e-6 | 1.9e-3 | 1.6e-3 | 8.2e-4 |
  | 35 | 4.9e-6 | 4.5e-6 | 6.1e-6 | 2.3e-5 | 1.9e-3 | 1.7e-3 | 1.0e-3 |

  - **Before the tie:** up to layer 7 the error stays within 1.5x across all 9 bands (a slow rise with length, 5.1e-6 to 7.3e-6 at
    layer 7), so nothing breaks with length.
  - **After the tie:** from layer 8, every position after 1919 reads that position's changed state through attention, and those
    positions jump to about 9e-5 to 2e-3. Positions before 1920 stay below 1e-5 through layer 28. At layer 35 the 1792-1919 band rises to
    2.3e-5; causal attention means position 1919 can't reach them, so something else does. That rise is not traced. It may be
    smaller routing flips that stay above cosine 0.999, which `--explain` does not list.
  - **Later dips:** that small extra error then tips a few more routing near ties after position 1919. That is why every dip is at
    1919 or later. 13 of the 22 first drops have a router gap within the smallest 5.1% of their layer, at the drop's layer or the one
    before. The other 9 all come after earlier ties, which fits propagation.
  - **Indexer selection:** positions 2,176 and later show no error beyond what that band already inherits.
- **Indexer near ties are common at this scale.** 76 of the 132 selecting positions have a kept/dropped gap below 1e-4 of the spread,
  so `--tie-margin` refuses (`--max-tie-fraction`); read `--explain` instead.
- **Production precision** (`ERNIE_REAL_PRECISION=production`, TF32 GEMMs, #935), on the 13-position decode parity: all 43 layers reach
  cosine 0.9999989 or more at every single position (the whole-stream relative error per layer is 9e-5 to 5e-4), and the argmax
  after the last position is 1232 on both sides. Without `ERNIE_REAL_PRECISION` the harness runs the `reference` GEMM pins, so every
  other run above needs no extra configuration.

## What is not verified

- The indexer ReLU. The DeepGEMM `fp8_mqa_logits` kernel is not in the reference sources. The code takes the standard
  `sum_h w_h * ReLU(q_h . k)` form.
- Inputs longer than 2,308 tokens. The run above covers the compressor and the indexer selection up to that length. It doesn't
  cover long contexts such as 32K or 128K.
- The compressor's storage precision. The DeepSeek-V4 source stores latents as bf16 with FP8 non-rope dims; the reference and the
  engine's host path keep F32. The bf16 output of vLLM's Hadamard kernel is not simulated either.
- `csa_dense_mode: true` in the config. Neither the engine nor the vLLM-ERNIE files read for this port (`config.py`,
  `nvidia/attention.py`, `model.py`, `moe.py`, `mtp.py`) read it.
- MTP numerics on real weights.
- Production precision on Vulkan (F16 GEMM inputs), and at long input. CUDA's TF32 default is measured on the 13-position decode parity
  above; attention and KV-cache numerics keep their defaults in both arms.

## Run

Needs the full checkpoint in one directory (`config.json`, `model.safetensors.index.json` and the 128 shards), a CUDA GPU with at least
24 GB free, and the PTX built. The model does not fit in VRAM, so each forward streams the layers from the volume and uploads
their packed FP8 weights per call.

```bash
ERNIE_REAL_DIR=/path/to/ERNIE-5.5-Flash-Base \
ERNIE_REAL_DEVICE=cuda \
ERNIE_REAL_TOKENS=1,2,3,4,5 \
dotnet test tests/HartsyInference.Engine.Tests --filter "FullyQualifiedName~Ernie5RealCheckpointTests" --logger "console;verbosity=normal"
```

Expected: `Real_IndexMapsEveryKey` passes, and `Real_PrefillAndDecode_ProduceFiniteLogits` passes with finite, non-constant logits and
prints load, prefill and decode times. Two forwards read about 270 GiB each from the volume, so budget on the order of tens of minutes
of I/O on a network volume (an estimate, not a measurement).

For a rented pod (volume size, the pinned download, the GPU and PTX checks, and the PR record), follow
[ERNIE55_POD_RUNBOOK.md](ERNIE55_POD_RUNBOOK.md); `scripts/ernie55_pod_run.sh` runs its steps in order.

## Open

- Decode speed: on one GPU, decode is bound by how fast the experts' storage reads (the reads overlap the expert loop since #928; 32 readers
  did not speed up decode on one pod, while prefill gained about 9%). The layers can now be split over several GPUs with every weight,
  routed experts included, preloaded on its stage (`Ernie5LoadOptions.Placement` and `PreloadExperts`; the Text path does this for a
  `cuda:0+cuda:1` key or the engine's shard devices, and the harness takes `ERNIE_REAL_STAGES`). That takes the storage reads off the decode
  path. The stages run one after another, so it speeds up one sequence, not several at once.
- Four-stage run (2026-11-20, 4 x RTX PRO 6000, EU-RO-1, every weight preloaded): decode 0.21 to 0.24 s per token (mean per run) at reference and at
  production precision; a 2,300-token prefill 127 to 131 s (about 2.4 s per layer, 5.7 s for every seventh); load 2,146 to 2,162 s, the
  four stages one after another at about 128 MB/s each. No per-op profile and no Python speed baseline exist for this run.
  - Load: each upload staged its mapped bytes through one reader. The stage preload now reads each layer's weights ahead with
    `numerics.moeExpertPrefetchThreads` readers (`Ernie5Model.PreloadGroups`). Not measured on the volume.
  - Decode and prefill run the host-routed per-expert MoE loop (512 experts, 10 per token) and keep the residual stream on the host, so
    every op uploads and downloads its operands. The hyper-connection projection ran as a scalar host dot, 86 times per token; it is now
    vectorized (`HostMath`). A local one-token microbenchmark (8-core desktop, AVX2) went from 0.44 ms to 0.04 ms per call; the pod's
    time per token was not remeasured. A device-resident stream and a device-routed expert path for split-feature routing and SiTU are
    open.
  - Greedy decode at production precision after the synthetic 2,300-id prompt (`1000 + 37 i mod 90000`, mostly `<ERNIE_mask:N>` ids)
    repeats tab, `}`, tab (1400, 1308, 1400). The prefill argmax, 1423, is the same at both precisions, and the last-position argmax
    matched the reference on the same prompt with the 8-token feed. No reference-precision greedy run and no reference logits exist for
    the generated positions, so this is not confirmed as model behaviour: run the reference over the prompt plus the generated ids with
    `--logits` and compare `choices.json` (`tools/ernie5-reference/README.md`).
- MTP speculative decoding in the Text path, and MTP parity on real weights.
- A host or device memory guard before the load.
- License confirmation (EU/UK/South Korea exclusion, US$50M commercial threshold) for any use beyond this verification.
- Chat template: this base checkpoint ships none, so the Text path sends a plain completion prompt with the `tokenizer.json` post-processor's
  `<|ERNIE_MASK|>` first, as `/v1/completions` does.
