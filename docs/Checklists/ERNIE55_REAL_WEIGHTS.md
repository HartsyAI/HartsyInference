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

#928 lets each expert run as soon as its own weights are read. A third pod (`339aa8517`, same harness; the table's runs were
unprofiled) had a slower volume, 490 MB/s in one `dd` stream against 750 to 800 MB/s before, so its times are not comparable with the
table above; the arms within it are:

| Unprofiled run, 16 decode steps | Prefill | Decode, sum of the 16 step wall times | Steps 1 to 8 |
| --- | --- | --- | --- |
| 16 readers (first run on the pod) | 36.6 s | 68.5 s | 7.4, 8.6, 6.2, 4.1, 4.0, 6.0, 4.2, 3.5 s |
| 32 readers | 32.9 s | 61.8 s | 6.9, 6.5, 5.2, 3.7, 3.7, 5.3, 3.9, 3.3 s |
| 16 readers again | 36.1 s | 62.2 s | 7.1, 6.4, 5.3, 3.8, 3.9, 5.3, 4.1, 3.3 s |

32 readers match 16 on decode; the first run's extra time is the pod warming up, not the reader count. Prefill, which reads more experts
per layer, is about 9% faster with 32. The greedy tokens match in every run. A fourth, profiled run (16 readers, after the three above;
the stage timer's syncs add time, so it is not a table arm) puts 41.5 of 65 s of layer time in `moe.prefetch.wait`, so decode stays
bound by the volume's read speed. How much the overlap itself saves is within the volume's run-to-run variation and was not isolated;
running the commits before and after #928 back to back on one pod, after a warm-up run, would isolate it.

## What is not verified

- The indexer ReLU. The DeepGEMM `fp8_mqa_logits` kernel is not in the reference sources. The code takes the standard
  `sum_h w_h * ReLU(q_h . k)` form.
- Decode and MTP numerics on real weights. Only the 5-token prefill is compared against a reference (above).
- Production precision. CUDA GEMMs default to TF32 and Vulkan GEMMs round inputs to F16 (`EnableF16Gemm`), about 1e-3 relative
  to F32. `Ernie5RealCheckpointTests` pins the same two settings as the `reference` knob profile
  (`KnobProfiles.Reference`: `HighPrecisionGemm` on, `VkF16Gemm` off), so the verification run needs no extra configuration. Production
  defaults (TF32, F16 GEMM) are not measured by this harness.

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

- Decode speed: decode is bound by how fast the experts' storage reads (the reads overlap the expert loop since #928; 32 readers did not
  speed up decode on one pod, while prefill gained about 9%). A local disk, more page cache, or several GPUs holding the experts are the
  next steps.
- MTP speculative decoding in the Text path, and decode and MTP parity on real weights.
- A host or device memory guard before the load.
- License confirmation (EU/UK/South Korea exclusion, US$50M commercial threshold) for any use beyond this verification.
- Chat template: this base checkpoint ships none, so the Text path sends a plain completion prompt with the `tokenizer.json` post-processor's
  `<|ERNIE_MASK|>` first, as `/v1/completions` does.
