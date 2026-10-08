# Heterogeneous MoE execution: upstream research and provenance

Scope: the mechanisms behind Strata (https://github.com/Niko1221/Strata, MIT) and how each one maps onto HartsyInference's
generic primitives. Source snapshot: Strata `main` at commit `d5ea7133` (2026-10-07). This is a research record; the
implementation status of each mechanism lives in [ROADMAP](../Checklists/ROADMAP.md).

## Strata mechanisms, separated

Strata's results come from several independent mechanisms. Each one is tracked separately so the runtime can adopt
them one at a time.

| Mechanism | What it does | Generic or model-specific |
|---|---|---|
| Initial VRAM placement | Fills slots in profile order from a shipped ranking; no eviction in base mode | Generic (policy) |
| Adaptive residency tier | Usage decays ×0.7 per round; in-layer swaps when a candidate beats its victim by 1.5 | Generic (policy) |
| Miss execution | Misses run on the CPU concurrently with GPU hits; the GPU graph spin-waits on mapped-host flags | Generic (policy + protocol) |
| PCIe promotion | A `--pcie-frac` share of distinct misses is DMA'd into a 16-blob staging buffer and computed on the GPU | Generic (policy) |
| Prefill expert streaming | Next layer's non-resident experts stream through a pinned ring during attention | Generic (strategy) |
| KV streaming | K/V authoritative in pinned mapped host memory; VRAM holds 4-cell pages chosen by CLOCK | Generic (state tier) |
| MTP drafter | The model's own next-token head with its experts resident | Model-native |
| Suffix drafter | Trigram hash with 4 ways, up to 5 draft tokens | Generic (draft provider) |
| Multi-GPU helpers | Each expert has one owner; results return through pinned host buffers; no P2P in this mode | Generic (placement) |
| Peer tier | `cudaDeviceEnablePeerAccess` second adaptive tier | Generic (placement), hardware-gated |

Pinned upstream limits, kept here because the table rows are short: the adaptive tier swaps at most 96 experts per round, and
publishes its residency one window late; prefill streams in chunks of up to 8,192 tokens. A later implementation that
differs from these must say so.

## What is fundamental and what is specific

Fundamental (keep in generic code): a logical expert identity; a residency map separate from storage; policy objects
separate from storage; CPU and GPU executing disjoint subsets of one routed batch; explicit ownership of every
host/device buffer; pinned memory as a bounded budget with a working unpinned fallback.

Specific (keep in model code or per-model data): the Q2_0 block layout and its AVX-512 VNNI kernel; Qwen3.8-Flash-Next's
fixed dimensions (2560 hidden, 640 intermediate, 512 experts, top-10, 48 layers); its gated-DeltaNet and QSA attention;
its 28.8 GB n-gram table; the 1,382,400-byte blob size, which is a compile-time constant there.

## Provenance and licensing

- Strata itself is MIT. Its vendored `third_party/ggml/ggml-common.h` is ggml (MIT), pinned to a llama.cpp commit.
- ggml-cpu, ggml-cuda MMQ, `common.cuh` and `quantize.cuh` are compiled in for prefill.
- `src/kernels/cuda/iq_kernels.cu` transcribes ggml's `vecdotq.cuh`, `dequantize.cuh` and `quantize.cu`.
- `sycl/include/dpct/*` headers are Intel's, Apache-2.0 with LLVM exception.
- The control vector is under the Qwen Community License. That is a model license, separate from runtime licensing.

HartsyInference re-implements the mechanisms; it does not copy Strata source. Any future adaptation of ggml code keeps the
notice and records origin in the module header.

## Reference disagreements and gaps in Strata

Recorded because they affect how much weight the published numbers can carry.

- Strata does not quantize models. Its "conversion" repacks GGUFs that others quantized (ISTA-DASLab GSQ-RCO, Unsloth
  UD). No imatrix or calibration code exists in the repository; the calibration happened upstream.
- Its Q2_0 type is repository-specific. Strata's pinned `gguf-py` assigns it type id 42, which is not a mainline type.
  Hartsy keeps Q2_0 inside its own pack format until container compatibility is settled.
- Documents cited but absent from the repository: `docs/pack-format.md`, the C++ `tools/strata-pack`,
  `docs/kv-streaming-design.md`, `tools/routing_kfold.py`, `tools/mtp_probe.py`, `tools/verify_q2_0_geometry.py`,
  and `bench/micro/cpu_s2.cpp`.
- `SECOND_GPU.md` says the shipped profile ranks 8,000 pairs; the shipped file ranks 24,576.
- Performance is reported on one machine (RTX 5070 12 GB, Ryzen 5 7600, 64 GB DDR5-5200, Windows). The RTX 3090 numbers
  in the docs are estimates, not measurements.
- The Q2_0 encoder that Strata uses for its MTP layer is a minimum-MSE scale search. It is well below the GSQ-RCO quality of
  the released quants. Quality comparisons must use perplexity against those upstream quants.

## Positions taken for HartsyInference

- Expert placement is decided by the runtime from tier capabilities, never by model code.
- Routing semantics do not change with cache state. Cache policy changes where experts run, and output must match
  within the bounds in [MOE_ARCHITECTURE](../MOE_ARCHITECTURE.md#correctness-contract).
- The first heterogeneous test runs on an existing model with existing F32 weights. Quantized packs are not a dependency
  of that test.
- DeepSeek-V4.1 is the stress test for generality. Qwen3.8-Flash-Next is an optional comparison, not a design target.
