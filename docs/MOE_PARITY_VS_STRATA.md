# MoE parity and speed against Strata and llama.cpp

Status: method, pins and model inventory recorded. Measurements are pending. Results go to
`benchmarks/results/` (dated) and the MoE section of `benchmarks/scoreboards/LLM.md` under the
benchmark protocol in `benchmarks/README.md`. The frozen suite manifest for this program is added
with its first measured run.

## The question

Does our MoE path (expert offload, the residency budget, speculative decode) give the same tokens as
llama.cpp and Strata on the same GGUF file, and how fast is it on each GPU we can rent?

## Engines and pins

| Engine | Source | Pinned revision | Build |
|---|---|---|---|
| HartsyInference | this repository | recorded per run (commit in every result) | .NET 10, Release |
| llama.cpp | `ggml-org/llama.cpp` tag `v0.6.0` | commit `d81235049` | CMake + Ninja, `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=<sm>`, targets `llama-server llama-cli llama-bench` |
| Strata | `Niko1221/Strata` tag `v0.1.41` | commit `fb58e0db` (MIT) | CMake + Ninja, `-DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF -DCMAKE_CUDA_ARCHITECTURES=<sm>`, target `strata` |

Strata reads GGUF directly (`tools/strata_inspect.py` reports whether it runs a file). Its
`*_pack.py` converters are optional performance paths and are not used in this baseline. Its
`setup.sh` is not used because it downloads a default model.

## Harness

- Strata's `bench/bench_vs_llama.py`, used unmodified. Both engines run behind an OpenAI-compatible
  server with the same prompt text and sampler: temperature 0.0 and 0.7, `top_k` 20, `top_p` 0.95,
  `seed` 1. Each request is non-streaming.
- Timings come from each server (llama.cpp names): prompt tok/s, decode tok/s, TTFT as `prompt_ms`.
  Our server returns the same `timings` object from PR #355 (draft).
- Peak VRAM is sampled with `nvidia-smi` minus the idle baseline. Peak RAM is the engine process tree's
  RSS. Each configuration is started, warmed up, measured, and stopped in turn.
- Cells: a 4K-token prompt with 2 repetitions per temperature, and a 32K-token prompt with 1
  repetition per temperature. Each generates 256 tokens. Results are medians over at least 3 rounds,
  with configurations interleaved so the order does not favor one engine.
- Metrics per cell: prompt tok/s, decode tok/s, TTFT, peak VRAM, peak RAM, and dollars per 1M decode
  tokens (hourly price divided by decode throughput).
- Known gaps: the harness does not use streaming. Top-k logprobs for the parity checks are a follow-up.

## Prompts

- Corpus: the `docs/` tree of this repository at `d8719ead`: 160 Markdown files, 2,723,829 bytes,
  SHA-256 `32774f6e7775d571…`. It tokenizes to 886,720 tokens with the DeepSeek-V2-Lite tokenizer.
- `prompt-4k.txt`: exactly 4,000 tokens, SHA-256 `19f490347fcc0985…`.
- `prompt-32k.txt`: exactly 32,000 tokens, SHA-256 `128bb502d5b5e6db…`.
- Each prompt is decoded from the corpus with the model's own tokenizer and re-encoded to the exact
  count. Different tokenizers give different counts for the same text. Each result records the count
  its engine reports.

## Models

GGUF files on the home RAID at `/mnt/model-storage/Models/llm/moe-parity/`. Sizes and SHA-256 values are
the Hugging Face LFS hashes, checked after download. The manifest is `MANIFEST.tsv` in that folder.

| Model | Quant | Upstream | Bytes | SHA-256 | Engine status on `main` |
|---|---|---|---:|---|---|
| Granite-3.0-1B-A400M instruct | Q4_K_M | local copy | 821845024 | `074f09e13484e54e73c93830d34e9fa9917a6319fb8bae762a22594b9b4da0dc` | granite-moe; verified on real weights |
| OLMoE-1B-7B-0924 instruct | Q4_K_M | RAID copy, hash computed locally | 4213512672 | `8c310f1435a1222338fd2d3d974975be9cd908180b644bab0c2a94da1ac32f3f` | olmoe; wired from GGUF |
| Granite-3.0-3B-A800M instruct | Q4_K_M | `bartowski/granite-3.0-3b-a800m-instruct-GGUF` | 2059356000 | `aa6287bafd182d06a6d11b0d001458deec902e8c0522875fc20cde2f561ebf6d` | granite-moe; not run yet |
| Granite-3.0-3B-A800M instruct | Q8_0 | same repo | 3593000096 | `b5ee3140bb0a0f7574c835d0b2e18c5ee179f66a7daded44fa488ebf3fd6c397` | not run yet |
| DeepSeek-V2-Lite | Q4_K_M | `mradermacher/DeepSeek-V2-Lite-GGUF` | 10364416736 | `70985c39ba3e51a53e6ad6ae25ce29a3efc92b4802f61645760e1bc1eb89ad1a` | MLA; output incoherent (open) |
| DeepSeek-V2-Lite | Q8_0 | same repo | 16702517984 | `61d2ca1bfb557cad505fd021ceb9f6b8894039cf679c81e7e4456acd36688de5` | MLA; output incoherent (open) |
| Qwen3-30B-A3B | Q4_K_M | `unsloth/Qwen3-30B-A3B-GGUF` | 18556686912 | `9f1a24700a339b09c06009b729b5c809e0b64c213b8af5b711b3dbdfd0c5ba48` | qwen3moe; wired, not verified end to end |
| Qwen3-30B-A3B | Q8_0 | same repo | pending | pending | download in progress |
| Mixtral-8x7B-Instruct v0.1 | Q4_K_M | `TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF` | 26441533376 | `9193684683657e90707087bd1ed19fd0b277ab66358d19edeadc26d6fdec4f53` | llama + experts; wired, not verified end to end |
| Mixtral-8x7B-Instruct v0.1 | Q8_0 | same repo | pending | pending | download in progress |

Not yet downloaded: Mixtral-8x22B (about 90 GB in two parts) and Qwen3.8-Flash-Next. The second has no
GGUF that both engines are known to read, and our engine does not implement its architecture.

## GPUs

RunPod Secure Cloud, on-demand, prices from the console when the run starts. Stock varies by data
center, so every result records its data center.

| GPU | VRAM | Architecture | Price per hour | Use |
|---|---|---|---:|---|
| RTX A40 | 48 GB | sm_86 | 0.59 | baseline and MoE checks (used) |
| L40S | 48 GB | sm_89 | 1.09 | cross-data-center checks (used) |
| RTX 3090 | 24 GB | sm_86 | 0.50 | planned |
| RTX 5090 | 32 GB | sm_120 | 1.19 | planned |
| A100 80GB PCIe | 80 GB | sm_80 | 1.79 | planned |
| RTX PRO 6000 | 96 GB | sm_120 | 2.49 | planned |
| H100 SXM | 80 GB | sm_90 | 3.99 | optional |

Local cards, not used for these measurements: RTX 3060 12 GB (sm_86) and RTX 4090 24 GB (sm_89).
The 4090 is shared with other work.

## Parity definitions

- **P1, router.** The top-k expert IDs must be identical for every token and layer, against the CPU
  reference.
- **P2, logits.** Top-1 agreement at least 99% over 256 generated tokens, with the mean absolute logprob
  difference reported. Needs top-k logprobs from the API (follow-up).
- **P3, generation.** Greedy decoding of 256 tokens on 10 fixed prompts: exact-match rate and first
  divergence index against llama.cpp and Strata on the same GGUF. Perplexity on a fixed 4K-token text
  must be within 0.5% of llama.cpp.
- **Cross-SKU determinism.** The same greedy run must give the same tokens on every GPU we measure. A
  mismatch is reported as a finding.

## Status

- Timings and `/health` for the harness: draft PR #355 (`api/llama-compatible-timings`). CI green.
- DeepSeek-V2-Lite: fixed on draft PR #357 (`moe/dsv2-lite-mla`). MLA's decoupled RoPE pairs adjacent dims, and a
  `deepseek2` GGUF without `expert_weights_norm` no longer renormalizes its top-k weights. The engine's greedy text,
  re-tokenized with the HF tokenizer, matches the HF bf16 reference 16/16. llama.cpp v0.6.0 gives a different first
  token on the same string. That is not settled: `llama-cli` wraps the prompt in its chat template and prints no ids,
  so parity against llama.cpp needs explicit-id runs through `llama-server`.
- Smoke test, one request, not a benchmark: Granite-3B Q4_K_M on the A40, commit `915324e8`. Output
  `The capital of France is Paris.`, prompt 35.6 tok/s, decode 33.4 tok/s, peak VRAM 1.5 GB, host RSS 1.9 GB.
- Expert placement on CUDA. The text placement planner (alpha.329) decides each GGUF's placement before loading it: one GPU,
  a layer split across GPUs, or expert offload (`TextPlacementPlanner`, `vram.textPlacement`, `TextRequest.Placement`).
  A MoE layer's experts take one device allocation per projection (alpha.328), so Qwen3-30B-A3B Q4_K_M (18.5 GB) runs
  on one 24 GB RTX 4090 at about 24 tok/s. Under expert offload (alpha.331) a `CudaExpertCache` holds the experts its
  budget allows and the rest run on the CPU from the packed kernels (`MoeExpertOffload`). Measured on the RTX 3060 alone
  (7.4 GB cache, 46% of the experts): 61-70% of routed rows from the cache and 4-5 tok/s decode, bound by the CPU
  kernels (about 1 ms per expert row with the row-parallel kernel); faster kernels are the open follow-up.
- The F32 host expert runtime (`RunRoutedThroughRuntime`, `CpuOnlyPolicy`) still runs only when
  `MoeFeedForward.UseHostExpertRuntime` is set; expert offload is the production path. `CudaExpertDeviceRunner` (F32 only)
  has no construction site: resident quantized experts run through the existing device projections instead.
- The CPU device path dequantizes everything to F32 (`GgufLanguageModel.Load`, `dequantizeToF32`). The load guard
  (`TextService.EnsureRamHeadroomFor`) requires 2.5 times the file on that path, which is less than the F32 size. For
  DeepSeek-V2-Lite the F32 weights are about 63 GB against 26 GB required, and the local host had 41 GB free. Until the
  guard counts the F32 size, a CPU load of these models on the 62 GB local host can exhaust memory.
- The CUDA guard counts only quantized tensors outside `GpuSupportedQuant` (Q2_K through Q8_0). Strata's routed experts
  use IQ2_XS, IQ3_XXS and IQ4_NL. Those types are not device-supported, so the loader expands them to F32 on the
  device, which is about 483 GB for Qwen3.8-Flash-Next. That is the blocker for a Strata comparison, not host RAM.
- Strata cannot run the other models in this matrix. `tools/strata_inspect.py` at `fb58e0db` reports that Strata runs
  only Qwen3.8-Flash-Next. It refuses Granite-3B (`granitemoe`), OLMoE-1B-7B (`olmoe`), Qwen3-30B-A3B (`qwen3moe`), Mixtral-8x7B (`llama`)
  and DeepSeek-V2-Lite (`deepseek2`). A Strata comparison needs device kernels for Strata's IQ expert types first.
  Until then the baseline is llama.cpp alone.
- Pins. The LLM Assistant extension pins engine alpha.270. That release already contains `MoeFeedForward`, and no
  commit has touched that file since. Twenty-three LLM-layer commits came after it, mostly DeepSeek V4.1, so loading a
  MoE GGUF in the extension needs no pin bump. The extension's model list scans `*.gguf` recursively with no
  architecture filter, so the MoE files already list.
- The harness engine is commit `915324e8`: E3 timings (`b1e67476`) with both DeepSeek-V2-Lite fixes cherry-picked.
  llama.cpp and Strata are built on the baseline pod at the pinned revisions. llama.cpp is the baseline for all models.

## Reproduce

```bash
# engine tests for the timing and compat surface
dotnet test tests/HartsyInference.API.Tests --filter "FullyQualifiedName~CompatTimingsTests|FullyQualifiedName~CompatEndpointsTests"

# prompts: decode the corpus with the model's own tokenizer to exactly 4000 and 32000 tokens, then write
# prompt-4k.txt and prompt-32k.txt into the run directory. The harness's own `prompts` mode uses Strata's
# tokenizer format, so it is not used for these models.

# harness, after building both servers and staging the GGUF and the two prompt files
python bench/bench_vs_llama.py run --machine CFG.json --out DIR --rounds 3
python bench/bench_vs_llama.py summary --out DIR
```
