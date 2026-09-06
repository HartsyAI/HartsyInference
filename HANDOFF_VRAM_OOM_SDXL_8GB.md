# Handoff: SDXL OOMs on an 8GB card even though the driver reports plenty free

## Summary

A real-money end-to-end test of SwarmUI-CloudBackends against a fresh Vast.ai instance (RTX 3070 Laptop, 8GB / 7842 MB usable VRAM, single GPU) got HartsyInference running as a SwarmUI backend and tried a plain SDXL 1024x1024 txt2img generation with default settings (`LowVram = "Auto"`). It failed with `OutOfVramException`, retried once (per the existing retry-once-on-OOM logic), and failed identically the second time. Z-Image Turbo was tried first and failed for an unrelated, expected reason (missing `qwen_3_4b.safetensors` text encoder, never downloaded - not part of this handoff).

This is not obviously a "the model just doesn't fit" case: the driver-level free-memory numbers logged immediately before and after the failure look fine (7649 MB free of 7842 MB), but the allocator's own accounting at the moment of failure says 0.0% free. That gap between "the driver thinks there's room" and "the allocator has none to give" is the thing worth root-causing, not just papering over with a bigger card.

## Environment

* GPU: NVIDIA GeForce RTX 3070 Laptop GPU, 8.00 GiB total, 7842.1 MB usable (per `[VRAM]` log lines), single GPU (`GPU_ID=0`, no `DitShardGpuId`/`TextEncoderGpuId` set).
* Backend settings: `ComputeBackend=auto`, `LowVram=Auto (recommended)`, `OverQueue=1`, defaults otherwise. This is the `HartsyInference (Pure C# Inference)` backend added fresh in SwarmUI's Server > Backends UI - not a config carried over from anywhere else.
* Model: `OfficialStableDiffusion/sd_xl_base_1.0` (the exact file SwarmUI's own installer downloads for "Stable Diffusion XL 1.0 (Base)" - `stabilityai/stable-diffusion-xl-base-1.0` on Hugging Face), 1024x1024, 20 steps, cfg 7, no refiner/LoRA/ControlNet/IP-adapter involved. First generation on a completely fresh process (nothing else ever loaded onto this GPU in this run).
* SwarmUI 0.9.8.3, HartsyInference extension built from `HartsyAI/SwarmUI-HartsyInference-Backend` at whatever commit was current 2026-09-06 (references `HartsyInference.Engine`/`HartsyInference.Cuda` from this repo).

## Exact repro

1. Fresh SwarmUI instance, only the `HartsyInference (Pure C# Inference)` backend configured, defaults untouched.
2. Load `sd_xl_base_1.0`, generate one 1024x1024 image, 20 steps, cfg 7, any prompt.
3. Fails both on the very first attempt and on the automatic retry.

## Full log (chronological, both attempts)

```
01:43:52.597 [Info] [VRAM] sdxl: policy Auto, device cuda:0 (7649 MB free of 7842 MB), model supports CfgParallel, ComponentPlacement.
01:44:03.883 [Info] [SdxlRecipe] SDXL ready.
01:44:04.391 [Info] SDXL txt2img: 1024x1024, 20 steps, cfg=7, seed=1888704162
01:44:04.391 [Info] Encoding text with dual CLIP encoders...
01:44:11.510 [Info] Text encoding done in 7118ms
01:44:12.926 [Warning] [CudaMemory] OOM on first attempt: requested=10.0 MB, free=3.1 MB, total=7842.1 MB (0.0% free)
01:44:12.927 [Warning] [CudaMemory] OOM after sync+pool-trim retry: requested=10.0 MB, free=3.1 MB, total=7842.1 MB (0.0% free)
01:44:12.928 [Error] [Cuda] PreloadWeights failed after 1133 weight(s) — rolling back this batch.: HartsyInference.Core.Exceptions.OutOfVramException: Out of VRAM: requested 10 MB but only 3 MB available.
   at HartsyInference.Cuda.CudaMemory.ThrowExhausted(Int32 result, UIntPtr requested)
   at HartsyInference.Cuda.GpuTransferHelper.PreloadWeight(Tensor weight)
   at HartsyInference.Cuda.CudaBackend.PreloadWeights(IEnumerable`1 weights)
01:44:13.131 [Error] [Engine] Generation ran out of VRAM — releasing device memory held by the partially-loaded pipeline.: HartsyInference.Core.Exceptions.OutOfVramException: Out of VRAM: requested 10 MB but only 3 MB available.
   at HartsyInference.Diffusion.Pipelines.SdxlPipeline.GenerateFromTokens(...)
   at HartsyInference.Engine.Recipes.Image.SdxlRecipePipeline.Generate(ImageRequest request, IProgress`1 progress, CancellationToken cancel)
   at HartsyInference.Engine.Services.ImagesService.<>c__DisplayClass3_1.<GenerateAsync>b__1()
   at HartsyInference.Engine.InferenceEngine.GenerateWithVramCleanup[T](Func`1 generate)
01:44:13.132 [Info] [Cuda] FreeAllDeviceMemory: free 7649 MB → 7649 MB
01:44:13.132 [Warning] [HartsyInference] Out of VRAM: requested 10 MB but only 3 MB available. Freeing cached models and retrying once.

--- retry (second, identical attempt) ---

01:44:13.427 [Info] [VRAM] sdxl: policy Auto, device cuda:0 (7649 MB free of 7842 MB), model supports CfgParallel, ComponentPlacement.
01:44:22.241 [Info] [SdxlRecipe] SDXL ready.
01:44:22.544 [Info] SDXL txt2img: 1024x1024, 20 steps, cfg=7, seed=1888704162
01:44:22.544 [Info] Encoding text with dual CLIP encoders...
01:44:28.046 [Info] Text encoding done in 5502ms
01:44:29.407 [Warning] [CudaMemory] OOM on first attempt: requested=10.0 MB, free=3.1 MB, total=7842.1 MB (0.0% free)
01:44:29.407 [Warning] [CudaMemory] OOM after sync+pool-trim retry: requested=10.0 MB, free=3.1 MB, total=7842.1 MB (0.0% free)
01:44:29.407 [Error] [Cuda] PreloadWeights failed after 1133 weight(s) — rolling back this batch.: ...
01:44:29.605 [Error] [Engine] Generation ran out of VRAM — releasing device memory held by the partially-loaded pipeline.: ...
01:44:29.605 [Info] [Cuda] FreeAllDeviceMemory: free 7649 MB → 7649 MB
01:44:29.606 [Warning] Refused to generate image for local: Out of VRAM on the HartsyInference backend (needed 10 MB, 3 MB free). Out of VRAM: requested 10 MB but only 3 MB available.
```

The user-facing message SwarmUI shows (from `DescribeVramFailure`, `HartsyInferenceBackend.cs:1761`):

> Out of VRAM on the HartsyInference backend (needed 10 MB, 3 MB free). Out of VRAM: requested 10 MB but only 3 MB available.
> Lower the resolution or batch size, or set this backend's `DitShardGpuId` to a second CUDA ordinal to pool both cards' VRAM, or set `LowVram` to stream weights instead of keeping them resident. Freeing cached models and retrying once already failed, so this is not a stale-cache problem.

## What's suspicious (not just "8GB is too small")

1. **Both attempts fail at the exact same point**: `PreloadWeights failed after 1133 weight(s)`, needing exactly 10 MB more both times, with the pool reporting exactly `free=3.1 MB` both times. That's a deterministic, reproducible wall - consistent with an allocation strategy that always lands 10 MB short on this card/model combo, not a random fragmentation issue.
2. **The driver-level number right before and right after says 7649 MB free**, not "nearly full". `[VRAM] sdxl: policy Auto ... (7649 MB free of 7842 MB)` is logged *before* weight preload starts, and `[Cuda] FreeAllDeviceMemory: free 7649 MB → 7649 MB` is logged immediately *after* the failed batch is rolled back - both showing the card is nearly empty. Whatever ate the VRAM during `PreloadWeights` is either (a) genuinely correct - SDXL's real footprint at `ComponentPlacement` residency plus dual CLIP plus activation buffers for 1024x1024 truly doesn't fit in ~7.8 GB - or (b) the `Auto` policy's up-front decision (made with 7649 MB free, before anything is loaded) doesn't correctly account for what SDXL + `ComponentPlacement` actually needs on a card this size, so it commits to full-resident placement it can't finish.
3. **The `Auto` policy did not fall back to a lower-memory strategy** on either attempt. The one retry that exists (`HartsyInferenceBackend.cs:579-594`, "Freeing cached models and retrying once") only calls `FreeMemory(false)` and re-dispatches with the *same* policy - it does not try `LowVram` streaming as a second-chance fallback before giving up. If `Auto`'s initial choice was wrong for this GPU, the retry can never succeed, because it's not a stale-cache problem (as the error message itself points out) - it's a policy problem.
4. **1133 weights loaded successfully before failing on what's presumably the last (or near-last) one.** Worth checking whether that's actually the full SDXL UNet+CLIP+VAE weight count for this config, or whether something is being loaded that shouldn't be resident yet (e.g. a component that `ComponentPlacement` should have deferred/streamed but didn't).

## Where to look

All in this repo (`HartsyInference`) unless noted:

* `src/HartsyInference.Cuda/CudaMemory.cs` - `ThrowExhausted`, the OOM diagnostic logging (`LogOomDiagnostic`), and the sync+pool-trim retry that runs *inside* a single allocation attempt (separate from the higher-level "retry once" in the extension).
* `src/HartsyInference.Cuda/GpuTransferHelper.cs` - `PreloadWeight`, where the specific 10 MB allocation that fails is requested.
* `src/HartsyInference.Cuda/CudaBackend.cs` - `PreloadWeights`, the loop that got through 1133 weights before failing; worth logging (or checking under a debugger) the running VRAM total as it iterates, to see whether the allocator's internal ledger and the driver's real free-memory diverge *during* the loop, not just before/after it.
* `src/HartsyInference.Engine/Recipes/Image/SdxlRecipe.cs` (`MemorySupports => CfgParallel | ComponentPlacement`) and `SdxlRecipePipeline.cs` - where the `Auto` policy decides how to place SDXL's components for a given amount of free memory. This is the most likely place the actual bug lives: the policy decision happens once, up front, based on free memory before anything is loaded, and there is no evidence it re-checks or adapts once weights start actually landing on the device.
* `src/HartsyInference.Engine/Recipes/MemoryCapabilities.cs` / `MemorySupportReport.cs` - the general VRAM-policy machinery shared across recipes; check whether "how much does this model really need at this resolution" is estimated anywhere, or whether `Auto` on a tight-fit card just optimistically tries full-resident every time.
* Extension side (separate repo, `HartsyAI/SwarmUI-HartsyInference-Backend`, local checkout at `/home/kalebbroo/Desktop/Projects/SwarmUI/src/Extensions/SwarmUI-HartsyInference`): `Backends/HartsyInferenceBackend.cs:574-598` (the retry-once-on-OOM logic) and `:~1761` (`DescribeVramFailure`, the user-facing message). If the real fix belongs in the engine, this retry could still be improved to try `LowVram` streaming as the second attempt instead of the identical policy - cheap insurance even after the root cause is fixed, since a second identical attempt can only ever succeed if the first failure was pool fragmentation, and the logs show it isn't.

## Suggested first steps

1. Reproduce locally on any ~8 GB card (doesn't need to be this exact Vast.ai instance - it was destroyed after this test) with SDXL, 1024x1024, `LowVram=Auto`, and confirm the same `1133 weight(s)` / `10 MB` / `3.1 MB` numbers land in the same place. If they're not bit-for-bit identical, the fragmentation angle deserves more weight than the policy-sizing angle.
2. Try the same request with `LowVram` explicitly forced to whatever the "stream weights" setting is called (not `Auto`) and confirm it succeeds. If it does, that's strong evidence `Auto`'s decision heuristic is what needs fixing for this class of card, not the streaming path itself.
3. Instrument `PreloadWeights` (or turn on whatever verbose CUDA memory logging exists) to print device free-memory every N weights during the loop that got to 1133, to see where the real budget actually goes - is it the UNet, the dual CLIP, or activation buffers reserved up front for 1024x1024?
4. Once root-caused, decide whether the fix belongs in the `Auto` policy's up-front decision (estimate real footprint before committing to full-resident), or in the retry logic making a genuine second attempt with a different policy instead of repeating the first one.

## Why this wasn't fixed in the same session

This was found during a real-money, real-hardware end-to-end verification of the separate `SwarmUI-CloudBackends` extension (RunPod/Vast.ai cloud backend support) against a live Vast.ai instance running HartsyInference as its backend - not a HartsyInference engine debugging session. The instance that produced this log has since been destroyed (Vast.ai billing), so there's no live environment to keep poking at; everything relevant is captured above. The extension itself worked correctly the whole way through (backend registered, ran, dispatched the request) - this is purely an engine-side VRAM accounting issue.
