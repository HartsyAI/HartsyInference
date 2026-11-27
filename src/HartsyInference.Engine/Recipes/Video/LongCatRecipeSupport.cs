using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors;
using HartsyInference.Engine.Features;
using HartsyInference.Engine.Recipes;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines.LongCat;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.Tokenizers;
using HartsyInference.Core.Memory.Vram.Streaming;
using HartsyInference.ModelAssets.Lora;
using HartsyInference.Video;

namespace HartsyInference.Engine.Recipes.Video;

/// <summary>What every LongCat recipe loads: the DiT from the checkpoint directory, the VAE, and umT5 with its tokenizer. The latent channel count
/// is the DiT's own <see cref="LongCatConfig.InChannels"/>, so a tiny config and the release (16 channels) share one path.</summary>
internal sealed record LongCatParts(LongCatTransformer Dit, int Channels, int CaptionChannels, bool HasAudio, IWanVaeDecoder Decoder,
    IWanVaeEncoder Encoder, T5TextEncoder Umt5, T5Tokenizer Tokenizer, AdjunctGate? DmdGate = null, AdjunctGate? OpsdGate = null,
    AdjunctGate? CfgStepGate = null);

/// <summary>Where the umT5 and VAE come from. Null in production (the engine's side-model downloads); a test injects small stand-ins here, so the
/// recipe path runs end to end without the multi-gigabyte files. Each factory returns objects the caller owns.</summary>
internal sealed record LongCatSideModels(Func<(T5TextEncoder Encoder, T5Tokenizer Tokenizer)> LoadUmt5,
    Func<(IWanVaeDecoder Decoder, IWanVaeEncoder Encoder)> LoadVae);

/// <summary>The shared LongCat recipe support: the Wan VAE's compression, the loader, and the host-side steps (prompt encode, noise,
/// first-frame conditioning, latent decode) that the base and avatar families run in common.</summary>
internal static class LongCatRecipeSupport
{
    /// <summary>Temporal compression of the Wan 2.1 VAE: a clip of <c>1 + 4·(latentFrames − 1)</c> frames.</summary>
    internal const int TemporalCompressionFactor = 4;

    /// <summary>Spatial compression of the Wan 2.1 VAE: width and height are multiples of 8.</summary>
    internal const int SpatialCompressionFactor = 8;

    /// <summary>Flow shift of the base family's sampler (the research plan's base value). Avatar uses 7; see the avatar recipe.</summary>
    internal const float BaseFlowShift = 12f;

    /// <summary>Width and height must be multiples of this: the VAE's 8 times the DiT's 2×2 patch.</summary>
    internal const int PatchGrid = 16;

    /// <summary>One binary gibibyte, the unit of the streaming headroom terms.</summary>
    private const long GiB = 1L << 30;

    /// <summary>The allocator pool's transients for one denoise step, independent of the token count: 5.5 GiB. With the safety margin the fixed part
    /// is 6.5 GiB. The 5 GiB fixed part was lowered from the 8.3 GiB estimate on the basis of spare VRAM at peak; the 4090 full-clip run gave
    /// prefix 0 with 22,852 MB free before the shared preload, so the lower figure is not yet measured safe. 6.5 GiB stays until the
    /// logged plan (free after shared, shared, headroom, block) settles it on a measured run.</summary>
    internal const long FixedPoolBytes = GiB * 11 / 2;

    /// <summary>The safety margin Auto leaves on top of the fixed terms (1 GiB): the conservative slack that keeps a near-full card from streaming
    /// too late.</summary>
    internal const long SafetyBytes = GiB;

    /// <summary>Rows per projection and FFN tile: the block's shared tile height, which bounds the FFN and the self-attention QKV intermediates to one
    /// tile rather than the whole clip.</summary>
    internal const int ActivationTileRows = LongCatBlock.DefaultTileRows;

    /// <summary>The device headroom the streaming plan reserves before it sizes the resident prefix: the fixed context, pool and safety terms, plus
    /// the step's activations. Activations split by what the tiling bounds: the per-token block copies that stay whole (about 14 × hidden F32 elements
    /// per token: the residual, norm, joined Q, K and V, attention output) scale with the tokens, and one tile's fused QKV and SwiGLU intermediates
    /// (3 × hidden + 3 × FFN per row) scale with the tile. The first 4090 runs measured the untiled estimate; the split is derived from it, not
    /// re-measured, and needs a GPU re-check against a tiled run.</summary>
    internal static long ActivationHeadroomBytes(long tokens, int hidden, int ffnHidden)
    {
        const long bytesPerElement = 4;
        long wholePerToken = 14L * hidden * bytesPerElement;
        long tileRows = Math.Min(tokens, ActivationTileRows);
        long tileBytes = tileRows * (3L * hidden + 3L * ffnHidden) * bytesPerElement;
        return FixedPoolBytes + SafetyBytes + tokens * wholePerToken + tileBytes;
    }

    /// <summary>The attention tokens one denoise step sees at a latent grid: latent frames × (latent height / 2) × (latent width / 2), the DiT's 2×2 patch.</summary>
    internal static long TokenCount(int latentFrames, int latentHeight, int latentWidth) =>
        (long)latentFrames * (latentHeight / 2) * (latentWidth / 2);

    /// <summary>Places the DiT for one denoise loop through the shared streaming scope: resident when it fits the device, block-streamed from host
    /// memory when it does not. Disposing the scope evicts the streamed window; the resident prefix stays on the device until the pipeline's pin
    /// releases it (the denoise loop's finally frees the whole DiT). <paramref name="cacheBytes"/> is the KV cache the run keeps for its blocks; it
    /// counts against the same budget as the activations, and a run whose cache leaves no room is refused before its denoise starts.</summary>
    internal static BlockStreamingScope OpenDenoiseStream(IBackend backend, LongCatParts parts, string modelName, long tokens, long cacheBytes = 0,
        long? measuredPeakBytes = null)
    {
        long headroom = BudgetHeadroomBytes(tokens, parts.Dit.Config.HiddenSize, parts.Dit.Config.FfnHiddenDim, measuredPeakBytes, cacheBytes);
        if (cacheBytes > 0 && HasDeviceBudget(backend))
        {
            // The planner's own inputs, read after the shared weights are placed and the pool trimmed, as BlockStreamingScope does.
            backend.PreloadWeights(parts.Dit.EnumerateSharedWeights());
            backend.TrimMemoryPool();
            RequireRoom(backend.FreeMemoryBytes(), 0, headroom - cacheBytes, cacheBytes, "The chain's later run");
        }
        return BlockStreamingScope.Open(new BlockStreamingOptions
        {
            Backend = backend, Denoiser = parts.Dit, ModelName = modelName, HeadroomBytes = headroom,
        });
    }

    /// <summary>The device bytes a cached step holds for one block beyond its activations: the staged pair of the cached keys and values, and the joined
    /// copy the attention builds from them (past rows followed by the new rows). The new rows' own keys and values are activations and are counted in
    /// the per-token term. Two pairs of the conditioning tokens: 487.5 MiB at 832x480 with five conditioning latents, 288 MiB at 640x368.</summary>
    internal static long CachedStepBytes(LongCatConfig config, int condLatents, int latentHeight, int latentWidth) =>
        2 * KvCacheBytes(config, condLatents, latentHeight, latentWidth, offloaded: true);

    /// <summary>The bytes a run's KV cache takes on the device, in F32: every block's keys and values for the conditioning latents (12.3 GB at 832x480
    /// with five of them, 7.25 GB at 640x368). Offloaded, the cache lives in host memory and the device holds one block's pair at a time.</summary>
    internal static long KvCacheBytes(LongCatConfig config, int condLatents, int latentHeight, int latentWidth, bool offloaded) =>
        LongCatKvCache.EstimateBytes(config, condLatents, latentHeight, latentWidth, offloaded);

    /// <summary>One measured later-run workload: the size, the new and context latents of its segment, whether it was the distilled pipeline, and the
    /// allocator's reserved bytes at the worst step of the later segment.</summary>
    internal readonly record struct MeasuredLaterRun(int Width, int Height, int NewLatents, int ContextLatents, bool Distilled, long ReservedPeakBytes,
        DeviceType Device);

    /// <summary>The later runs measured on the GPU: main c5fec5d37 with the refusal bypassed in a scratch copy, all eight steps, no OOM. The key is
    /// exactly what was measured: the device type (CUDA), size, latents and the distilled flag. A segment of other new or context latents, another size, or the undistilled pipeline has no entry and keeps the
    /// per-token estimate. The distilled flag is inferred from the eight steps (the Distill8 schedule); it is not in the measurement's log. Audio
    /// attention was not sampled. Reserved bytes are the pool's RESERVED_MEM_CURRENT (CudaBackend.cs:146-152), which excludes the resident blocks,
    /// since those are allocated outside the pool (GpuTransferHelper.cs:326, 482).</summary>
    internal static readonly MeasuredLaterRun[] MeasuredLaterRuns =
    {
        new(640, 368, NewLatents: 20, ContextLatents: 4, Distilled: true, 7808L << 20, DeviceType.Cuda),
        new(832, 480, NewLatents: 20, ContextLatents: 4, Distilled: true, 11360L << 20, DeviceType.Cuda),
    };

    /// <summary>The explicit margin on a measured peak: the safety margin the estimate also keeps. It covers what the measurement did not sample (audio
    /// attention), the tile heights the attention sizes from free memory (a few tens of MiB of drift per call), and one run's sample against the
    /// next. It is not scaled between the two sizes: the margin is set against the unsampled terms, which do not grow with the measured ones.</summary>
    internal const long MeasuredMarginBytes = SafetyBytes;

    /// <summary>The reserved peak measured for exactly this workload (device type, size, new and context latents, distilled), or null when it was not measured. A null
    /// keeps the per-token estimate and its refusal.</summary>
    internal static long? MeasuredReservedPeak(DeviceType device, int width, int height, int newLatents, int contextLatents, bool distilled)
    {
        // The pool's reserved bytes are the CUDA allocator's. A Vulkan (or any other) run has no measurement, so it keeps the estimate.
        foreach (MeasuredLaterRun run in MeasuredLaterRuns)
            if (run.Device == device && run.Width == width && run.Height == height && run.NewLatents == newLatents
                && run.ContextLatents == contextLatents && run.Distilled == distilled)
                return run.ReservedPeakBytes;
        return null;
    }

    /// <summary>The later runs' figures for one build: the token count of a later step, its KV cache bytes, the measured peak it may use, and whether
    /// its cache is offloaded. A mixed (1.6) run is uncached: its step runs every context and new frame, holds no cache, has no measurement, and
    /// does not offload.</summary>
    internal sealed record LongCatLaterRunFigures(long Tokens, long CacheBytes, long? MeasuredPeak, bool OffloadKv);

    internal static LongCatLaterRunFigures LaterRunFigures(bool mixed, LongCatAvatarChainPlan plan, int latentHeight, int latentWidth,
        LongCatConfig config, int pixelWidth, int pixelHeight, bool distilled, DeviceType device)
    {
        if (mixed)
        {
            return new LongCatLaterRunFigures(TokenCount(plan.ContextLatents + 1 + plan.NewLatentsPerSegment, latentHeight, latentWidth), 0, null,
                OffloadKv: false);
        }
        return new LongCatLaterRunFigures(TokenCount(plan.NewLatentsPerSegment, latentHeight, latentWidth),
            CachedStepBytes(config, plan.ContextLatents + 1, latentHeight, latentWidth),
            MeasuredReservedPeak(device, pixelWidth, pixelHeight, plan.NewLatentsPerSegment, plan.ContextLatents, distilled), OffloadKv: true);
    }

    /// <summary>The activation budget of a later run: the measured reserved peak plus its margin when the size was measured, else the per-token
    /// estimate (the fixed pool, the safety margin, and the activations of <paramref name="tokens"/> rows).</summary>
    internal static long ActivationBudget(long tokens, int hidden, int ffnHidden, long? measuredPeakBytes) =>
        measuredPeakBytes is long peak ? peak + MeasuredMarginBytes : ActivationHeadroomBytes(tokens, hidden, ffnHidden);

    /// <summary>The headroom a run reserves: the activation budget (see <see cref="ActivationBudget"/>) and the KV cache.</summary>
    internal static long BudgetHeadroomBytes(long tokens, int hidden, int ffnHidden, long? measuredPeakBytes, long cacheBytes) =>
        ActivationBudget(tokens, hidden, ffnHidden, measuredPeakBytes) + cacheBytes;

    /// <summary>Refuses a run whose budget is negative: the free memory, less the shared weights still to be placed, the activation headroom and the KV
    /// cache, must be at least zero. Otherwise the streamed blocks have nowhere to land and the first step runs out of memory. <paramref name="who"/>
    /// names the run in the error; the cache is named separately from the activations.</summary>
    internal static void RequireRoom(long freeBytes, long sharedBytesToPlace, long headroomBytes, long cacheBytes, string who)
    {
        long spendable = freeBytes - sharedBytesToPlace - headroomBytes - cacheBytes;
        if (spendable < 0)
        {
            throw new InvalidOperationException(
                $"{who} needs {MiB(headroomBytes)} MiB of activation headroom" +
                (cacheBytes > 0 ? $" and {MiB(cacheBytes)} MiB of KV cache" : "") +
                (sharedBytesToPlace > 0 ? $" and {MiB(sharedBytesToPlace)} MiB of shared weights to place" : "") +
                $", but only {MiB(freeBytes)} MiB is free: {MiB(-spendable)} MiB short. Lower the resolution or the chain length.");
        }
    }

    /// <summary>The activation budget of a later run of <paramref name="tokens"/> rows, the figure its streaming scope reserves: measured when the size
    /// has a measurement, else the per-token estimate.</summary>
    internal static long LaterRunHeadroom(LongCatParts parts, long tokens, long? measuredPeakBytes = null) =>
        ActivationBudget(tokens, parts.Dit.Config.HiddenSize, parts.Dit.Config.FfnHiddenDim, measuredPeakBytes);

    /// <summary>Before any segment runs: an upper bound on a later run's budget. The free memory is read after a pool trim. Only the shared weights
    /// not yet on the device are subtracted, so a shared weight that an earlier step already uploaded (the audio projection, once it is used twice)
    /// is not counted twice. A later run's free memory cannot exceed this, so a refusal here is one the later run would also get.
    /// <paramref name="stagedBytes"/> is the later run's cached-step bytes, which its scope has to cover as well; <paramref name="measuredPeakBytes"/>
    /// is the size's measured reserved peak, or null when the size has none.</summary>
    internal static void RequireChainRoom(IBackend backend, LongCatParts parts, long laterTokens, long stagedBytes, long? measuredPeakBytes = null)
    {
        backend.TrimMemoryPool();
        if (!HasDeviceBudget(backend))
            return;
        RequireRoom(backend.FreeMemoryBytes(), NonResidentSharedBytes(backend, parts.Dit), LaterRunHeadroom(parts, laterTokens, measuredPeakBytes),
            stagedBytes, "The chain's later run");
    }

    /// <summary>Whether the backend has device memory to budget. One without it (the CPU, which reports a total of 0) has no room to check: its 0 free
    /// bytes mean "not applicable", not "full".</summary>
    private static bool HasDeviceBudget(IBackend backend) => backend.GetVramInfo().TotalBytes > 0;

    /// <summary>The bytes of the DiT's shared weights that are not yet on the device, at their stored dtype.</summary>
    internal static long NonResidentSharedBytes(IBackend backend, LongCatTransformer dit)
    {
        long total = 0;
        foreach (Tensor tensor in dit.EnumerateSharedWeights())
            if (!backend.IsWeightResident(tensor))
                total += tensor.DType.ComputeByteCount(tensor.ElementCount);
        return total;
    }

    private static long MiB(long bytes) => (bytes + (1L << 20) - 1) >> 20;

    /// <summary>Loads the shared parts. Everything disposable goes into <paramref name="loaders"/>, which the caller owns from here on.
    /// <paramref name="sides"/> replaces the umT5 and VAE downloads when it is not null.</summary>
    /// <summary>The DMD distillation LoRA's static multiplier: the reference's <c>--lora_multiplier</c> default, applied when the LoRA is gated on.</summary>
    internal const float DmdMultiplier = 0.8f;

    /// <summary>The two LoRA files of the 1.6 build: the DMD (loraA) and OPSD (loraB) adapters, found beside the checkpoint's own folder as the
    /// release lays them out (<c>loraA/lora.safetensors</c>, <c>loraB/lora.safetensors</c>).</summary>
    internal sealed record LongCatLoraFiles(string DmdPath, string OpsdPath);

    /// <summary>Refuses a 1.6 build whose loraA or loraB is missing, before any weight is read, and names the row that holds them.</summary>
    internal static void RequireAvatar16Loras(string checkpointPath)
    {
        LongCatLoraFiles files = Avatar16LoraFiles(checkpointPath);
        foreach (string path in new[] { files.DmdPath, files.OpsdPath })
        {
            if (!File.Exists(path))
            {
                throw new NotSupportedException($"The 1.6 build needs its LoRA at '{path}', which is not there. Refusing to run without it. "
                    + "The 1.6 build is the longcat-avatar-1.6 row (longcat-avatar:1.6 from the CLI), with loraA and loraB beside its base folder.");
            }
        }
    }

    /// <summary>The 1.6 LoRA files for a checkpoint directory (the INT8 or bf16 base folder under the 1.6 row).</summary>
    internal static LongCatLoraFiles Avatar16LoraFiles(string checkpointPath)
    {
        string root = Path.GetDirectoryName(CheckpointDirectory(checkpointPath)) ?? ".";
        return new LongCatLoraFiles(Path.Combine(root, "loraA", "lora.safetensors"), Path.Combine(root, "loraB", "lora.safetensors"));
    }

    /// <summary>Where the 1.5 distillation LoRA sits: <c>lora/dmd_lora.safetensors</c> beside the checkpoint's own folder (the reference layout).</summary>
    internal static string DmdLoraPath(string checkpointPath) =>
        Path.Combine(Path.GetDirectoryName(CheckpointDirectory(checkpointPath)) ?? ".", "lora", "dmd_lora.safetensors");

    /// <summary>The base distilled LoRA's path, which must exist: the distilled build is refused rather than run as the undistilled base.</summary>
    internal static string RequireCfgStepLora(string checkpointPath)
    {
        string path = CfgStepLoraPath(checkpointPath);
        if (!File.Exists(path))
            throw new NotSupportedException($"The distilled base build needs its cfg_step LoRA at '{path}', which is not there. "
                + "Refusing to run the undistilled base as the distilled model.");
        return path;
    }

    /// <summary>A LoRA the recipe relies on must apply every unit it carries: a unit whose target the checkpoint lacks, or whose shape differs, is
    /// skipped silently by a partial stack. Refused, naming the counts, rather than run without it.</summary>
    internal static void RequireFullyApplied(LoraStack.ApplyReport report, string name)
    {
        if (report.Applied == 0 || report.SkippedMissing > 0 || report.SkippedShape > 0)
        {
            throw new NotSupportedException($"LoRA '{name}' did not apply in full: {report.Applied} units applied, "
                + $"{report.SkippedMissing} with no target in this checkpoint, {report.SkippedShape} with a mismatched shape. "
                + "The distilled base build needs all of them; refusing to run a partly applied LoRA.");
        }
    }

    /// <summary>Where the base distilled LoRA sits: <c>lora/cfg_step_lora.safetensors</c> beside the checkpoint's folder (the reference layout).</summary>
    internal static string CfgStepLoraPath(string checkpointPath) =>
        Path.Combine(Path.GetDirectoryName(CheckpointDirectory(checkpointPath)) ?? ".", "lora", "cfg_step_lora.safetensors");

    /// <summary>Loads the parts. With <paramref name="attachDmd"/>, the DMD LoRA is attached as a runtime gated term on the DiT's weights (the
    /// INT8 base included), so it is live only while its <see cref="AdjunctGate"/> is enabled. A missing LoRA file is refused here, before any
    /// weight is read, so a distilled request never runs on the bare base.</summary>
    internal static LongCatParts LoadParts(RecipeContext context, List<IDisposable> loaders, LongCatSideModels? sides, bool attachDmd = false,
        LongCatLoraFiles? mixedLoras = null, bool attachCfgStep = false)
    {
        // The placements the engine sets that LongCat cannot honour: CFG-parallel, DiT shards and context parallel are refused; a separate text-encoder
        // or VAE backend is warned about by the engine, which checks first (UnhonouredBackends); this second check does not warn again.
        UnhonouredBackends.Refuse(context, "LongCat", warn: false);
        // Planning refuses a request's LoRAs; a caller that constructs the recipe directly is told too, instead of running without them.
        if (context.Loras is { Entries.Count: > 0 })
        {
            throw new NotSupportedException("LongCat does not take a user LoRA; the release's own LoRAs are the only ones applied.");
        }
        // Planning refuses a device that cannot hold the weights; a caller that skipped it, or planned before the download, is refused here,
        // before an INT8 release is widened on the host.
        foreach (IBackend backend in context.TransformerBackends)
        {
            if (LongCatWeightFit.Refusal(backend, CheckpointDirectory(context.CheckpointPath)) is { } refusal)
                throw new NotSupportedException(refusal);
        }
        LongCatCheckpoint checkpoint = LongCatCheckpoint.Open(CheckpointDirectory(context.CheckpointPath));
        loaders.Add(checkpoint);
        // An INT8 checkpoint arrives as dense keys whose linears are I8 with their row scale attached (LongCatCheckpoint). The weight policy
        // keeps an int8 linear packed only where a backend runs it, and widens it elsewhere, before the DiT captures the weights.
        Dictionary<string, Tensor> weights = new(checkpoint.Weights, StringComparer.Ordinal);
        loaders.Add(QuantizedWeightPolicy.PrepareForBackends(weights, context.TransformerBackends, DType.F32));
        // Before the LoRAs and the DiT, which carry a weight's companions on from here.
        ApplyInt8WeightOnly(weights);
        // The DiT is registered before any LoRA or weight is read, so a failure at any later step disposes it, and the LoRA files and their
        // stack join the loaders the moment each is built, in the order that disposes the stack after the files it was built from.
        LongCatTransformer dit = new(checkpoint.Config);
        loaders.Add(dit);
        AdjunctGate? dmdGate = null;
        AdjunctGate? opsdGate = null;
        AdjunctGate? cfgStepGate = null;
        List<GatedLora> gatedLoras = [];
        if (mixedLoras is { } mixed)
        {
            // The 1.6 pair: the DMD and OPSD LoRAs, each at multiplier 1; the mixed schedule switches their gates per step.
            RequireAvatar16Loras(context.CheckpointPath);
            LoraFile dmd = LoraFile.Load(mixed.DmdPath);
            loaders.Add(dmd);
            LoraFile opsd = LoraFile.Load(mixed.OpsdPath);
            loaders.Add(opsd);
            dmdGate = new AdjunctGate();
            opsdGate = new AdjunctGate();
            gatedLoras.Add(new GatedLora(dmd, dmdGate, 1f));
            gatedLoras.Add(new GatedLora(opsd, opsdGate, 1f));
        }
        else if (attachDmd)
        {
            string dmdPath = DmdLoraPath(context.CheckpointPath);
            if (!File.Exists(dmdPath))
                throw new NotSupportedException($"The distilled build needs its distillation LoRA at '{dmdPath}', which is not there. Refusing to run the bare base as the distilled model.");
            LoraFile dmd = LoraFile.Load(dmdPath);
            loaders.Add(dmd);
            dmdGate = new AdjunctGate();
            gatedLoras.Add(new GatedLora(dmd, dmdGate, DmdMultiplier));
        }
        if (attachCfgStep)
        {
            // The base distilled path (16 steps, guidance 1.0): the cfg_step LoRA is a runtime gated term, on only while the distilled run is.
            LoraFile cfgStep = LoraFile.Load(RequireCfgStepLora(context.CheckpointPath));
            loaders.Add(cfgStep);
            cfgStepGate = new AdjunctGate();
            gatedLoras.Add(new GatedLora(cfgStep, cfgStepGate, 1f));
        }
        if (gatedLoras.Count > 0)
        {
            // The one LoRA path: RecipeLoraMerge merges the gated adapters into the transformer weights before the DiT captures them.
            LoraStack merged = RecipeLoraMerge.Apply((IReadOnlyList<LoraResolver.LoraSpec>?)null, context.Backend,
                new LoraMergeTargets { Transformer = weights }, "LongCat", gatedLoras)!;
            if (attachCfgStep)
            {
                try
                {
                    RequireFullyApplied(merged.Report, "cfg_step_lora.safetensors");
                }
                catch
                {
                    merged.Dispose();
                    throw;
                }
            }
            loaders.Add(merged);
        }
        dit.LoadWeights(weights);

        (IWanVaeDecoder decoder, IWanVaeEncoder encoder) = sides is null
            ? LoadProductionVae(loaders, context.Cancel)
            : sides.LoadVae();
        if (decoder is IDisposable disposableDecoder)
            loaders.Add(disposableDecoder);
        if (encoder is IDisposable disposableEncoder)
            loaders.Add(disposableEncoder);

        (T5TextEncoder umt5, T5Tokenizer tokenizer) = sides is null
            ? LoadProductionUmt5(loaders, context.Cancel)
            : sides.LoadUmt5();
        loaders.Add(umt5);
        loaders.Add(tokenizer);
        // The avatar's audio modules are the audio_proj.* weights; a base checkpoint has none.
        bool hasAudio = checkpoint.Weights.Keys.Any(key => key.StartsWith("audio_proj.", StringComparison.Ordinal));
        return new LongCatParts(dit, checkpoint.Config.InChannels, checkpoint.Config.CaptionChannels, hasAudio, decoder, encoder, umt5, tokenizer, dmdGate,
            opsdGate, cfgStepGate);
    }

    /// <summary>Under <c>numerics.longcatInt8WeightOnly</c>, marks every packed <c>int8_tensorwise</c> linear for a full-precision GEMM
    /// (<see cref="QuantWeightInfo.FullPrecisionMatMul"/>), which the CUDA backend reads: the weight is dequantized and the activations stay as they
    /// are. A weight the weight policy already widened is dense and needs no mark. Returns the number marked; with the knob off, none.</summary>
    internal static int ApplyInt8WeightOnly(IReadOnlyDictionary<string, Tensor> weights)
    {
        if (!EngineKnobs.LongcatInt8WeightOnly.Value)
            return 0;
        int marked = 0;
        foreach (Tensor weight in weights.Values)
        {
            if (weight.DType != DType.I8 || weight.QuantInfo is not { Format: "int8_tensorwise", RowScale: not null } info)
                continue;
            weight.QuantInfo = info with { FullPrecisionMatMul = true };
            marked++;
        }
        Logs.Info($"[LongCatRecipeSupport] numerics.longcatInt8WeightOnly: {marked} INT8 linears marked weight-only (CUDA unpacks each for a dense "
            + "GEMM on every call).");
        return marked;
    }

    /// <summary>The checkpoint's folder. The catalog's primary asset is the first shard, so a file path resolves to the folder it sits in.</summary>
    internal static string CheckpointDirectory(string path) =>
        File.Exists(path) ? Path.GetDirectoryName(path) ?? path : path;

    private static (IWanVaeDecoder, IWanVaeEncoder) LoadProductionVae(List<IDisposable> loaders, CancellationToken cancel)
    {
        string vaePath = ModelDownloader.EnsureSideModelAsync(SideModels.Wan21Vae, onProgress: null, cancel).GetAwaiter().GetResult();
        return VideoRecipeUtils.LoadWanVae(vaePath, isWan21: true, loaders);
    }

    private static (T5TextEncoder, T5Tokenizer) LoadProductionUmt5(List<IDisposable> loaders, CancellationToken cancel)
    {
        string umt5Path = ModelDownloader.EnsureSideModelAsync(SideModels.Umt5Xxl, onProgress: null, cancel).GetAwaiter().GetResult();
        return VideoRecipeUtils.LoadUmt5(umt5Path, loaders);
    }

    /// <summary>Encodes the positive and negative prompts in one umT5 batch, keeping the padded positions as the encoder produced them
    /// (LongCat's DiT zeroes and masks them itself).</summary>
    internal static (LongCatPrompt Positive, LongCatPrompt Negative) EncodePrompts(IBackend backend, LongCatParts parts, string prompt, string negative)
    {
        int[] positiveTokens = parts.Tokenizer.Encode(prompt);
        int[] negativeTokens = parts.Tokenizer.Encode(negative);
        Tensor[] embeds = VideoRecipeUtils.EncodeUmT5PromptBatchKeepingPadding(backend, parts.Umt5, parts.CaptionChannels,
            parts.Tokenizer.MaxLength, positiveTokens, negativeTokens);
        try
        {
            return (ToPrompt(embeds[0], positiveTokens), ToPrompt(embeds[1], negativeTokens));
        }
        finally
        {
            foreach (Tensor embed in embeds)
                embed.Dispose();
        }
    }

    private static LongCatPrompt ToPrompt(Tensor embeds, int[] tokens)
    {
        int validTokens = T5Tokenizer.CreateAttentionMask(tokens).Sum();
        return new LongCatPrompt(embeds.AsReadOnlySpan<float>().ToArray(), tokens.Length, validTokens);
    }

    /// <summary>The initial noise for a latent of <c>[channels, frames, height, width]</c>, from the request's seed.</summary>
    internal static float[] Noise(int seed, int channels, int frames, int height, int width)
    {
        using Tensor noise = SeedGenerator.CreateNoise(new TensorShape([1L, channels, frames, height, width]), seed);
        return noise.AsReadOnlySpan<float>().ToArray();
    }

    /// <summary>The noise of a request's first (or only) generation: the file of <c>diagnostics.longcatNoiseNpy</c> when it is set, else
    /// <see cref="Noise"/>. A chain's later segments keep their seeded noise.</summary>
    internal static float[] InitialNoise(int seed, int channels, int frames, int height, int width)
    {
        using Tensor noise = SavedNoise.Create(EngineKnobs.LongCatNoiseNpy, new TensorShape([1L, channels, frames, height, width]), seed, "LongCat");
        return noise.AsReadOnlySpan<float>().ToArray();
    }

    /// <summary>Rounds each value to the nearest bfloat16, ties to even, in place (<see cref="BFloat16Bits"/>).</summary>
    internal static void RoundToBf16(float[] values)
    {
        ArgumentNullException.ThrowIfNull(values);
        for (int i = 0; i < values.Length; i++)
            values[i] = BFloat16Bits.ToFloat(BFloat16Bits.FromFloat(values[i]));
    }

    /// <summary>Encodes a clip after rounding it to bf16, as the reference does before the VAE encode: the first frame
    /// (pipeline_longcat_video_avatar.py:1152; base I2V pipeline_longcat_video.py:753) and each continuation's context
    /// (generate_avc casts the video at line 1466 before its encode at 1468).</summary>
    internal static Tensor EncodeClipAsBf16(IBackend backend, IWanVaeEncoder encoder, Tensor clip, int channels)
    {
        float[] values = clip.AsReadOnlySpan<float>().ToArray();
        RoundToBf16(values);
        using Tensor rounded = LongCatFrameOps.FromHost(values, clip.Shape);
        return LongCatVaeSeam.EncodeClip(backend, encoder, rounded, channels);
    }

    /// <summary>The normalized latent of a single RGB24 frame, <c>[channels, 1, height / 8, width / 8]</c>, used as the conditioning frame.</summary>
    internal static float[] EncodeFirstFrame(IBackend backend, IWanVaeEncoder encoder, byte[] rgb24, int width, int height, int channels)
    {
        return WithWeightsPlaced(encoder.EnumerateWeights(), backend.PreloadWeights, backend.FreeWeights, () =>
        {
            using Tensor clip = VideoRecipeUtils.TileRgbToClip(rgb24, width, height, 1);
            using Tensor latent = EncodeClipAsBf16(backend, encoder, clip, channels);
            using Tensor host = VideoRecipeUtils.HostCopy(latent);
            return host.AsReadOnlySpan<float>().ToArray();
        }, backend.TrimMemoryPool);
    }

    /// <summary>Runs <paramref name="body"/> with <paramref name="weights"/> placed on the device, and releases them afterwards, also when the body
    /// throws. The VAE's weights need this: a weight used twice without a preload is promoted to a resident weight and then stays on the device until
    /// someone frees it (<c>GpuTransferHelper.ShouldAutoPromote</c>), so a segment's decode or encode would leave the card short for the next segment's
    /// streaming plan. The seam takes the preload and free as delegates, so the order is testable without a device.</summary>
    internal static T WithWeightsPlaced<T>(IEnumerable<Tensor> weights, Action<IEnumerable<Tensor>> preload, Action<IEnumerable<Tensor>> free,
        Func<T> body, Action? trim = null)
    {
        ArgumentNullException.ThrowIfNull(weights);
        ArgumentNullException.ThrowIfNull(preload);
        ArgumentNullException.ThrowIfNull(free);
        ArgumentNullException.ThrowIfNull(body);
        Tensor[] placed = [.. weights];
        preload(placed);
        try
        {
            return body();
        }
        finally
        {
            free(placed);
            // The decode or encode leaves its transients in the allocator pool, where FreeMemoryBytes counts them as used. Returning them here, before
            // any streaming plan reads the card, keeps them out of the next run's budget.
            trim?.Invoke();
        }
    }

    /// <summary>Runs one denoise loop with the DiT placed by a streaming scope, and frees the DiT on every path. The scope is disposed inside
    /// the try (evicting the streamed window), then the DiT's weights are freed in the finally, so a cancel or an exception mid-denoise does not
    /// leave the resident prefix on the device. Each finished step calls the scope's EndStep before the caller's progress callback.</summary>
    internal static float[] RunDenoiseStreamed(IBackend backend, LongCatTransformer dit, Func<BlockStreamingScope> openScope,
        Func<Action<GenerationProgress>, float[]> denoise, Action<GenerationProgress>? progress)
    {
        float[] latents;
        try
        {
            using (BlockStreamingScope stream = openScope())
            {
                latents = denoise(step =>
                {
                    stream.EndStep();
                    progress?.Invoke(step);
                });
            }
            backend.Sync();
        }
        finally
        {
            backend.FreeWeights(dit.EnumerateWeights());
        }
        return latents;
    }

    /// <summary>Decodes a denoised latent of <c>[channels, frames, height, width]</c> to RGB24 frames.</summary>
    internal static byte[][] DecodeFrames(IBackend backend, IWanVaeDecoder decoder, float[] latents, int channels, int frames, int height, int width)
    {
        using Tensor latent = new(new TensorShape([1L, channels, frames, height, width]), DType.F32);
        latents.AsSpan().CopyTo(latent.AsSpan<float>());
        return WithWeightsPlaced(decoder.EnumerateWeights(), backend.PreloadWeights, backend.FreeWeights, () =>
        {
            using Tensor rgb = LongCatVaeSeam.DecodeLatents(backend, decoder, latent, channels);
            return VideoRgbFrames.ExtractAllFrames(rgb);
        }, backend.TrimMemoryPool);
    }

    /// <summary>The 1.5 chain's context (use_vcond False in meituan-longcat/LongCat-Video @ 58eb336, pipeline 1467-1490): the segment's last
    /// <c>1 + 4·(contextLatents − 1)</c> decoded pixel frames, as RGB24 like the reference's uint8 output, re-encoded by the VAE to the context's
    /// latents <c>[channels, contextLatents, hw]</c>. The other decoded frames are released when this returns.</summary>
    internal static float[] ReencodedContext(IBackend backend, IWanVaeDecoder decoder, IWanVaeEncoder encoder, float[] segmentLatents, int channels,
        int segmentFrames, int contextLatents, int latentHeight, int latentWidth)
    {
        CheckSegment(segmentLatents, channels, segmentFrames, latentHeight, latentWidth);
        byte[][] decoded = DecodeFrames(backend, decoder, segmentLatents, channels, segmentFrames, latentHeight, latentWidth);
        return EncodeContext(backend, encoder, decoded, contextLatents, channels, latentHeight, latentWidth);
    }

    /// <summary>Re-encodes the last <c>1 + 4·(contextLatents − 1)</c> frames of an already decoded segment (see <see cref="ReencodedContext"/>), so the
    /// caller can keep the segment's other frames for its output.</summary>
    internal static float[] EncodeContext(IBackend backend, IWanVaeEncoder encoder, byte[][] decoded, int contextLatents, int channels,
        int latentHeight, int latentWidth)
    {
        int contextPixels = 1 + LongCatVaeSeam.TemporalScale * (contextLatents - 1);
        if (decoded.Length < contextPixels)
            throw new InvalidOperationException($"The decoded segment has {decoded.Length} frames; the context needs {contextPixels}.");
        int width = latentWidth * LongCatVaeSeam.SpatialScale, height = latentHeight * LongCatVaeSeam.SpatialScale;
        return WithWeightsPlaced(encoder.EnumerateWeights(), backend.PreloadWeights, backend.FreeWeights, () =>
        {
            using Tensor clip = VideoRecipeUtils.PackRgbFramesToClip(decoded[^contextPixels..], width, height);
            using Tensor latent = EncodeClipAsBf16(backend, encoder, clip, channels);
            using Tensor host = VideoRecipeUtils.HostCopy(latent);
            return host.AsReadOnlySpan<float>().ToArray();
        }, backend.TrimMemoryPool);
    }

    private static void CheckSegment(float[] segmentLatents, int channels, int segmentFrames, int latentHeight, int latentWidth)
    {
        if (segmentLatents.Length != channels * segmentFrames * latentHeight * latentWidth)
            throw new ArgumentException($"The segment holds {segmentLatents.Length} values; expected {channels * segmentFrames * latentHeight * latentWidth}.", nameof(segmentLatents));
    }
}
