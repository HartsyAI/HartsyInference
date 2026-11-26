using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
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
    IWanVaeEncoder Encoder, T5TextEncoder Umt5, T5Tokenizer Tokenizer, AdjunctGate? DmdGate = null);

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
        if (cacheBytes > 0)
        {
            // The planner's own inputs, read after the shared weights are placed and the pool trimmed, as BlockStreamingScope does.
            backend.PreloadWeights(parts.Dit.EnumerateSharedWeights());
            backend.TrimMemoryPool();
            RequireCacheRoom(backend.FreeMemoryBytes(), headroom - cacheBytes, cacheBytes);
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

    /// <summary>The headroom a run reserves: the fixed pool and the safety margin, the activations, and the cache. The activations are the per-token
    /// estimate, or <paramref name="measuredPeakBytes"/> when a measured peak of the run's activations is given in its place.</summary>
    internal static long BudgetHeadroomBytes(long tokens, int hidden, int ffnHidden, long? measuredPeakBytes, long cacheBytes) =>
        measuredPeakBytes is long peak
            ? FixedPoolBytes + SafetyBytes + peak + cacheBytes
            : ActivationHeadroomBytes(tokens, hidden, ffnHidden) + cacheBytes;

    /// <summary>Refuses a run whose budget is negative: the free memory after the shared weights, less the activation headroom and the cache, must be
    /// at least zero, or the streamed blocks have nowhere to land and the first step dies out of memory.</summary>
    internal static void RequireCacheRoom(long freeAfterShared, long headroomBeforeCache, long cacheBytes)
    {
        long spendable = freeAfterShared - headroomBeforeCache - cacheBytes;
        if (spendable < 0)
        {
            throw new InvalidOperationException(
                $"The chain's later run needs {Mb(headroomBeforeCache)} MB of activation headroom and {Mb(cacheBytes)} MB of KV cache, but only " +
                $"{Mb(freeAfterShared)} MB is free after the shared weights: {Mb(-spendable)} MB short. Lower the resolution or the chain length.");
        }
    }

    private static long Mb(long bytes) => bytes >> 20;

    /// <summary>Loads the shared parts. Everything disposable goes into <paramref name="loaders"/>, which the caller owns from here on.
    /// <paramref name="sides"/> replaces the umT5 and VAE downloads when it is not null.</summary>
    /// <summary>The DMD distillation LoRA's static multiplier: the reference's <c>--lora_multiplier</c> default, applied when the LoRA is gated on.</summary>
    internal const float DmdMultiplier = 0.8f;

    /// <summary>Where the 1.5 distillation LoRA sits: <c>lora/dmd_lora.safetensors</c> beside the checkpoint's own folder (the reference layout).</summary>
    internal static string DmdLoraPath(string checkpointPath) =>
        Path.Combine(Path.GetDirectoryName(CheckpointDirectory(checkpointPath)) ?? ".", "lora", "dmd_lora.safetensors");

    /// <summary>Loads the parts. With <paramref name="attachDmd"/>, the DMD LoRA is attached as a runtime gated term on the DiT's weights (the
    /// INT8 base included), so it is live only while its <see cref="AdjunctGate"/> is enabled. A missing LoRA file is refused here, before any
    /// weight is read, so a distilled request never runs on the bare base.</summary>
    internal static LongCatParts LoadParts(RecipeContext context, List<IDisposable> loaders, LongCatSideModels? sides, bool attachDmd = false)
    {
        LongCatCheckpoint checkpoint = LongCatCheckpoint.Open(CheckpointDirectory(context.CheckpointPath));
        loaders.Add(checkpoint);
        // An INT8 checkpoint arrives as dense keys whose linears are I8 with their row scale attached (LongCatCheckpoint). The weight policy
        // keeps an int8 linear packed only where a backend runs it, and widens it elsewhere, before the DiT captures the weights.
        Dictionary<string, Tensor> weights = new(checkpoint.Weights, StringComparer.Ordinal);
        loaders.Add(QuantizedWeightPolicy.PrepareForBackends(weights, context.TransformerBackends, DType.F32));
        AdjunctGate? dmdGate = null;
        LoraStack? loras = null;
        LoraFile? dmd = null;
        if (attachDmd)
        {
            string dmdPath = DmdLoraPath(context.CheckpointPath);
            if (!File.Exists(dmdPath))
                throw new NotSupportedException($"The distilled build needs its distillation LoRA at '{dmdPath}', which is not there. Refusing to run the bare base as the distilled model.");
            dmd = LoraFile.Load(dmdPath);
            loras = new LoraStack();
            dmdGate = new AdjunctGate();
            loras.AddGated(dmd, dmdGate, DmdMultiplier);
            loras.ApplyTo(weights, LoraTarget.Transformer, context.Backend);
        }
        LongCatTransformer dit = new(checkpoint.Config);
        loaders.Add(dit);
        dit.LoadWeights(weights);
        if (dmd is not null)
        {
            loaders.Add(dmd);
            loaders.Add(loras!);
        }

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
        return new LongCatParts(dit, checkpoint.Config.InChannels, checkpoint.Config.CaptionChannels, hasAudio, decoder, encoder, umt5, tokenizer, dmdGate);
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

    /// <summary>The normalized latent of a single RGB24 frame, <c>[channels, 1, height / 8, width / 8]</c>, used as the conditioning frame.</summary>
    internal static float[] EncodeFirstFrame(IBackend backend, IWanVaeEncoder encoder, byte[] rgb24, int width, int height, int channels)
    {
        return WithWeightsPlaced(encoder.EnumerateWeights(), backend.PreloadWeights, backend.FreeWeights, () =>
        {
            using Tensor clip = VideoRecipeUtils.TileRgbToClip(rgb24, width, height, 1);
            using Tensor latent = LongCatVaeSeam.EncodeClip(backend, encoder, clip, channels);
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
            using Tensor latent = LongCatVaeSeam.EncodeClip(backend, encoder, clip, channels);
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
