using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines.LongCat;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.ModelAssets.Tokenizers;
using HartsyInference.Video;

namespace HartsyInference.Engine.Recipes.Video;

/// <summary>What every LongCat recipe loads: the DiT from the checkpoint directory, the VAE, and umT5 with its tokenizer. The latent channel count
/// is the DiT's own <see cref="LongCatConfig.InChannels"/>, so a tiny config and the release (16 channels) share one path.</summary>
internal sealed record LongCatParts(LongCatTransformer Dit, int Channels, int CaptionChannels, bool HasAudio, IWanVaeDecoder Decoder,
    IWanVaeEncoder Encoder, T5TextEncoder Umt5, T5Tokenizer Tokenizer);

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

    /// <summary>Loads the shared parts. Everything disposable goes into <paramref name="loaders"/>, which the caller owns from here on.
    /// <paramref name="sides"/> replaces the umT5 and VAE downloads when it is not null.</summary>
    internal static LongCatParts LoadParts(RecipeContext context, List<IDisposable> loaders, LongCatSideModels? sides)
    {
        LongCatCheckpoint checkpoint = LongCatCheckpoint.Open(CheckpointDirectory(context.CheckpointPath));
        loaders.Add(checkpoint);
        if (checkpoint.IsInt8)
        {
            throw new NotSupportedException(
                $"'{context.CheckpointPath}' is an INT8 LongCat checkpoint. The INT8 weight route is not wired into the LongCat recipes yet.");
        }
        LongCatTransformer dit = new(checkpoint.Config);
        loaders.Add(dit);
        dit.LoadWeights(checkpoint.Weights);

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
        return new LongCatParts(dit, checkpoint.Config.InChannels, checkpoint.Config.CaptionChannels, hasAudio, decoder, encoder, umt5, tokenizer);
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
        using Tensor clip = VideoRecipeUtils.TileRgbToClip(rgb24, width, height, 1);
        using Tensor latent = LongCatVaeSeam.EncodeClip(backend, encoder, clip, channels);
        using Tensor host = VideoRecipeUtils.HostCopy(latent);
        return host.AsReadOnlySpan<float>().ToArray();
    }

    /// <summary>Decodes a denoised latent of <c>[channels, frames, height, width]</c> to RGB24 frames.</summary>
    internal static byte[][] DecodeFrames(IBackend backend, IWanVaeDecoder decoder, float[] latents, int channels, int frames, int height, int width)
    {
        using Tensor latent = new(new TensorShape([1L, channels, frames, height, width]), DType.F32);
        latents.AsSpan().CopyTo(latent.AsSpan<float>());
        using Tensor rgb = LongCatVaeSeam.DecodeLatents(backend, decoder, latent, channels);
        return VideoRgbFrames.ExtractAllFrames(rgb);
    }
}
