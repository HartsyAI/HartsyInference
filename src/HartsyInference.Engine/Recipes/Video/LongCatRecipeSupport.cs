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

/// <summary>What every LongCat recipe loads: the DiT from the checkpoint directory, the Wan 2.1 VAE, and umT5 with its tokenizer.
/// The shared host-side steps (prompt encode, noise, first-frame conditioning, latent decode) live here too, so the base and avatar
/// families run one implementation.</summary>
internal sealed record LongCatParts(LongCatTransformer Dit, int CaptionChannels, IWanVaeDecoder Decoder, IWanVaeEncoder Encoder,
    T5TextEncoder Umt5, T5Tokenizer Tokenizer);

/// <summary>The shared LongCat recipe support. The VAE is the Wan 2.1 z=16 VAE, which LongCat shares with Wan.</summary>
internal static class LongCatRecipeSupport
{
    /// <summary>Latent channels of the Wan 2.1 VAE.</summary>
    internal const int LatentChannels = 16;

    /// <summary>Temporal compression of the Wan 2.1 VAE: a clip of <c>1 + 4·(latentFrames − 1)</c> frames.</summary>
    internal const int TemporalCompressionFactor = 4;

    /// <summary>Spatial compression of the Wan 2.1 VAE: width and height are multiples of 8.</summary>
    internal const int SpatialCompressionFactor = 8;

    /// <summary>Flow shift of the base family's sampler (the research plan's base value). Avatar uses 7; see the avatar recipe.</summary>
    internal const float BaseFlowShift = 12f;

    /// <summary>Loads the shared parts. Everything disposable goes into <paramref name="loaders"/>, which the caller owns from here on.</summary>
    internal static LongCatParts LoadParts(RecipeContext context, List<IDisposable> loaders)
    {
        string umt5Path = ModelDownloader.EnsureSideModelAsync(SideModels.Umt5Xxl, onProgress: null, context.Cancel).GetAwaiter().GetResult();
        string vaePath = ModelDownloader.EnsureSideModelAsync(SideModels.Wan21Vae, onProgress: null, context.Cancel).GetAwaiter().GetResult();

        LongCatCheckpoint checkpoint = LongCatCheckpoint.Open(context.CheckpointPath);
        loaders.Add(checkpoint);
        if (checkpoint.IsInt8)
        {
            throw new NotSupportedException(
                $"'{context.CheckpointPath}' is an INT8 LongCat checkpoint. The INT8 weight route is not wired into the LongCat recipes yet.");
        }
        LongCatTransformer dit = new(checkpoint.Config);
        loaders.Add(dit);
        dit.LoadWeights(checkpoint.Weights);

        (IWanVaeDecoder decoder, IWanVaeEncoder encoder) = VideoRecipeUtils.LoadWanVae(vaePath, isWan21: true, loaders);
        if (decoder is IDisposable disposableDecoder)
            loaders.Add(disposableDecoder);
        if (encoder is IDisposable disposableEncoder)
            loaders.Add(disposableEncoder);

        (T5TextEncoder umt5, T5Tokenizer tokenizer) = VideoRecipeUtils.LoadUmt5(umt5Path, loaders);
        loaders.Add(umt5);
        loaders.Add(tokenizer);
        return new LongCatParts(dit, checkpoint.Config.CaptionChannels, decoder, encoder, umt5, tokenizer);
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
    internal static float[] Noise(int seed, int frames, int height, int width)
    {
        using Tensor noise = SeedGenerator.CreateNoise(new TensorShape([1L, LatentChannels, frames, height, width]), seed);
        return noise.AsReadOnlySpan<float>().ToArray();
    }

    /// <summary>The normalized latent of a single RGB24 frame, <c>[channels, 1, height / 8, width / 8]</c>, used as the conditioning frame.</summary>
    internal static float[] EncodeFirstFrame(IBackend backend, IWanVaeEncoder encoder, byte[] rgb24, int width, int height)
    {
        using Tensor clip = VideoRecipeUtils.TileRgbToClip(rgb24, width, height, 1);
        using Tensor latent = LongCatVaeSeam.EncodeClip(backend, encoder, clip, LatentChannels);
        using Tensor host = VideoRecipeUtils.HostCopy(latent);
        return host.AsReadOnlySpan<float>().ToArray();
    }

    /// <summary>Decodes a denoised latent of <c>[channels, frames, height, width]</c> to RGB24 frames.</summary>
    internal static byte[][] DecodeFrames(IBackend backend, IWanVaeDecoder decoder, float[] latents, int frames, int height, int width)
    {
        using Tensor latent = new(new TensorShape([1L, LatentChannels, frames, height, width]), DType.F32);
        latents.AsSpan().CopyTo(latent.AsSpan<float>());
        using Tensor rgb = LongCatVaeSeam.DecodeLatents(backend, decoder, latent, LatentChannels);
        return VideoRgbFrames.ExtractAllFrames(rgb);
    }
}
