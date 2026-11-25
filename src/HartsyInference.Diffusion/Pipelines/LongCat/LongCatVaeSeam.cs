using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Vae;

namespace HartsyInference.Diffusion.Pipelines.LongCat;

/// <summary>The VAE seam of the LongCat-Video family: the pixel clip to the normalized latent the DiT denoises, and back. The LongCat VAE is the
/// Wan 2.1 VAE (the base repo's <c>autoencoder_kl_wan</c>, 4× temporal and 8× spatial compression, 16 latent channels); the encoder and decoder
/// here are the existing Wan classes, and the normalization is inside them (the latent is already normalized on both sides).</summary>
/// <remarks>
/// The seam checks the shapes the pipeline relies on. A clip must be <c>[1, 3, T, H, W]</c> with <c>T = 1 + 4·k</c> and <c>H</c> and <c>W</c>
/// multiples of 8; a latent must be <c>[1, channels, F, h, w]</c> with the DiT's channel count. The encoder's and decoder's returned shapes are
/// checked against those, and a wrong shape disposes the returned tensor before it throws. Pixel and latent values are not checked: that needs
/// the VAE weights (the real-weight run, B6). Decode is one call over the whole latent; the reference decodes in chunks, which is a memory
/// choice, not a numerical one, and is not implemented here.
/// </remarks>
public static class LongCatVaeSeam
{
    /// <summary>Video frames per latent frame (the Wan VAE's temporal compression).</summary>
    public const int TemporalScale = 4;

    /// <summary>Pixels per latent cell in each spatial direction (the Wan VAE's spatial compression).</summary>
    public const int SpatialScale = 8;

    /// <summary>Encodes an RGB clip <c>[1, 3, T, H, W]</c> in [-1, 1] to the normalized latent <c>[1, channels, 1 + (T−1)/4, H/8, W/8]</c>.</summary>
    public static Tensor EncodeClip(IBackend backend, IWanVaeEncoder encoder, Tensor rgbClip, int latentChannels)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(encoder);
        ArgumentNullException.ThrowIfNull(rgbClip);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(latentChannels);
        if (rgbClip.Shape.Rank != 5 || rgbClip.Shape[0] != 1 || rgbClip.Shape[1] != 3)
            throw new ArgumentException($"The clip must be [1, 3, T, H, W], got {rgbClip.Shape}.", nameof(rgbClip));
        long frames = rgbClip.Shape[2], height = rgbClip.Shape[3], width = rgbClip.Shape[4];
        if (frames < 1 || (frames - 1) % TemporalScale != 0)
            throw new ArgumentException($"The clip must have 1 + {TemporalScale}·k frames, got {frames}.", nameof(rgbClip));
        if (height % SpatialScale != 0 || width % SpatialScale != 0)
            throw new ArgumentException($"The clip's height and width must be multiples of {SpatialScale}, got {height}×{width}.", nameof(rgbClip));

        Tensor latent = encoder.Encode(backend, rgbClip);
        long expectedFrames = (frames - 1) / TemporalScale + 1;
        try
        {
            RequireShape(latent, latentChannels, expectedFrames, height / SpatialScale, width / SpatialScale, "encoded latent");
        }
        catch
        {
            latent.Dispose();
            throw;
        }
        return latent;
    }

    /// <summary>Decodes a normalized latent <c>[1, channels, F, h, w]</c> to an RGB clip <c>[1, 3, 1 + 4·(F−1), 8h, 8w]</c> in [-1, 1].</summary>
    public static Tensor DecodeLatents(IBackend backend, IWanVaeDecoder decoder, Tensor latent, int latentChannels)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(decoder);
        ArgumentNullException.ThrowIfNull(latent);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(latentChannels);
        if (latent.Shape.Rank != 5 || latent.Shape[0] != 1 || latent.Shape[1] != latentChannels)
            throw new ArgumentException($"The latent must be [1, {latentChannels}, F, h, w], got {latent.Shape}.", nameof(latent));
        long frames = latent.Shape[2], h = latent.Shape[3], w = latent.Shape[4];
        if (frames < 1)
            throw new ArgumentException($"The latent must have at least one frame, got {frames}.", nameof(latent));
        Tensor clip = decoder.Decode(backend, latent);
        try
        {
            RequireShape(clip, 3, 1 + TemporalScale * (frames - 1), h * SpatialScale, w * SpatialScale, "decoded clip");
        }
        catch
        {
            clip.Dispose();
            throw;
        }
        return clip;
    }

    private static void RequireShape(Tensor tensor, long channels, long frames, long height, long width, string what)
    {
        if (tensor.Shape.Rank != 5 || tensor.Shape[0] != 1 || tensor.Shape[1] != channels || tensor.Shape[2] != frames
            || tensor.Shape[3] != height || tensor.Shape[4] != width)
            throw new InvalidOperationException($"The {what} is {tensor.Shape}; expected [1, {channels}, {frames}, {height}, {width}].");
    }
}
