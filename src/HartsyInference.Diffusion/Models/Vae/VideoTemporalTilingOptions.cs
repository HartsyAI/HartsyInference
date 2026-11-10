namespace HartsyInference.Diffusion.Models.Vae;

/// <summary>Window geometry for <see cref="VideoTemporalTiling"/>. Tile and stride count latent frames; the blend counts
/// latent intervals, so its decoded ramp is <c>BlendLatentFrames * CompressionRatio</c> frames.</summary>
public sealed record VideoTemporalTilingOptions
{
    /// <summary>Latent frames in one decode window, both endpoints included. A window decodes to <c>C·(n−1)+1</c> frames.</summary>
    public required int TileLatentFrames { get; init; }

    /// <summary>Latent frames between consecutive window starts.</summary>
    public required int StrideLatentFrames { get; init; }

    /// <summary>Decoded frames per latent interval: the VAE's temporal compression (8 for LTX-2).</summary>
    public required int CompressionRatio { get; init; }

    /// <summary>Cross-fade width in latent intervals; must not exceed the stride or the window overlap.</summary>
    public required int BlendLatentFrames { get; init; }

    /// <summary>Converts diffusers' decoded-frame settings (<c>tile_sample_min_num_frames</c>,
    /// <c>tile_sample_stride_num_frames</c>) to latent units. Both are decoded spans: diffusers divides them by the
    /// compression ratio before indexing latents (<c>_temporal_tiled_decode</c>, autoencoder_kl_ltx2.py). A span of
    /// <c>k·C</c> decoded frames is <c>k</c> latent intervals, so a window holds <c>k+1</c> latent frames, and the
    /// blend (<c>min − stride</c> decoded frames) is <c>k − stride/C</c> intervals.</summary>
    public static VideoTemporalTilingOptions FromUpstreamSampleFrames(int tileSampleMinFrames, int tileSampleStrideFrames, int compression)
    {
        if (compression < 1 || tileSampleMinFrames % compression != 0 || tileSampleStrideFrames % compression != 0)
        {
            throw new ArgumentException(
                $"Upstream tile spans ({tileSampleMinFrames}, {tileSampleStrideFrames}) must be multiples of the compression ratio {compression}.");
        }

        int intervals = tileSampleMinFrames / compression;
        int strideLatent = tileSampleStrideFrames / compression;
        VideoTemporalTilingOptions options = new()
        {
            TileLatentFrames = intervals + 1,
            StrideLatentFrames = strideLatent,
            CompressionRatio = compression,
            BlendLatentFrames = intervals - strideLatent,
        };
        // An upstream setting that cannot be blended is refused here, where it can be named, not later inside Decode.
        options.Validate();
        return options;
    }

    /// <summary>Checks that the windows overlap enough to blend, and that the blend fits inside both that overlap and one stride, so
    /// no frame is ramped by more than two windows. <see cref="VideoTemporalTiling.Decode"/> applies this before it decodes
    /// anything.</summary>
    /// <exception cref="ArgumentException">The geometry cannot be blended.</exception>
    public void Validate()
    {
        if (CompressionRatio < 1)
        {
            throw new ArgumentException("Compression ratio must be at least 1.");
        }
        if (StrideLatentFrames < 1)
        {
            throw new ArgumentException("Stride must be at least 1 latent frame.");
        }

        // Consecutive windows share this many latent intervals; the blend must fit inside it and inside one stride,
        // so that no frame is ramped by more than two windows.
        int overlap = TileLatentFrames - 1 - StrideLatentFrames;
        int maxBlend = Math.Min(StrideLatentFrames, overlap);
        if (maxBlend < 1)
        {
            throw new ArgumentException($"Tile {TileLatentFrames} with stride {StrideLatentFrames} leaves no overlap to blend across.");
        }
        if (BlendLatentFrames < 1 || BlendLatentFrames > maxBlend)
        {
            throw new ArgumentException($"Blend {BlendLatentFrames} must be in [1, {maxBlend}].");
        }
    }
}
