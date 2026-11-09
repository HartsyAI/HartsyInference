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
        return new VideoTemporalTilingOptions
        {
            TileLatentFrames = intervals + 1,
            StrideLatentFrames = strideLatent,
            CompressionRatio = compression,
            BlendLatentFrames = intervals - strideLatent,
        };
    }
}
