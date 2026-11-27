using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.World.Camera;

namespace HartsyInference.World.Pipelines.SanaWm;

/// <summary>The clip decode over a loaded LTX-2 VAE decoder, window by window through <see cref="VideoTemporalTiling"/>. The
/// latent goes in normalized: <see cref="LtxVideo2VaeDecoder.Decode"/> un-normalizes it, so nothing here does. Borrows the
/// decoder; its owner releases it.</summary>
public sealed class SanaWmLtx2VideoDecoder : ISanaWmVideoDecoder
{
    private readonly IBackend _backend;
    private readonly LtxVideo2VaeDecoder _decoder;
    private readonly VideoTemporalTilingOptions _tiling;

    /// <summary>Wraps a loaded decoder with upstream's temporal windows (96 decoded frames, stride 64).</summary>
    public SanaWmLtx2VideoDecoder(IBackend backend, LtxVideo2VaeDecoder decoder)
        : this(backend, decoder, VideoTemporalTilingOptions.FromUpstreamSampleFrames(
            SanaWmPipeline.DecodeWindowFrames, SanaWmPipeline.DecodeStrideFrames, SanaWmCameraConditioning.TemporalStride))
    {
    }

    /// <summary>Wraps a loaded decoder with explicit temporal windows.</summary>
    public SanaWmLtx2VideoDecoder(IBackend backend, LtxVideo2VaeDecoder decoder, VideoTemporalTilingOptions tiling)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(decoder);
        ArgumentNullException.ThrowIfNull(tiling);
        _backend = backend;
        _decoder = decoder;
        _tiling = tiling;
    }

    /// <inheritdoc/>
    public Tensor Decode(Tensor latent) => VideoTemporalTiling.Decode(latent, _tiling, window => _decoder.Decode(_backend, window));
}
