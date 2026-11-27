using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Streaming;

namespace HartsyInference.Diffusion.Models.Vae;

/// <summary><see cref="IStreamingVideoDecoder"/> over a causal LTX-2 VAE decoder (upstream
/// <c>CausalVaeStreamingDecoder</c>): the latent is decoded a fixed number of latent frames per decoder pass, every causal
/// conv carrying its input tail in a <see cref="CausalStreamCache"/>. The latent goes in normalized;
/// <see cref="LtxVideo2VaeDecoder"/> un-normalizes it. Borrows the decoder; owns the cache.</summary>
/// <remarks>Upstream decodes one latent frame per pass (<c>decode_per_frame_with_cache</c>), which is the default here. More
/// frames per pass compute the same convolutions over the same inputs in fewer launches.</remarks>
public sealed class LtxVideo2StreamingDecoder : IStreamingVideoDecoder
{
    private const int FrameAxis = 2;

    private readonly IBackend _backend;
    private readonly LtxVideo2VaeDecoder _decoder;
    private readonly CausalStreamCache _cache;
    private readonly int _latentFramesPerPass;
    private int _disposed;

    /// <summary>Wraps a loaded causal decoder.</summary>
    /// <param name="latentFramesPerPass">Latent frames per decoder pass; 1 mirrors upstream.</param>
    /// <exception cref="ArgumentException">The decoder is not causal.</exception>
    public LtxVideo2StreamingDecoder(IBackend backend, LtxVideo2VaeDecoder decoder, int latentFramesPerPass = 1)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(decoder);
        ArgumentOutOfRangeException.ThrowIfLessThan(latentFramesPerPass, 1);
        if (!decoder.IsCausal)
            throw new ArgumentException("A streaming decode needs a causal LTX-2 VAE decoder (decoder_causal: true).", nameof(decoder));
        _backend = backend;
        _decoder = decoder;
        _latentFramesPerPass = latentFramesPerPass;
        _cache = new CausalStreamCache(backend, FrameAxis);
        TemporalStride = decoder.OutputFrames(2) - decoder.OutputFrames(1);
    }

    /// <inheritdoc />
    public int TemporalStride { get; }

    /// <inheritdoc />
    public int DecodedLatentFrames { get; private set; }

    /// <inheritdoc />
    public int PixelFrames(int latentFrames)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(latentFrames, 1);
        return DecodedLatentFrames == 0 ? (latentFrames - 1) * TemporalStride + 1 : latentFrames * TemporalStride;
    }

    /// <inheritdoc />
    public Tensor DecodeNext(Tensor latent)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        ArgumentNullException.ThrowIfNull(latent);
        if (latent.Shape.Rank != 5 || latent.Shape[0] != 1 || latent.Shape[FrameAxis] < 1)
            throw new ArgumentException($"The latent must be [1, C, T, h, w] with at least one frame; got {latent.Shape}.", nameof(latent));
        int frames = (int)latent.Shape[FrameAxis];
        if (frames <= _latentFramesPerPass)
            return DecodePass(latent, frames);

        List<Tensor> parts = [];
        try
        {
            for (int start = 0; start < frames; start += _latentFramesPerPass)
            {
                int count = Math.Min(_latentFramesPerPass, frames - start);
                using Tensor slice = StreamingTensors.Slice(_backend, latent, FrameAxis, start, count);
                parts.Add(DecodePass(slice, count));
            }

            return StreamingTensors.Concat(_backend, parts.ToArray(), FrameAxis);
        }
        finally
        {
            foreach (Tensor part in parts)
                part.Dispose();
        }
    }

    /// <inheritdoc />
    public void Reset()
    {
        _cache.Reset();
        DecodedLatentFrames = 0;
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        _cache.Dispose();
    }

    private Tensor DecodePass(Tensor latent, int frames)
    {
        Tensor rgb = _decoder.DecodeStreaming(_backend, latent, _cache);
        DecodedLatentFrames += frames;
        return rgb;
    }
}
