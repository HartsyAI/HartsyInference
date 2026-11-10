using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Vae;

/// <summary>A video VAE decoder driven a few latent frames at a time, keeping its temporal context between calls, so a
/// stream's pixels can be delivered while later latents are still being generated.</summary>
public interface IStreamingVideoDecoder : IDisposable
{
    /// <summary>Pixel frames per latent frame after the stream's first latent frame.</summary>
    int TemporalStride { get; }

    /// <summary>Latent frames decoded since the stream started or was last reset.</summary>
    int DecodedLatentFrames { get; }

    /// <summary>Decodes the next latent frames of the stream. Returns F32 RGB <c>[1, 3, F, H, W]</c> in [-1, 1] holding only the
    /// frames these latents add: <see cref="PixelFrames"/> of them. The caller owns the result and still owns
    /// <paramref name="latent"/>.</summary>
    /// <param name="latent">Normalized latent <c>[1, C, T, h, w]</c>, the frames that follow those already decoded.</param>
    Tensor DecodeNext(Tensor latent);

    /// <summary>Pixel frames the next <paramref name="latentFrames"/> latent frames decode to: the stream's first latent
    /// frame gives one frame, every other one <see cref="TemporalStride"/>.</summary>
    int PixelFrames(int latentFrames);

    /// <summary>Drops the temporal context; the next call starts a new stream.</summary>
    void Reset();
}
