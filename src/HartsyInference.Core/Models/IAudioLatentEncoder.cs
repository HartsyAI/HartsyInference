using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Models;

/// <summary>Encode half of a waveform VAE: PCM <c>[B, channels, T]</c> → deterministic latent mean
/// <c>[B, latent_dim, T / hop]</c>. Pipelines test their <see cref="IAudioLatentDecoder"/> for this interface when
/// they also need conditioning latents (e.g. ACE-Step 1.5's silence/reference-audio encode) — encode-capable codecs
/// (Oobleck with encoder weights) implement both.</summary>
public interface IAudioLatentEncoder
{
    /// <summary>False when the loaded checkpoint is decode-only (no encoder weights).</summary>
    bool CanEncode { get; }

    /// <summary>PCM channel count the encoder expects (1 = mono, 2 = stereo).</summary>
    int AudioChannels { get; }

    /// <summary>Encodes PCM <c>[B, channels, T]</c> (T a multiple of the hop) to the latent mean
    /// <c>[B, latent_dim, T / hop]</c> — the distribution mode, the convention diffusion conditioning uses.</summary>
    Tensor EncodeMode(IBackend backend, Tensor pcm);
}
