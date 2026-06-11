using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Core.Models;

/// <summary>Continuous audio-latent decoder (waveform VAE decoder half): latents <c>[B, latent_dim, T]</c> → PCM
/// <c>[B, channels, T · hop]</c>. Lets pipelines in one package (e.g. ACE-Step 1.5 in Diffusion) decode through a
/// codec that lives in another (Oobleck in Audio) without a cross-package reference — the composition root wires
/// the concrete VAE in.</summary>
public interface IAudioLatentDecoder
{
    /// <summary>Decodes latents <c>[B, latent_dim, T]</c> to PCM <c>[B, audio_channels, T · hop]</c> in ≈[-1, 1].</summary>
    Tensor Decode(IBackend backend, Tensor latent);

    /// <summary>Enumerates weight tensors for GPU preload/free pairing.</summary>
    IEnumerable<Tensor> EnumerateWeights();
}
