using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends;

// Short-time Fourier transform.
public partial interface IBackend
{
    /// <summary>Short-time Fourier transform.</summary>
    void Stft(Tensor output, Tensor input, int fftSize, int hopLength, Tensor window);
}
