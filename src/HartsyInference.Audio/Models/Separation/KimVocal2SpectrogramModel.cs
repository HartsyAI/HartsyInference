using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Separation;

/// <summary>Runs a <see cref="KimVocal2Network"/> as an <see cref="IMdxSpectrogramModel"/> on one backend.</summary>
public sealed class KimVocal2SpectrogramModel(IBackend backend, KimVocal2Network network, int frames) : IMdxSpectrogramModel
{
    private readonly IBackend _backend = backend ?? throw new ArgumentNullException(nameof(backend));
    private readonly KimVocal2Network _network = network ?? throw new ArgumentNullException(nameof(network));

    /// <inheritdoc/>
    public unsafe void Run(float[] spectrogram, float[] prediction)
    {
        long count = (long)_network.InChannels * _network.DimF * frames;
        if (spectrogram.Length != count || prediction.Length != count)
            throw new ArgumentException($"Expected {count} values for [{_network.InChannels}, {_network.DimF}, {frames}].");
        using Tensor input = new(new TensorShape(1, _network.InChannels, _network.DimF, frames), DType.F32);
        spectrogram.AsSpan().CopyTo(new Span<float>((void*)input.DataPointer, spectrogram.Length));
        using Tensor output = _network.Forward(_backend, input);
        new ReadOnlySpan<float>((void*)output.DataPointer, prediction.Length).CopyTo(prediction);
    }
}
