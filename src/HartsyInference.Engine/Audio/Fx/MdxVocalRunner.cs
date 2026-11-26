using HartsyInference.Audio.Models.Separation;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Engine.Audio;

/// <summary>A loaded Kim_Vocal_2 vocal separator: the network plus the F32 weights read from its <c>.onnx</c>. The weights are host tensors, so one
/// cached runner serves whichever backend each call names; each backend holds its own device copies.</summary>
internal sealed class MdxVocalRunner(KimVocal2Network network, Dictionary<string, Tensor> weights) : IDisposable
{
    /// <summary>Sample rate the network was trained at; input must be decoded to it.</summary>
    internal int SampleRate => MdxSettings.KimVocal2.SampleRate;

    /// <summary>Separates the vocals from a 44.1 kHz stereo mix on <paramref name="backend"/>, as <see cref="MdxSeparator.Separate"/>.</summary>
    internal float[][] Separate(IBackend backend, float[] left, float[] right, double? overlap = null, double? durationSeconds = null,
        Action<int, int>? progress = null, CancellationToken cancel = default)
    {
        MdxSettings settings = overlap is null ? MdxSettings.KimVocal2 : MdxSettings.KimVocal2 with { Overlap = overlap.Value };
        MdxSeparator separator = new(settings, new KimVocal2SpectrogramModel(backend, network, settings.Frames));
        return separator.Separate(left, right, durationSeconds, progress, cancel);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        network.Dispose();
        foreach (Tensor t in weights.Values) t.Dispose();
        weights.Clear();
    }
}
