using HartsyInference.Audio.Models.Separation;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Engine.Audio;

/// <summary>A loaded Kim_Vocal_2 vocal separator: the network plus the F32 weights read from its <c>.onnx</c>, bound to one backend.</summary>
internal sealed class MdxVocalRunner(KimVocal2Network network, Dictionary<string, Tensor> weights, IBackend backend) : IDisposable
{
    /// <summary>Sample rate the network was trained at; input must be decoded to it.</summary>
    internal int SampleRate => MdxSettings.KimVocal2.SampleRate;

    /// <summary>Separates the vocals from a 44.1 kHz stereo mix, as <see cref="MdxSeparator.Separate"/>.</summary>
    internal float[][] Separate(float[] left, float[] right, double? overlap = null, double? durationSeconds = null,
        Action<int, int>? progress = null, CancellationToken cancel = default) =>
        Build(overlap).Separate(left, right, durationSeconds, progress, cancel);

    /// <summary>The reference steps after separation, on the float stem: mono mix and SoXR-HQ stand-in resample (<see cref="MdxSeparator.ToMono"/>).</summary>
    internal float[] ToMono(float[][] stem, int targetRate, int? targetSamplesAtSourceRate) =>
        Build(null).ToMono(stem, targetRate, targetSamplesAtSourceRate);

    private MdxSeparator Build(double? overlap)
    {
        MdxSettings settings = overlap is null ? MdxSettings.KimVocal2 : MdxSettings.KimVocal2 with { Overlap = overlap.Value };
        return new MdxSeparator(settings, new KimVocal2SpectrogramModel(backend, network, settings.Frames));
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        network.Dispose();
        foreach (Tensor t in weights.Values) t.Dispose();
        weights.Clear();
    }
}
