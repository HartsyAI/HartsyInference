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

    /// <summary>Separates the vocals and returns them as mono at <paramref name="targetRate"/> (mean of the channels, SoXR-HQ stand-in resample),
    /// cropped to <paramref name="durationSeconds"/> of source when given.</summary>
    internal float[] SeparateMono(float[] left, float[] right, int targetRate, double? durationSeconds = null,
        Action<int, int>? progress = null, CancellationToken cancel = default)
    {
        MdxSeparator separator = Build(null);
        float[][] vocals = separator.Separate(left, right, durationSeconds, progress, cancel);
        int? crop = durationSeconds is null ? null : (int)Math.Min(Math.Ceiling(durationSeconds.Value * SampleRate), left.Length);
        return separator.ToMono(vocals, targetRate, crop);
    }

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
