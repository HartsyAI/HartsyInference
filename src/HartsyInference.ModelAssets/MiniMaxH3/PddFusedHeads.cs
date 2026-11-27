using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.MiniMaxH3;

/// <summary>One generation step's effective F32 video head and, for the audio build, its audio head. The image build
/// has no audio head, so the audio pair is absent together or present together.</summary>
public sealed class PddFusedHeads : IDisposable
{
    private int _disposed;

    /// <summary>Creates an owned set of effective projection tensors.</summary>
    public PddFusedHeads(Tensor videoWeight, Tensor videoBias, Tensor? audioWeight, Tensor? audioBias)
    {
        VideoWeight = videoWeight ?? throw new ArgumentNullException(nameof(videoWeight));
        VideoBias = videoBias ?? throw new ArgumentNullException(nameof(videoBias));
        if ((audioWeight is null) != (audioBias is null))
        {
            throw new ArgumentException("The audio head needs both a weight and a bias, or neither.");
        }

        AudioWeight = audioWeight;
        AudioBias = audioBias;
    }

    /// <summary>Effective video projection weight.</summary>
    public Tensor VideoWeight { get; }

    /// <summary>Effective video projection bias.</summary>
    public Tensor VideoBias { get; }

    /// <summary>Effective audio projection weight, or null when the checkpoint has no audio head.</summary>
    public Tensor? AudioWeight { get; }

    /// <summary>Effective audio projection bias, or null when the checkpoint has no audio head.</summary>
    public Tensor? AudioBias { get; }

    /// <summary>Releases the effective head tensors.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        VideoWeight.Dispose();
        VideoBias.Dispose();
        AudioWeight?.Dispose();
        AudioBias?.Dispose();
    }
}
