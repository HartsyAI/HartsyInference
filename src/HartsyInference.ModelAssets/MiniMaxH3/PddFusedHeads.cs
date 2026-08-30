using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.MiniMaxH3;

/// <summary>One generation step's effective F32 video and audio projection heads.</summary>
public sealed class PddFusedHeads : IDisposable
{
    private int _disposed;

    /// <summary>Creates an owned set of effective projection tensors.</summary>
    public PddFusedHeads(Tensor videoWeight, Tensor videoBias, Tensor audioWeight, Tensor audioBias)
    {
        VideoWeight = videoWeight ?? throw new ArgumentNullException(nameof(videoWeight));
        VideoBias = videoBias ?? throw new ArgumentNullException(nameof(videoBias));
        AudioWeight = audioWeight ?? throw new ArgumentNullException(nameof(audioWeight));
        AudioBias = audioBias ?? throw new ArgumentNullException(nameof(audioBias));
    }

    /// <summary>Effective video projection weight.</summary>
    public Tensor VideoWeight { get; }

    /// <summary>Effective video projection bias.</summary>
    public Tensor VideoBias { get; }

    /// <summary>Effective audio projection weight.</summary>
    public Tensor AudioWeight { get; }

    /// <summary>Effective audio projection bias.</summary>
    public Tensor AudioBias { get; }

    /// <summary>Releases the four effective head tensors.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        VideoWeight.Dispose();
        VideoBias.Dispose();
        AudioWeight.Dispose();
        AudioBias.Dispose();
    }
}
