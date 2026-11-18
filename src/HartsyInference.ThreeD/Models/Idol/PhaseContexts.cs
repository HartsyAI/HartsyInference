using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.Idol;

/// <summary>Idol-turbo's image stream under each phase expert, from <see cref="IdolTurboAdapters.EmbedContexts"/>; owns both tensors.</summary>
public sealed class PhaseContexts : IDisposable
{
    internal PhaseContexts(Tensor slow, Tensor fast)
    {
        Slow = slow;
        Fast = fast;
    }

    /// <summary>The stream for the high-noise steps.</summary>
    public Tensor Slow { get; }

    /// <summary>The stream for the low-noise steps.</summary>
    public Tensor Fast { get; }

    /// <summary>The stream of the phase a pass at <paramref name="tModel"/> (Idol's <c>1000·t</c>) is in.</summary>
    public Tensor At(float tModel) => IdolTurboAdapters.IsSlow(tModel) ? Slow : Fast;

    /// <summary>Releases both streams.</summary>
    public void Dispose()
    {
        Slow.Dispose();
        Fast.Dispose();
    }
}
