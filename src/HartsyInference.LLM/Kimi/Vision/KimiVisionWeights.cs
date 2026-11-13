using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.Kimi.Vision;

/// <summary>The F32 tensors of a MoonViT-V2 tower and its patch merger. Owns every tensor it holds and disposes them once.</summary>
public sealed class KimiVisionWeights : IDisposable
{
    private readonly List<Tensor> _owned;
    private int _disposed;

    /// <summary>Owns <paramref name="owned"/>, which must hold every tensor the other arguments reference.</summary>
    internal KimiVisionWeights(List<Tensor> owned, Tensor patchEmbed, Tensor posEmbed, IReadOnlyList<KimiVisionBlockWeights> blocks,
        Tensor finalNorm, Tensor projectorUp, Tensor projectorDown, Tensor projectorNorm)
    {
        ArgumentNullException.ThrowIfNull(owned);
        ArgumentNullException.ThrowIfNull(blocks);
        _owned = owned;
        PatchEmbed = patchEmbed;
        PosEmbed = posEmbed;
        Blocks = blocks;
        FinalNorm = finalNorm;
        ProjectorUp = projectorUp;
        ProjectorDown = projectorDown;
        ProjectorNorm = projectorNorm;
    }

    /// <summary>Patch projection <c>[Hidden, 3·patch²]</c>, bias-free.</summary>
    public Tensor PatchEmbed { get; }

    /// <summary>Learned position table <c>[InitPosGrid², Hidden]</c>, row-major over the grid.</summary>
    public Tensor PosEmbed { get; }

    /// <summary>The encoder blocks, in order.</summary>
    public IReadOnlyList<KimiVisionBlockWeights> Blocks { get; }

    /// <summary>The final RMS norm of the tower <c>[Hidden]</c>.</summary>
    public Tensor FinalNorm { get; }

    /// <summary>Projector's first linear <c>[MergedDim, MergedDim]</c>, bias-free.</summary>
    public Tensor ProjectorUp { get; }

    /// <summary>Projector's second linear <c>[TextHidden, MergedDim]</c>, bias-free.</summary>
    public Tensor ProjectorDown { get; }

    /// <summary>The projector's post RMS norm <c>[TextHidden]</c>.</summary>
    public Tensor ProjectorNorm { get; }

    /// <summary>Every tensor this object owns, empty once disposed.</summary>
    internal IReadOnlyList<Tensor> OwnedTensors => _owned;

    /// <inheritdoc/>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        foreach (Tensor t in _owned) t.Dispose();
        _owned.Clear();
    }
}
