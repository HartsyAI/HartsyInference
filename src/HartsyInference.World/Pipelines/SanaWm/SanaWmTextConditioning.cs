using HartsyInference.Core.Tensors;

namespace HartsyInference.World.Pipelines.SanaWm;

/// <summary>The caption features of a SANA-WM run: the prompt and the negative prompt, each <c>[1, 1, 300, 2304]</c> F32 with an
/// F32 validity mask <c>[1, 300]</c>, all on the host. Owns the four tensors.</summary>
public sealed class SanaWmTextConditioning : IDisposable
{
    private Tensor? _condition;
    private Tensor? _conditionMask;
    private Tensor? _negative;
    private Tensor? _negativeMask;

    /// <summary>Takes ownership of the four tensors.</summary>
    public SanaWmTextConditioning(Tensor condition, Tensor conditionMask, Tensor negative, Tensor negativeMask)
    {
        ArgumentNullException.ThrowIfNull(condition);
        ArgumentNullException.ThrowIfNull(conditionMask);
        ArgumentNullException.ThrowIfNull(negative);
        ArgumentNullException.ThrowIfNull(negativeMask);
        _condition = condition;
        _conditionMask = conditionMask;
        _negative = negative;
        _negativeMask = negativeMask;
    }

    /// <summary>Features of the CHI-prefixed prompt.</summary>
    public Tensor Condition => _condition ?? throw Released();

    /// <summary>Validity of <see cref="Condition"/>'s positions.</summary>
    public Tensor ConditionMask => _conditionMask ?? throw Released();

    /// <summary>Features of the negative prompt.</summary>
    public Tensor Negative => _negative ?? throw Released();

    /// <summary>Validity of <see cref="Negative"/>'s positions.</summary>
    public Tensor NegativeMask => _negativeMask ?? throw Released();

    /// <summary>The features of <paramref name="branch"/>.</summary>
    public Tensor Features(SanaWmGuidanceBranch branch) => branch == SanaWmGuidanceBranch.Conditional ? Condition : Negative;

    /// <summary>The mask of <paramref name="branch"/>.</summary>
    public Tensor Mask(SanaWmGuidanceBranch branch) => branch == SanaWmGuidanceBranch.Conditional ? ConditionMask : NegativeMask;

    /// <inheritdoc/>
    public void Dispose()
    {
        Interlocked.Exchange(ref _condition, null)?.Dispose();
        Interlocked.Exchange(ref _conditionMask, null)?.Dispose();
        Interlocked.Exchange(ref _negative, null)?.Dispose();
        Interlocked.Exchange(ref _negativeMask, null)?.Dispose();
    }

    private static ObjectDisposedException Released() => new(nameof(SanaWmTextConditioning));
}
