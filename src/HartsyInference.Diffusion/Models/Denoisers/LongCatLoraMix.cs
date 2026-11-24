using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>The two runtime LoRA gates of the Avatar distilled path: the first LoRA (DMD, <c>loraA</c>) is live for steps before
/// <see cref="MixSteps"/>, the second (OPSD, <c>loraB</c>) from there on, never both (<c>--mix_steps 4</c>, <c>--mix_alpha 1.0</c> in the pinned
/// reference). Attach each LoRA to its gate with <c>LoraStack.AddGated</c>, apply the stack to the weights before
/// <see cref="LongCatTransformer.LoadWeights"/>, and wrap every step's forward in <see cref="ForStep"/>. A single always-on LoRA (the base distill or
/// refinement LoRA) needs only one <see cref="AdjunctGate"/> and no mix.</summary>
/// <remarks>The gate state is read on every GEMM, so the caller serializes generations that share the weights, and a host that captures a graph
/// of a step must key the capture on which gate is live.</remarks>
public sealed class LongCatLoraMix
{
    /// <summary>Creates a mix that hands over from the first LoRA to the second at step <paramref name="mixSteps"/>.</summary>
    public LongCatLoraMix(int mixSteps)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(mixSteps);
        MixSteps = mixSteps;
    }

    /// <summary>Steps (0-based) that run the first LoRA; later steps run the second.</summary>
    public int MixSteps { get; }

    /// <summary>The gate of the first LoRA (DMD).</summary>
    public AdjunctGate First { get; } = new();

    /// <summary>The gate of the second LoRA (OPSD).</summary>
    public AdjunctGate Second { get; } = new();

    /// <summary>The gate that is live at <paramref name="step"/>.</summary>
    public AdjunctGate GateFor(int step)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(step);
        return step < MixSteps ? First : Second;
    }

    /// <summary>Enables the gate of <paramref name="step"/> at <paramref name="strength"/> until the scope is disposed; the other
    /// gate stays off.</summary>
    /// <param name="strength">A finite, non-negative multiplier on the LoRA's own stored scale (a negative stored scale is folded at load).</param>
    /// <remarks>Refuses to enable a gate while the other one is live, which is what a step scope left open from an earlier step would look like.</remarks>
    public IDisposable ForStep(int step, float strength = 1f)
    {
        AdjunctGate gate = GateFor(step);
        AdjunctGate other = ReferenceEquals(gate, First) ? Second : First;
        if (other.Enabled)
            throw new InvalidOperationException("The other LoRA gate is still enabled; a scope from an earlier step was not disposed.");
        return gate.Enable(strength);
    }
}
