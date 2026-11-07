using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends.Gating;

/// <summary>What <see cref="IBackend.LinearFused"/> folds around a projection: an optional <see cref="IBackend.GroupScale"/> of the
/// input before the GEMM and an activation of the output after it. A backend whose GEMM already walks the input (to quantize
/// it) or the output (to store it) applies them in that pass; the default runs the separate ops.</summary>
/// <param name="Epilogue">The activation of the output.</param>
/// <param name="PrologueGates">The gates of the input's group scale, or null for none.</param>
/// <param name="Prologue">The group scale of the input, read only with <paramref name="PrologueGates"/>.</param>
public readonly record struct LinearFusion(LinearActivation Epilogue, Tensor? PrologueGates = null, GroupScaleSpec Prologue = default)
{
    /// <summary>GELU after the projection.</summary>
    public static LinearFusion Gelu => new(LinearActivation.GeluTanh);

    /// <summary>LTX-2's doubled-sigmoid gate of every head before the output projection.</summary>
    public static LinearFusion HeadGated(Tensor gateLogits, int heads, int headDim)
        => new(LinearActivation.None, gateLogits, GroupScaleSpec.PerGroup(headDim, heads, GateFunction.TwoSigmoid));
}
