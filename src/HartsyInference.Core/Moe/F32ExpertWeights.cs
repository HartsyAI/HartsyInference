namespace HartsyInference.Core.Moe;

/// <summary>Host F32 weights of one expert: gate and up are <c>[I, H]</c>, down is <c>[H, I]</c>, all row-major.</summary>
public sealed record F32ExpertWeights(int Hidden, int Intermediate, float[] Gate, float[] Up, float[] Down)
{
    /// <summary>Checks the array lengths against the shape.</summary>
    /// <exception cref="ArgumentException">A matrix does not hold the shape's elements.</exception>
    public F32ExpertWeights Validated()
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(Hidden);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(Intermediate);
        ArgumentNullException.ThrowIfNull(Gate);
        ArgumentNullException.ThrowIfNull(Up);
        ArgumentNullException.ThrowIfNull(Down);
        // Gate and up are [I, H] and down is [H, I]: each holds the same element count.
        long matrixElements = (long)Intermediate * Hidden;
        if (Gate.Length != matrixElements) throw new ArgumentException($"Gate must hold {matrixElements} values.", nameof(Gate));
        if (Up.Length != matrixElements) throw new ArgumentException($"Up must hold {matrixElements} values.", nameof(Up));
        if (Down.Length != matrixElements) throw new ArgumentException($"Down must hold {matrixElements} values.", nameof(Down));
        return this;
    }
}
