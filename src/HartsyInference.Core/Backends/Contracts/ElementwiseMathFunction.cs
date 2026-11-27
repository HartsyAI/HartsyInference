namespace HartsyInference.Core.Backends.Contracts;

/// <summary>The function an elementwise math op applies; the value is the <c>op</c> code the CUDA kernel reads.</summary>
public enum ElementwiseMathFunction
{
    /// <summary><c>e^x</c>.</summary>
    Exp = 0,

    /// <summary><c>1 / x</c>, IEEE division.</summary>
    Reciprocal = 1,
}
