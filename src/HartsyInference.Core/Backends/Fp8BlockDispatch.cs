using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends;

/// <summary>Routes a linear on an FP8 E4M3 weight that carries its F32 block-scale grid (<see cref="QuantWeightInfo.Fp8BlockScaleInv"/>) to
/// <see cref="IBackend.Fp8BlockLinear"/>. Each backend's <see cref="IBackend.Linear"/> calls this first, so every model that loads such a
/// checkpoint shares one branch, whichever projection calls it.</summary>
public static class Fp8BlockDispatch
{
    /// <summary>Runs the linear as an FP8 block-scaled GEMM and returns true, or returns false when <paramref name="weight"/> carries no block-scale grid.</summary>
    public static bool TryLinear(IBackend backend, Tensor output, Tensor input, Tensor weight, Tensor? bias)
    {
        ArgumentNullException.ThrowIfNull(backend);
        QuantWeightInfo? quant = weight.QuantInfo;
        if (weight.DType != DType.F8E4M3 || quant?.Fp8BlockScaleInv is not { } scale) return false;
        // The block GEMM does not apply a low-rank adjunct, so refuse rather than drop it silently.
        if (weight.LowRankAdjunct is { IsActive: true })
            throw new NotSupportedException("A block-scaled FP8 weight cannot also carry an active low-rank adjunct; the FP8 linear does not apply it.");
        backend.Fp8BlockLinear(output, input, weight, scale, bias, quant.Fp8BlockSize, quant.Fp8BlockSize);
        return true;
    }
}
