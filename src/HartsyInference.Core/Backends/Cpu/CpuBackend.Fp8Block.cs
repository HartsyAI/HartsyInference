using HartsyInference.Core.Backends.Host;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends.Cpu;

public sealed partial class CpuBackend
{
    /// <inheritdoc />
    public void Fp8BlockLinear(Tensor output, Tensor input, Tensor weight, Tensor weightScaleInv, Tensor? bias,
        int blockRows = 128, int blockCols = 128)
    {
        ThrowIfDisposed();
        Fp8BlockLinearReference.Apply(output, input, weight, weightScaleInv, bias, blockRows, blockCols);
    }
}
