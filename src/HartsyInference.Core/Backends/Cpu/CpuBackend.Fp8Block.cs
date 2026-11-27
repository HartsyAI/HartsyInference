using HartsyInference.Core.Backends.Cpu.Kernels;
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
        // Checks the operands as the reference does, then decodes the weight once per call in parallel and runs the vectorized
        // GEMM. The transient F32 copy is one weight, the price of the speed; the reference loop stays the oracle in tests.
        Fp8BlockLinearReference.Validate(output, input, weight, weightScaleInv, bias, blockRows, blockCols);
        using Tensor dense = Fp8BlockLinearReference.DequantizeF32(weight, weightScaleInv, blockRows, blockCols);
        MatMulKernels.LinearTransB(output, input, dense, bias);
    }

    /// <inheritdoc />
    public void BlockDiagonalLinear(Tensor output, Tensor input, Tensor weight, Tensor? weightScaleInv, int groups,
        int blockRows = 128, int blockCols = 128)
    {
        ThrowIfDisposed();
        BlockDiagonalLinearReference.Apply(output, input, weight, weightScaleInv, groups, blockRows, blockCols);
    }
}
