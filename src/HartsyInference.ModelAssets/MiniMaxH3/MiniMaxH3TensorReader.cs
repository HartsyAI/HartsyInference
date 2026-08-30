using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.MiniMaxH3;

/// <summary>Conversion-time scalar access across the floating dtypes used by published H3 assets.</summary>
internal static unsafe class MiniMaxH3TensorReader
{
    /// <summary>Reads one tensor element as F64 without changing the source tensor.</summary>
    internal static double Read(Tensor tensor, long index)
    {
        if ((ulong)index >= (ulong)tensor.ElementCount) throw new ArgumentOutOfRangeException(nameof(index));
        void* pointer = tensor.DataPointer;
        if (tensor.DType == DType.F64) return ((double*)pointer)[index];
        if (tensor.DType == DType.F32) return ((float*)pointer)[index];
        if (tensor.DType == DType.F16) return (float)BitConverter.UInt16BitsToHalf(((ushort*)pointer)[index]);
        if (tensor.DType == DType.BF16) return BitConverter.Int32BitsToSingle(((ushort*)pointer)[index] << 16);
        throw new HartsyInferenceException($"H3 conversion requires a floating tensor; got {tensor.DType}.");
    }
}
