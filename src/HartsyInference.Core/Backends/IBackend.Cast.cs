using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends;

// Dtype casts between F32, F16 and BF16.
public partial interface IBackend
{
    /// <summary>Cast tensor from FP32 to FP16. Default: CPU scalar loop.</summary>
    unsafe void CastToF16(Tensor output, Tensor input)
    {
        float* src = (float*)input.DataPointer;
        Half* dst = (Half*)output.DataPointer;
        int count = (int)input.Shape.ElementCount;
        for (int i = 0; i < count; i++)
            dst[i] = (Half)src[i];
    }

    /// <summary>Cast tensor from FP16 or BF16 to FP32. Default: CPU scalar loop.</summary>
    unsafe void CastToF32(Tensor output, Tensor input)
    {
        if (input.DType == DType.BF16)
        {
            ushort* bits = (ushort*)input.DataPointer;
            float* values = (float*)output.DataPointer;
            int n = (int)input.Shape.ElementCount;
            for (int i = 0; i < n; i++)
                values[i] = BFloat16Bits.ToFloat(bits[i]);
            return;
        }
        Half* src = (Half*)input.DataPointer;
        float* dst = (float*)output.DataPointer;
        int count = (int)input.Shape.ElementCount;
        for (int i = 0; i < count; i++)
            dst[i] = (float)src[i];
    }

    /// <summary>Cast tensor from FP32 to BF16. Default: CPU fallback via Tensor.CastTo.</summary>
    void CastToBf16(Tensor output, Tensor input) => CastToBf16Reference(output, input);

    /// <summary>The host implementation of <see cref="CastToBf16"/>, for a backend that overrides it and needs to
    /// bail on a dtype its kernel does not cover.</summary>
    /// <remarks>Static because an override cannot reach its own interface default — the call binds back to the
    /// class and recurses until the stack ends.</remarks>
    static unsafe void CastToBf16Reference(Tensor output, Tensor input)
    {
        // F32 input rounds to nearest even, as the device cast does; Tensor.CastTo truncates.
        if (input.DType == DType.F32 && output.DType == DType.BF16)
        {
            float* src = (float*)input.DataPointer;
            ushort* dst = (ushort*)output.DataPointer;
            int n = (int)input.Shape.ElementCount;
            for (int i = 0; i < n; i++)
                dst[i] = BFloat16Bits.FromFloat(src[i]);
            return;
        }
        Tensor casted = input.CastTo(DType.BF16);
        try
        {
            unsafe
            {
                Buffer.MemoryCopy((void*)casted.DataPointer, (void*)output.DataPointer,
                    output.ElementCount * 2, casted.ElementCount * 2);
            }
        }
        finally { casted.Dispose(); }
    }
}
