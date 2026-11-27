using HartsyInference.Core.Backends.Contracts;
using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends.Host;

/// <summary>The executable form of the elementwise math contract (<c>docs/BACKEND_OPS.md</c>): each value is read as F32, mapped
/// in F32 and rounded once to the output dtype, BF16 to nearest even as the device cast does. The <see cref="IBackend"/>
/// default and the oracle the kernels are checked against.</summary>
public static unsafe class ElementwiseMathReference
{
    /// <summary><c>output = e^input</c>.</summary>
    public static void Exp(Tensor output, Tensor input) => Apply(output, input, ElementwiseMathFunction.Exp);

    /// <summary><c>output = 1 / input</c>.</summary>
    public static void Reciprocal(Tensor output, Tensor input) => Apply(output, input, ElementwiseMathFunction.Reciprocal);

    /// <summary>The function applied to one F32 value.</summary>
    public static float Scalar(ElementwiseMathFunction function, float x) => function switch
    {
        ElementwiseMathFunction.Exp => MathF.Exp(x),
        ElementwiseMathFunction.Reciprocal => 1f / x,
        _ => throw new ArgumentOutOfRangeException(nameof(function), function, "Unknown elementwise math function."),
    };

    private static void Apply(Tensor output, Tensor input, ElementwiseMathFunction function)
    {
        long count = ElementwiseMathContract.Validate(output, input, function);
        bool inBf16 = input.DType == DType.BF16;
        bool outBf16 = output.DType == DType.BF16;
        void* source = input.DataPointer;
        void* destination = output.DataPointer;
        for (long i = 0; i < count; i++)
        {
            float x = inBf16 ? BFloat16Bits.ToFloat(((ushort*)source)[i]) : ((float*)source)[i];
            float y = Scalar(function, x);
            if (outBf16)
                ((ushort*)destination)[i] = BFloat16Bits.FromFloat(y);
            else
                ((float*)destination)[i] = y;
        }
    }
}
