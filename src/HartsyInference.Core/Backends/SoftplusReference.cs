using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends;

/// <summary>CPU reference for softplus, matching <c>torch.nn.functional.softplus</c> (beta 1, threshold 20).</summary>
public static class SoftplusReference
{
    private const float Threshold = 20f;

    /// <summary>Softplus of one value; the log1p is evaluated in double so tiny results keep their precision.</summary>
    public static float Scalar(float x)
    {
        if (x > Threshold) return x;
        double e = Math.Exp(x);
        double u = 1.0 + e;
        return (float)(u == 1.0 ? e : Math.Log(u) * e / (u - 1.0));
    }

    /// <summary>Elementwise softplus over F32 tensors of equal length; in-place is allowed.</summary>
    public static unsafe void Apply(Tensor output, Tensor input)
    {
        if (output.DType != DType.F32 || input.DType != DType.F32)
            throw new NotSupportedException("Softplus supports F32 only.");
        if (output.ElementCount != input.ElementCount)
            throw new ArgumentException($"Softplus output has {output.ElementCount} elements, input {input.ElementCount}.");
        float* o = (float*)output.DataPointer;
        float* i = (float*)input.DataPointer;
        for (long n = 0; n < input.ElementCount; n++) o[n] = Scalar(i[n]);
    }
}
