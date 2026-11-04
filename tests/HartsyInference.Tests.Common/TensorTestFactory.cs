using HartsyInference.Core.Tensors;

namespace HartsyInference.Tests.Common;

/// <summary>Builds small host-side F32 tensors for unit tests.</summary>
public static class TensorTestFactory
{
    /// <summary>Returns a 1-D F32 tensor of length <paramref name="n"/> with every element set to 1.</summary>
    public static unsafe Tensor Ones(int n)
    {
        Tensor t = new(new TensorShape(n), DType.F32);
        float* p = (float*)t.DataPointer;
        for (int i = 0; i < n; i++) p[i] = 1f;
        return t;
    }
}
