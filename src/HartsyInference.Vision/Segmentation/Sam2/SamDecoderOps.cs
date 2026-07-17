using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Segmentation.Sam2;

/// <summary>Small allocating wrappers shared by the mask-decoder transformer: element-wise add and affine
/// LayerNorm that return a fresh tensor (the two-way transformer threads many <c>x = norm(x + sub)</c> steps
/// and reads cleaner without hand-managed scratch at every line).</summary>
internal static class SamDecoderOps
{
    /// <summary>Returns a new tensor holding <c>a + b</c> (same shape).</summary>
    public static Tensor AddNew(IBackend backend, Tensor a, Tensor b)
    {
        Tensor outT = new Tensor(a.Shape, DType.F32);
        backend.Add(outT, a, b);
        return outT;
    }

    /// <summary>Returns a new tensor holding <c>LayerNorm(x)</c> over the last dim.</summary>
    public static Tensor LayerNormNew(IBackend backend, Tensor x, Tensor weight, Tensor bias, float eps)
    {
        Tensor outT = new Tensor(x.Shape, DType.F32);
        backend.LayerNorm(outT, x, weight, bias, eps);
        return outT;
    }
}
