using HartsyInference.Core.Tensors;
using HartsyInference.Vision.Dinov2;

namespace HartsyInference.Vision.Annotators;

/// <summary>Folds an inference-mode BatchNorm into its preceding conv at load time, shared by the annotator
/// loaders (NormalBAE at eps 1e-3, UperNet-seg at eps 1e-5): <c>w' = w·γ/√(σ²+ε)</c>,
/// <c>b' = (b−μ)·γ/√(σ²+ε) + β</c>. Handles bias-free convs (b = 0).</summary>
internal static unsafe class ConvBnFold
{
    /// <summary>Returns the folded conv weight/bias for <c>{convKey}.weight[/bias]</c> +
    /// <c>{bnKey}.weight/bias/running_mean/running_var</c>.</summary>
    public static (Tensor W, Tensor B) Fold(IReadOnlyDictionary<string, Tensor> w, string convKey, string bnKey, float eps)
    {
        Tensor cw = Dinov2VisionEncoder.F32(w[$"{convKey}.weight"]);
        Tensor? cb = w.TryGetValue($"{convKey}.bias", out Tensor? cbRaw) ? Dinov2VisionEncoder.F32(cbRaw) : null;
        Tensor gamma = Dinov2VisionEncoder.F32(w[$"{bnKey}.weight"]);
        Tensor beta = Dinov2VisionEncoder.F32(w[$"{bnKey}.bias"]);
        Tensor mean = Dinov2VisionEncoder.F32(w[$"{bnKey}.running_mean"]);
        Tensor var = Dinov2VisionEncoder.F32(w[$"{bnKey}.running_var"]);

        int outC = (int)cw.Shape[0];
        long perOut = cw.ElementCount / outC;
        Tensor foldedW = new(cw.Shape, DType.F32);
        Tensor foldedB = new(new TensorShape(outC), DType.F32);
        float* pw = (float*)cw.DataPointer;
        float* pfw = (float*)foldedW.DataPointer;
        float* pg = (float*)gamma.DataPointer;
        float* pb = (float*)beta.DataPointer;
        float* pm = (float*)mean.DataPointer;
        float* pv = (float*)var.DataPointer;
        float* pcb = cb is null ? null : (float*)cb.DataPointer;
        float* pfb = (float*)foldedB.DataPointer;
        for (int o = 0; o < outC; o++)
        {
            float scale = pg[o] / MathF.Sqrt(pv[o] + eps);
            for (long i = 0; i < perOut; i++)
                pfw[o * perOut + i] = pw[o * perOut + i] * scale;
            pfb[o] = ((pcb is null ? 0f : pcb[o]) - pm[o]) * scale + pb[o];
        }
        return (foldedW, foldedB);
    }
}
