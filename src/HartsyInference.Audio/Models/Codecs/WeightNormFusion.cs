using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Codecs;

/// <summary>Fuses PyTorch's <c>weight_norm</c> reparametrization (<c>weight_g</c> +
/// <c>weight_v</c>) into a single weight tensor at load time. EnCodec, Vocos, HiFiGAN,
/// BigVGAN, and most SEANet-style codecs ship with weight-norm applied throughout — we
/// pay the fusion cost once during model construction and store the fused tensor.
///
/// <para>Math (per-output-channel): <c>weight[oc, ...] = weight_g[oc] * weight_v[oc, ...] / ||weight_v[oc, ...]||_2</c>
/// where the L2 norm is taken over every axis except axis 0 (out_channels). Works for
/// any rank: Conv1d <c>[oc, ic, k]</c>, Conv2d <c>[oc, ic, kh, kw]</c>, Linear <c>[oc, ic]</c>.</para>
///
/// <para>The returned tensor is freshly allocated and owned by the caller. Original
/// <paramref name="weightG"/> / <paramref name="weightV"/> are not modified or
/// disposed.</para></summary>
public static unsafe class WeightNormFusion
{
    public static Tensor Fuse(Tensor weightG, Tensor weightV)
    {
        // fp16 checkpoints (e.g. GPT-SoVITS s2) store weight_norm params in F16 — cast up before fusing.
        Tensor gF = WhisperOps.EnsureF32(weightG);
        Tensor vF = WhisperOps.EnsureF32(weightV);
        try
        {
            return FuseF32(gF, vF);
        }
        finally
        {
            if (!ReferenceEquals(gF, weightG)) gF.Dispose();
            if (!ReferenceEquals(vF, weightV)) vF.Dispose();
        }
    }

    private static Tensor FuseF32(Tensor weightG, Tensor weightV)
    {
        if (weightV.Shape.Rank < 1) throw new ArgumentException($"weightV must be at least rank-1, got {weightV.Shape}.");

        int outCh = (int)weightV.Shape[0];
        long innerCount = weightV.ElementCount / outCh;
        if ((long)outCh * innerCount != weightV.ElementCount)
            throw new InvalidOperationException($"weightV element count ({weightV.ElementCount}) is not divisible by outCh ({outCh}).");

        // weight_g shape can be [out_ch] or [out_ch, 1, 1, ...] (singleton tail dims).
        // Either way it has out_ch active elements.
        if (weightG.ElementCount != outCh)
            throw new ArgumentException($"weightG must have {outCh} elements (matching weightV dim 0), got {weightG.ElementCount}.");

        Tensor fused = new(weightV.Shape, DType.F32);
        float* vp = (float*)weightV.DataPointer;
        float* gp = (float*)weightG.DataPointer;
        float* fp = (float*)fused.DataPointer;

        for (int oc = 0; oc < outCh; oc++)
        {
            // L2 norm of weightV[oc, ...].
            double sumSq = 0d;
            long rowOff = oc * innerCount;
            for (long i = 0; i < innerCount; i++)
            {
                float v = vp[rowOff + i];
                sumSq += (double)v * v;
            }
            float norm = MathF.Sqrt((float)sumSq);
            float invNorm = norm > 0f ? 1f / norm : 0f;
            float scale = gp[oc] * invNorm;
            for (long i = 0; i < innerCount; i++)
            {
                fp[rowOff + i] = vp[rowOff + i] * scale;
            }
        }
        return fused;
    }
}
