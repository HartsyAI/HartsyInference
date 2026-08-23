using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Host-side helpers shared by the Wan causal-conv encoders (S2V audio, Animate face): channels-first ↔ token layout flips and the fused no-affine LayerNorm + SiLU between conv stages.</summary>
internal static unsafe class WanEncoderOps
{
    /// <summary>[C, T] → [T, C].</summary>
    public static Tensor ChannelsToTokens(Tensor x, int c, int t)
    {
        Tensor o = new Tensor(new TensorShape(t, c), DType.F32);
        float* xp = (float*)x.DataPointer, op = (float*)o.DataPointer;
        for (int ci = 0; ci < c; ci++)
            for (int ti = 0; ti < t; ti++) op[(long)ti * c + ci] = xp[(long)ci * t + ti];
        return o;
    }

    /// <summary>[T, C] → [1, C, T] for Conv1d (channels-first with a batch dim).</summary>
    public static Tensor TokensToChannels(Tensor x, int c, int t)
    {
        Tensor o = new Tensor(new TensorShape(1, c, t), DType.F32);
        float* xp = (float*)x.DataPointer, op = (float*)o.DataPointer;
        for (int ti = 0; ti < t; ti++)
            for (int ci = 0; ci < c; ci++) op[(long)ci * t + ti] = xp[(long)ti * c + ci];
        return o;
    }

    /// <summary>In-place no-affine LayerNorm over the channel dim followed by SiLU, on a <c>[T, C]</c> tensor.</summary>
    public static void LayerNormSilu(Tensor x, int t, int c, float eps)
    {
        float* xp = (float*)x.DataPointer;
        for (int i = 0; i < t; i++)
        {
            long off = (long)i * c;
            double mean = 0; for (int d = 0; d < c; d++) mean += xp[off + d]; mean /= c;
            double var = 0; for (int d = 0; d < c; d++) { double dd = xp[off + d] - mean; var += dd * dd; }
            float inv = 1f / MathF.Sqrt((float)(var / c) + eps);
            for (int d = 0; d < c; d++)
            {
                float n = (float)((xp[off + d] - mean) * inv);
                xp[off + d] = n / (1f + MathF.Exp(-n));   // SiLU
            }
        }
    }
}
