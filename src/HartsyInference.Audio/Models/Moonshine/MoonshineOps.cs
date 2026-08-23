using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Moonshine;

/// <summary>Moonshine-specific low-level helpers; the headline op is <see cref="LayerNormNoBias"/>.</summary>
// Moonshine uses nn.LayerNorm(hidden, bias=False) for every norm, but IBackend's LayerNorm requires a
// bias tensor — rather than allocate a per-instance zero-bias tensor, we inline the computation here.
internal static unsafe class MoonshineOps
{
    /// <summary>Normalizes the last dim of a <c>[..., d]</c> tensor to zero mean / unit variance, then scales by <paramref name="weight"/> — no bias add.</summary>
    public static void LayerNormNoBias(Tensor output, Tensor input, Tensor weight, float eps)
    {
        int d = (int)input.Shape[input.Shape.Rank - 1];
        long total = input.ElementCount;
        long n = total / d;
        float* xp = (float*)input.DataPointer;
        float* yp = (float*)output.DataPointer;
        float* wp = (float*)weight.DataPointer;

        for (long row = 0; row < n; row++)
        {
            float* x = xp + row * d;
            float* y = yp + row * d;

            // mean over the last dim
            float mean = 0f;
            for (int i = 0; i < d; i++) mean += x[i];
            mean /= d;

            // variance over the last dim
            float sq = 0f;
            for (int i = 0; i < d; i++) { float v = x[i] - mean; sq += v * v; }
            float invStd = 1f / MathF.Sqrt(sq / d + eps);

            // affine (scale only — no bias)
            for (int i = 0; i < d; i++) y[i] = (x[i] - mean) * invStd * wp[i];
        }
    }

    /// <summary>Element-wise tanh, in place.</summary>
    public static void TanhInPlace(Tensor t)
    {
        float* p = (float*)t.DataPointer;
        long n = t.ElementCount;
        for (long i = 0; i < n; i++) p[i] = MathF.Tanh(p[i]);
    }
}
