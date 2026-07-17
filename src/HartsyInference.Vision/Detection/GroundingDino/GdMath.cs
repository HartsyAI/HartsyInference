using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Detection.GroundingDino;

/// <summary>Small numeric helpers shared across the Grounding DINO modules: exact erf-GELU, the Swin window
/// relative-position index, softmax, and (inverse) sigmoid.</summary>
public static unsafe class GdMath
{
    /// <summary>Exact erf-GELU in place (BERT/DETR/Swin use the erf form, not the tanh approximation).</summary>
    public static void ErfGelu(Tensor x)
    {
        float* p = (float*)x.DataPointer;
        long n = x.ElementCount;
        const float invSqrt2 = 0.70710678118654752f;
        for (long i = 0; i < n; i++)
        {
            float v = p[i];
            p[i] = v * 0.5f * (1f + Erf(v * invSqrt2));
        }
    }

    /// <summary>ReLU in place.</summary>
    public static void Relu(Tensor x)
    {
        float* p = (float*)x.DataPointer;
        long n = x.ElementCount;
        for (long i = 0; i < n; i++) if (p[i] < 0f) p[i] = 0f;
    }

    /// <summary>erf via Abramowitz &amp; Stegun 7.1.26 (max abs error ~1.5e-7).</summary>
    public static float Erf(float x)
    {
        float sign = x < 0 ? -1f : 1f;
        float ax = MathF.Abs(x);
        float tt = 1f / (1f + 0.3275911f * ax);
        float y = 1f - (((((1.061405429f * tt - 1.453152027f) * tt) + 1.421413741f) * tt - 0.284496736f) * tt
            + 0.254829592f) * tt * MathF.Exp(-ax * ax);
        return sign * y;
    }

    public static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));

    /// <summary>Inverse sigmoid (logit) with clamping, matching <c>torch.special.logit(x, eps)</c>.</summary>
    public static float InverseSigmoid(float x, float eps = 1e-5f)
    {
        float c = x < eps ? eps : (x > 1f - eps ? 1f - eps : x);
        return MathF.Log(c / (1f - c));
    }

    /// <summary>In-place softmax over a contiguous span.</summary>
    public static void Softmax(Span<float> v)
    {
        float max = float.NegativeInfinity;
        for (int i = 0; i < v.Length; i++) if (v[i] > max) max = v[i];
        float sum = 0f;
        for (int i = 0; i < v.Length; i++) { float e = MathF.Exp(v[i] - max); v[i] = e; sum += e; }
        float inv = 1f / sum;
        for (int i = 0; i < v.Length; i++) v[i] *= inv;
    }

    /// <summary>Swin relative-position index for a <paramref name="ws"/>×<paramref name="ws"/> window, flattened to
    /// <c>[ws*ws * ws*ws]</c>. Each entry indexes into <c>relative_position_bias_table[(2ws-1)^2, heads]</c>.</summary>
    public static int[] SwinRelativePositionIndex(int ws)
    {
        int n = ws * ws;
        int[] coordsY = new int[n], coordsX = new int[n];
        for (int y = 0; y < ws; y++)
            for (int x = 0; x < ws; x++)
            {
                coordsY[y * ws + x] = y;
                coordsX[y * ws + x] = x;
            }
        int[] index = new int[n * n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
            {
                int ry = coordsY[i] - coordsY[j] + ws - 1;
                int rx = coordsX[i] - coordsX[j] + ws - 1;
                index[i * n + j] = ry * (2 * ws - 1) + rx;
            }
        return index;
    }
}
