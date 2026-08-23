using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.F5Tts;

/// <summary>Low-level helpers used across F5-TTS modules, operating on F32 tensors via unsafe pointer math; new ops here relative to <c>WhisperOps</c> are Mish, SiLU, Global Response Normalization (GRN, from ConvNeXt-V2), AdaLN-Zero modulation, and a grouped Conv1D path for <see cref="F5ConvPosEmbed"/>.</summary>
internal static unsafe class F5Ops
{
    /// <summary>Mish activation, in place: <c>x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))</c>, used by F5-TTS's ConvPositionEmbedding between the two grouped Conv1D layers.</summary>
    public static void MishInPlace(Tensor t)
    {
        float* p = (float*)t.DataPointer;
        long n = t.ElementCount;
        for (long i = 0; i < n; i++)
        {
            float x = p[i];
            // Numerically stable softplus: log(1 + exp(x)) = max(x, 0) + log(1 + exp(-|x|))
            float ax = MathF.Abs(x);
            float sp = MathF.Max(x, 0f) + MathF.Log(1f + MathF.Exp(-ax));
            p[i] = x * MathF.Tanh(sp);
        }
    }

    /// <summary>Tanh-approximation GELU in place (<c>nn.GELU(approximate="tanh")</c>) — the variant F5-TTS's FeedForward uses (the ConvNeXt text stem uses the exact erf GELU instead).</summary>
    public static void TanhGeluInPlace(Tensor t)
    {
        const float c = 0.7978845608028654f;   // sqrt(2/pi)
        float* p = (float*)t.DataPointer;
        long n = t.ElementCount;
        for (long i = 0; i < n; i++)
        {
            float x = p[i];
            p[i] = 0.5f * x * (1f + MathF.Tanh(c * (x + 0.044715f * x * x * x)));
        }
    }

    /// <summary>SiLU activation in place: <c>x * sigmoid(x)</c>.</summary>
    public static void SiluInPlace(Tensor t)
    {
        float* p = (float*)t.DataPointer;
        long n = t.ElementCount;
        for (long i = 0; i < n; i++)
        {
            float x = p[i];
            p[i] = x / (1f + MathF.Exp(-x));
        }
    }

    /// <summary>Global Response Normalization (ConvNeXt-V2 Eq. 4) on the channel-last layout <c>[B, T, D]</c>: <c>out = gamma * (x * Gx/(mean(Gx)+eps)) + beta + x</c> where <c>Gx = ||x||_2</c> along the time axis.</summary>
    public static void Grn(Tensor x, Tensor gamma, Tensor beta, int batch, int t, int dim)
    {
        float* xp = (float*)x.DataPointer;
        float* gp = (float*)gamma.DataPointer;
        float* bp = (float*)beta.DataPointer;
        const float eps = 1e-6f;

        for (int b = 0; b < batch; b++)
        {
            int bOff = b * t * dim;

            // Gx[d] = sqrt(sum_t x[b, t, d]^2). Reads column-by-column through cache-hostile strides —
            // at our sizes (T ~1000, D ~512) this is still small enough to be fast in C# scalar.
            float[] gx = new float[dim];
            for (int d = 0; d < dim; d++)
            {
                float sq = 0f;
                for (int tt = 0; tt < t; tt++)
                {
                    float v = xp[bOff + tt * dim + d];
                    sq += v * v;
                }
                gx[d] = MathF.Sqrt(sq);
            }
            float mean = 0f;
            for (int d = 0; d < dim; d++) mean += gx[d];
            mean /= dim;
            float invMean = 1f / (mean + eps);

            // out[t, d] = gamma[d] * (x[t, d] * gx[d] * invMean) + beta[d] + x[t, d]
            for (int tt = 0; tt < t; tt++)
            {
                int row = bOff + tt * dim;
                for (int d = 0; d < dim; d++)
                {
                    float xv = xp[row + d];
                    float nx = gx[d] * invMean;
                    xp[row + d] = gp[d] * (xv * nx) + bp[d] + xv;
                }
            }
        }
    }

    /// <summary>AdaLayerNorm-Zero modulation: given a pre-normalized (affine-free LayerNorm) hidden tensor <c>x [B, T, D]</c> and per-batch <c>(shift, scale) [B, D]</c>, computes <c>x * (1 + scale) + shift</c> in place; used by every DiT block before attention and before FF.</summary>
    public static void AdaLnZeroModulate(Tensor x, float* shift, float* scale, int batch, int t, int dim)
    {
        float* xp = (float*)x.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            int sBase = b * dim;
            for (int tt = 0; tt < t; tt++)
            {
                int xBase = (b * t + tt) * dim;
                for (int d = 0; d < dim; d++)
                {
                    xp[xBase + d] = xp[xBase + d] * (1f + scale[sBase + d]) + shift[sBase + d];
                }
            }
        }
    }

    /// <summary>AdaLN-Zero gated residual add: <c>out = x + gate[B, 1, D] * h</c>, where <paramref name="gate"/> is a per-batch [D] vector scaling the sub-block output before adding back to the residual.</summary>
    public static void AdaLnZeroGatedAdd(Tensor output, Tensor x, Tensor h, float* gate, int batch, int t, int dim)
    {
        float* op = (float*)output.DataPointer;
        float* xp = (float*)x.DataPointer;
        float* hp = (float*)h.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            int gBase = b * dim;
            for (int tt = 0; tt < t; tt++)
            {
                int xBase = (b * t + tt) * dim;
                for (int d = 0; d < dim; d++)
                {
                    op[xBase + d] = xp[xBase + d] + gate[gBase + d] * hp[xBase + d];
                }
            }
        }
    }

    // RoPE for F5-TTS uses the same interleaved (GPT-J / x_transformers) convention as
    // Moonshine — pairs of consecutive dims (2i, 2i+1) share a frequency. We reuse
    // <see cref="Moonshine.RotaryEmbedding"/> directly rather than duplicating the math.
}
