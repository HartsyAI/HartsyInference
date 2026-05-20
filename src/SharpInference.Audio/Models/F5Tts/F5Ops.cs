using SharpInference.Core.Tensors;

namespace SharpInference.Audio.Models.F5Tts;

/// <summary>Low-level helpers used across F5-TTS modules. All operate on F32 tensors via
/// unsafe pointer math. New ops here (relative to <c>WhisperOps</c>): Mish activation,
/// SiLU activation, Global Response Normalization (GRN, from ConvNeXt-V2), AdaLN-Zero
/// modulation, and a grouped Conv1D path for <see cref="F5ConvPosEmbed"/>.</summary>
internal static unsafe class F5Ops
{
    /// <summary>Mish activation, in place: <c>x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))</c>.
    /// Used by F5-TTS's ConvPositionEmbedding between the two grouped Conv1D layers.</summary>
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

    /// <summary>Global Response Normalization (ConvNeXt-V2 Eq. 4). Operates on the
    /// channel-last layout <c>[B, T, D]</c>:
    /// <code>
    ///   Gx = ||x||_2 along time axis (dim=1)            shape [B, 1, D]
    ///   Nx = Gx / (mean(Gx, dim=-1, keepdim=True) + eps) shape [B, 1, D]
    ///   out = gamma * (x * Nx) + beta + x
    /// </code>
    /// <paramref name="gamma"/> and <paramref name="beta"/> are <c>[1, 1, D]</c>
    /// learnable parameters (loaded directly with that shape).</summary>
    public static void Grn(Tensor x, Tensor gamma, Tensor beta, int batch, int t, int dim)
    {
        float* xp = (float*)x.DataPointer;
        float* gp = (float*)gamma.DataPointer;
        float* bp = (float*)beta.DataPointer;
        const float eps = 1e-6f;

        for (int b = 0; b < batch; b++)
        {
            int bOff = b * t * dim;

            // Gx[d] = sqrt(sum_t x[b, t, d]^2). One pass over [t, d] reading column-by-column
            // through cache-hostile strides — at our sizes (T ~1000, D ~512) this is still
            // small enough to be fast in C# scalar.
            for (int d = 0; d < dim; d++)
            {
                float sq = 0f;
                for (int tt = 0; tt < t; tt++)
                {
                    float v = xp[bOff + tt * dim + d];
                    sq += v * v;
                }
                // Stash Gx into the *first* row of x temporarily? No — we need it persistent
                // through the next pass. Use beta-sized scratch via reuse: we'll write Nx into
                // a stack-allocated scratch.
                // Re-pass to compute the mean over D, then per-(t, d) the GRN output.
                // We can do it in two passes: build Gx[d] array, then mean over d.
                _ = sq;
            }

            // Two-pass implementation: first build Gx[D], then mean, then apply.
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

    /// <summary>Grouped 1D convolution for <see cref="F5ConvPosEmbed"/>. Splits the
    /// channel dim into <paramref name="groups"/> groups; each group has its own
    /// <c>(in_per_group, kernel)</c> weight slice. Input/output layout: <c>[B, C, T]</c>.
    /// Pad with zeros on both sides.</summary>
    public static void GroupedConv1D(
        Tensor output, Tensor input, Tensor weight, Tensor bias,
        int batch, int channels, int t, int kernel, int padding, int groups)
    {
        int inPerGroup = channels / groups;
        int outPerGroup = channels / groups;  // square groups in F5 (in == out per group)

        float* op = (float*)output.DataPointer;
        float* ip = (float*)input.DataPointer;
        float* wp = (float*)weight.DataPointer;
        float* bp = (float*)bias.DataPointer;

        // Weight layout: [outChannels=C, inPerGroup, kernel]. Row 'o' is the o-th output
        // channel's weights for its corresponding input group (group = o / outPerGroup).
        for (int b = 0; b < batch; b++)
        {
            for (int oc = 0; oc < channels; oc++)
            {
                int group = oc / outPerGroup;
                int icStart = group * inPerGroup;
                float biasV = bp[oc];
                int wRow = oc * inPerGroup * kernel;

                for (int j = 0; j < t; j++)
                {
                    float acc = biasV;
                    for (int ic = 0; ic < inPerGroup; ic++)
                    {
                        int inCh = icStart + ic;
                        int inBase = (b * channels + inCh) * t;
                        int wBase = wRow + ic * kernel;
                        for (int k = 0; k < kernel; k++)
                        {
                            int src = j + k - padding;
                            if ((uint)src < (uint)t) acc += ip[inBase + src] * wp[wBase + k];
                        }
                    }
                    op[(b * channels + oc) * t + j] = acc;
                }
            }
        }
    }

    /// <summary>AdaLayerNorm-Zero modulation step. Given a hidden tensor <c>x [B, T, D]</c>
    /// pre-normalized (LayerNorm with no affine), and modulation parameters
    /// <c>(shift, scale)</c> each of shape <c>[B, D]</c>, computes
    /// <c>x * (1 + scale[B, 1, D]) + shift[B, 1, D]</c> in place. Used by every DiT
    /// block before attention and before FF.</summary>
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

    /// <summary>AdaLN-Zero gated residual add: <c>out = x + gate[B, 1, D] * h</c>.
    /// <paramref name="gate"/> is a per-batch [D] vector that scales the contribution of the
    /// sub-block output before adding back to the residual.</summary>
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
