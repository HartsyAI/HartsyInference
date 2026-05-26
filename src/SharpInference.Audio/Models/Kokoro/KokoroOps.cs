using SharpInference.Audio.Models.Whisper;
using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Audio.Models.Kokoro;

/// <summary>Shared low-level helpers used by Kokoro's predictor and decoder. AdaIN1d
/// and AdaLayerNorm are both "style-conditioned affine over a normalized feature map"
/// — they only differ in WHICH axis is normalized (channel for AdaIN1d, last-dim for
/// AdaLayerNorm) and how the affine is applied.
///
/// <para>Both blocks load a single <c>fc</c> Linear from style_dim → 2*features. The
/// fc output is split into <c>(gamma, beta)</c> halves and passed to either
/// <see cref="IBackend.AdaInstanceNorm1d"/> (channel-axis InstanceNorm + affine) or
/// the LayerNorm-then-affine path inlined here.</para></summary>
internal static unsafe class KokoroOps
{
    /// <summary>Computes <c>fc(style)</c> and splits the result into gamma + beta halves.
    /// <paramref name="fcW"/> is <c>[2*features, style_dim]</c> and <paramref name="fcB"/>
    /// is <c>[2*features]</c> in PyTorch convention (Linear weight stored as
    /// <c>[out, in]</c>). Returns two fresh <c>[batch, features]</c> tensors that the
    /// caller must dispose.</summary>
    public static (Tensor Gamma, Tensor Beta) StyleToGammaBeta(IBackend backend, Tensor fcW, Tensor fcB, Tensor style, int features)
    {
        // style: [B, style_dim] → fc → [B, 2*features]
        int batch = (int)style.Shape[0];
        int styleDim = (int)style.Shape[1];

        // We treat style as [B, 1, style_dim] for ProjectLinear to give us [B, 1, 2*features].
        // Shape it back to [B, 2*features] for the split.
        TensorShape proj3Shape = new(batch, 1, 2 * features);
        Tensor proj3 = WhisperOps.ProjectLinear(backend, ReshapeAsRank3(style, batch, 1, styleDim), fcW, fcB, batch, 1, styleDim, 2 * features);

        Tensor gamma = new(new TensorShape(batch, features), DType.F32);
        Tensor beta = new(new TensorShape(batch, features), DType.F32);
        float* pp = (float*)proj3.DataPointer;
        float* gp = (float*)gamma.DataPointer;
        float* bp = (float*)beta.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            int row = b * 2 * features;
            for (int j = 0; j < features; j++)
            {
                gp[b * features + j] = pp[row + j];
                bp[b * features + j] = pp[row + features + j];
            }
        }
        proj3.Dispose();
        return (gamma, beta);
    }

    /// <summary>AdaLayerNorm: LayerNorm over the last (channel) dim of a <c>[B, T, C]</c>
    /// channels-last tensor (no affine), then apply the style-conditioned affine
    /// <c>(1 + gamma) * x_hat + beta</c> per-channel. <paramref name="gamma"/> and
    /// <paramref name="beta"/> are both <c>[B, C]</c> (broadcast over T).
    ///
    /// <para>This is the operator used inside Kokoro's DurationEncoder. <see cref="IBackend.LayerNorm"/>
    /// is called with all-ones weight and all-zeros bias to get the pure normalized
    /// tensor; then a small loop applies the per-channel affine.</para></summary>
    public static Tensor ApplyAdaLayerNorm(IBackend backend, Tensor x, Tensor gamma, Tensor beta, int t, int c, float eps = 1e-5f)
    {
        int batch = (int)x.Shape[0];
        if (x.Shape.Rank != 3 || (int)x.Shape[1] != t || (int)x.Shape[2] != c)
            throw new ArgumentException($"ApplyAdaLayerNorm expects [{batch}, {t}, {c}], got {x.Shape}.");

        // Step 1: unit-affine LayerNorm so we get the pure (x - mean) / std.
        Tensor onesW = new(new TensorShape(c), DType.F32);
        Tensor zerosB = new(new TensorShape(c), DType.F32);
        float* op = (float*)onesW.DataPointer;
        float* zp = (float*)zerosB.DataPointer;
        for (int i = 0; i < c; i++) { op[i] = 1f; zp[i] = 0f; }

        Tensor normed = new(x.Shape, DType.F32);
        backend.LayerNorm(normed, x, onesW, zerosB, eps);
        onesW.Dispose();
        zerosB.Dispose();

        // Step 2: per-channel (1+gamma)*x_hat + beta.
        Tensor output = new(x.Shape, DType.F32);
        float* xp = (float*)normed.DataPointer;
        float* op2 = (float*)output.DataPointer;
        float* gp = (float*)gamma.DataPointer;
        float* bp = (float*)beta.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            int gBase = b * c;
            for (int tt = 0; tt < t; tt++)
            {
                int rowBase = (b * t + tt) * c;
                for (int j = 0; j < c; j++)
                    op2[rowBase + j] = (1f + gp[gBase + j]) * xp[rowBase + j] + bp[gBase + j];
            }
        }
        normed.Dispose();
        return output;
    }

    /// <summary>Linear interpolation upsample by integer factor for a channels-first
    /// <c>[B, C, T]</c> tensor with <c>mode='nearest'</c>. Each input frame is repeated
    /// <paramref name="factor"/> times — output shape <c>[B, C, T * factor]</c>. This
    /// matches PyTorch's <c>F.interpolate(scale_factor=factor, mode='nearest')</c>
    /// at integer factors (the shortcut path of AdainResBlk1d).</summary>
    public static Tensor NearestUpsample1d(Tensor x, int factor)
    {
        if (factor <= 1) throw new ArgumentException($"NearestUpsample1d factor must be >1, got {factor}.");
        int batch = (int)x.Shape[0];
        int c = (int)x.Shape[1];
        int t = (int)x.Shape[2];
        int tOut = t * factor;
        Tensor output = new(new TensorShape(batch, c, tOut), DType.F32);
        float* ip = (float*)x.DataPointer;
        float* op = (float*)output.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            for (int cc = 0; cc < c; cc++)
            {
                long inBase = ((long)b * c + cc) * t;
                long outBase = ((long)b * c + cc) * tOut;
                for (int j = 0; j < t; j++)
                {
                    float val = ip[inBase + j];
                    for (int r = 0; r < factor; r++) op[outBase + j * factor + r] = val;
                }
            }
        }
        return output;
    }

    /// <summary>Depthwise (per-channel) transposed Conv1D. <paramref name="weight"/> is
    /// <c>[C, 1, K]</c> — one filter per channel. Used by the AdainResBlk1d "pool"
    /// layer in Kokoro's predictor when <c>upsample=True</c>: stride-2 depthwise transposed
    /// conv with kernel 3, padLeft=padRight=1, outputPadding=1 gives an output length of
    /// <c>2 * T</c>. The standard backend <see cref="IBackend.ConvTranspose1d"/> doesn't
    /// expose <c>groups</c>, so we implement the depthwise case inline.</summary>
    public static Tensor DepthwiseConvTranspose1d(Tensor input, Tensor weight, Tensor? bias, int stride, int padLeft, int padRight, int outputPadding)
    {
        if (input.Shape.Rank != 3) throw new ArgumentException($"input must be [B, C, T], got {input.Shape}.");
        int batch = (int)input.Shape[0];
        int channels = (int)input.Shape[1];
        int tIn = (int)input.Shape[2];
        int kernel = (int)weight.Shape[2];
        if ((int)weight.Shape[0] != channels || (int)weight.Shape[1] != 1)
            throw new ArgumentException($"DepthwiseConvTranspose1d expects weight [C, 1, K], got {weight.Shape} vs C={channels}.");

        int tOutRaw = (tIn - 1) * stride + (kernel - 1) + 1 + outputPadding;
        int tOut = tOutRaw - padLeft - padRight;
        Tensor output = new(new TensorShape(batch, channels, tOut), DType.F32);

        float* ip = (float*)input.DataPointer;
        float* op = (float*)output.DataPointer;
        float* wp = (float*)weight.DataPointer;
        float* bp = bias is null ? null : (float*)bias.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                float biasV = bp is null ? 0f : bp[c];
                int outBase = (b * channels + c) * tOut;
                for (int j = 0; j < tOut; j++) op[outBase + j] = biasV;

                int inBase = (b * channels + c) * tIn;
                int wBase = c * kernel;
                for (int i = 0; i < tIn; i++)
                {
                    float xv = ip[inBase + i];
                    if (xv == 0f) continue;
                    int outStart = i * stride - padLeft;
                    for (int k = 0; k < kernel; k++)
                    {
                        int j = outStart + k;
                        if ((uint)j < (uint)tOut) op[outBase + j] += xv * wp[wBase + k];
                    }
                }
            }
        }
        return output;
    }

    /// <summary>Treats the underlying f32 storage of <paramref name="src"/> as a rank-3
    /// shape view without copying. Used internally to feed a rank-2 <c>[B, D]</c> tensor
    /// into <see cref="WhisperOps.ProjectLinear"/> which requires <c>[B, T, D]</c>.</summary>
    private static Tensor ReshapeAsRank3(Tensor src, int b, int t, int d)
    {
        if (src.ElementCount != (long)b * t * d)
            throw new ArgumentException($"ReshapeAsRank3: element count mismatch ({src.ElementCount} vs {b}*{t}*{d}).");
        return src.Reshape(new TensorShape(b, t, d));
    }
}
