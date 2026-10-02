using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Kokoro;

/// <summary>Shared low-level helpers used by Kokoro's predictor and decoder. AdaIN1d
/// and AdaLayerNorm are both "style-conditioned affine over a normalized feature map"
/// — they only differ in WHICH axis is normalized (channel for AdaIN1d, last-dim for
/// AdaLayerNorm) and how the affine is applied.
///
/// <para>Both blocks load a single <c>fc</c> Linear from style_dim → 2*features. The
/// fc output is split into <c>(gamma, beta)</c> halves and passed to either
/// <see cref="IBackend.AdaInstanceNorm1d"/> (channel-axis InstanceNorm + affine) or
/// <see cref="IBackend.LayerNormModulate"/>.</para>
///
/// <para>Every tensor helper here is a backend op: the tensors flowing through the predictor and decoder stay
/// device-resident, and none of them reads a <c>DataPointer</c> (each such read is a stream drain).</para></summary>
internal static class KokoroOps
{
    /// <summary>A synthesis stage boundary: reports <paramref name="stage"/> to <paramref name="observer"/>, then, once
    /// <paramref name="cancel"/> is signalled, disposes the stage's <paramref name="live"/> tensors and throws
    /// <see cref="OperationCanceledException"/>.</summary>
    public static void StageBoundary(Action<string>? observer, string stage, CancellationToken cancel,
        params ReadOnlySpan<Tensor> live)
    {
        observer?.Invoke(stage);
        if (!cancel.IsCancellationRequested)
        {
            return;
        }
        foreach (Tensor tensor in live)
        {
            tensor.Dispose();
        }
        cancel.ThrowIfCancellationRequested();
    }

    /// <summary>Computes <c>fc(style)</c> and splits the result into gamma + beta halves.
    /// <paramref name="fcW"/> is <c>[2*features, style_dim]</c> and <paramref name="fcB"/>
    /// is <c>[2*features]</c> in PyTorch convention (Linear weight stored as
    /// <c>[out, in]</c>). <paramref name="style"/> is <c>[batch, style_dim]</c>. Returns two fresh
    /// <c>[batch, features]</c> tensors that the caller must dispose.</summary>
    public static (Tensor Gamma, Tensor Beta) StyleToGammaBeta(IBackend backend, Tensor fcW, Tensor fcB, Tensor style, int features)
    {
        int batch = (int)style.Shape[0];
        Tensor proj = new(new TensorShape(batch, 2 * features), DType.F32);
        backend.Linear(proj, style, fcW, fcB);
        Tensor gamma = new(new TensorShape(batch, features), DType.F32);
        Tensor beta = new(new TensorShape(batch, features), DType.F32);
        backend.SliceLastDim(gamma, proj, 0);
        backend.SliceLastDim(beta, proj, features);
        proj.Dispose();
        return (gamma, beta);
    }

    /// <summary>AdaLayerNorm: LayerNorm over the last (channel) dim of a <c>[B, T, C]</c>
    /// channels-last tensor (no affine), then the style-conditioned affine
    /// <c>(1 + gamma) * x_hat + beta</c> per channel — exactly <see cref="IBackend.LayerNormModulate"/>.
    /// <paramref name="gamma"/> and <paramref name="beta"/> are both <c>[B, C]</c> (broadcast over T).
    /// This is the operator inside Kokoro's DurationEncoder.</summary>
    public static Tensor ApplyAdaLayerNorm(IBackend backend, Tensor x, Tensor gamma, Tensor beta, int t, int c, float eps = 1e-5f)
    {
        int batch = (int)x.Shape[0];
        if (x.Shape.Rank != 3 || (int)x.Shape[1] != t || (int)x.Shape[2] != c)
            throw new ArgumentException($"ApplyAdaLayerNorm expects [{batch}, {t}, {c}], got {x.Shape}.");
        Tensor output = new(x.Shape, DType.F32);
        backend.LayerNormModulate(output, x, gamma, beta, eps);
        return output;
    }

    /// <summary>Nearest-neighbour upsample by an integer factor for a channels-first
    /// <c>[B, C, T]</c> tensor: each input frame is repeated <paramref name="factor"/> times —
    /// output shape <c>[B, C, T * factor]</c>. Matches PyTorch's
    /// <c>F.interpolate(scale_factor=factor, mode='nearest')</c> at integer factors
    /// (the shortcut path of AdainResBlk1d).</summary>
    public static Tensor NearestUpsample1d(IBackend backend, Tensor x, int factor)
    {
        if (factor <= 1) throw new ArgumentException($"NearestUpsample1d factor must be >1, got {factor}.");
        Tensor output = new(new TensorShape(x.Shape[0], x.Shape[1], x.Shape[2] * factor), DType.F32);
        backend.RepeatTime(output, x, factor);
        return output;
    }

    /// <summary>Depthwise (per-channel) transposed Conv1D. <paramref name="weight"/> is
    /// <c>[C, 1, K]</c> — one filter per channel. Used by the AdainResBlk1d "pool"
    /// layer in Kokoro's predictor when <c>upsample=True</c>: stride-2 depthwise transposed
    /// conv with kernel 3, padding 1, output_padding 1 gives an output length of <c>2 * T</c>.
    /// PyTorch's <c>output_padding</c> only shortens the right crop, so it maps onto the backend's
    /// asymmetric pads as <c>padRight = padding - outputPadding</c>.</summary>
    public static Tensor DepthwiseConvTranspose1d(IBackend backend, Tensor input, Tensor weight, Tensor? bias,
        int stride, int padding, int outputPadding)
    {
        if (input.Shape.Rank != 3) throw new ArgumentException($"input must be [B, C, T], got {input.Shape}.");
        int batch = (int)input.Shape[0];
        int channels = (int)input.Shape[1];
        int tIn = (int)input.Shape[2];
        int kernel = (int)weight.Shape[2];
        if ((int)weight.Shape[0] != channels || (int)weight.Shape[1] != 1)
            throw new ArgumentException($"DepthwiseConvTranspose1d expects weight [C, 1, K], got {weight.Shape} vs C={channels}.");
        if (outputPadding > padding)
            throw new ArgumentException($"outputPadding {outputPadding} exceeds padding {padding}.");

        int padRight = padding - outputPadding;
        int tOut = (tIn - 1) * stride + kernel - padding - padRight;
        Tensor output = new(new TensorShape(batch, channels, tOut), DType.F32);
        backend.ConvTranspose1d(output, input, weight, bias, stride, padLeft: padding, padRight: padRight, dilation: 1, groups: channels);
        return output;
    }

    /// <summary>Repeats a <c>[B, D]</c> style row across time into <c>[B, T, D]</c>, on device.</summary>
    public static Tensor RepeatStyleAcrossTime(IBackend backend, Tensor style, int batch, int t, int styleDim)
    {
        if (style.Shape.Rank != 2 || (int)style.Shape[0] != batch || (int)style.Shape[1] != styleDim)
            throw new ArgumentException($"RepeatStyleAcrossTime expects [{batch}, {styleDim}], got {style.Shape}.");
        // The style row is a host tensor (voice pack / style encoder output), so the view costs no device sync.
        Tensor column = style.Reshape(new TensorShape(batch, styleDim, 1));
        Tensor repeated = new(new TensorShape(batch, styleDim, t), DType.F32);
        backend.RepeatTime(repeated, column, t);
        Tensor output = new(new TensorShape(batch, t, styleDim), DType.F32);
        backend.Transpose2D(output, repeated, styleDim, t);
        repeated.Dispose();
        return output;
    }

    /// <summary>Concatenates channels-first <c>[B, C_i, T]</c> tensors along the channel dim.</summary>
    public static Tensor ConcatChannels(IBackend backend, ReadOnlySpan<Tensor> parts)
    {
        long batch = parts[0].Shape[0];
        long t = parts[0].Shape[2];
        long totalCh = 0;
        foreach (Tensor p in parts)
        {
            if (p.Shape.Rank != 3 || p.Shape[0] != batch || p.Shape[2] != t)
                throw new ArgumentException($"ConcatChannels: {p.Shape} does not match [{batch}, *, {t}].");
            totalCh += p.Shape[1];
        }
        Tensor output = new(new TensorShape(batch, totalCh, t), DType.F32);
        backend.Concat(output, parts, dim: 1);
        return output;
    }

    /// <summary>ReflectionPad1d((1, 0)) on a channels-first <c>[B, C, T]</c>: prepends one sample by reflection
    /// (<c>out[0] = x[1]</c>), as a last-dim slice plus a time-axis concat.</summary>
    public static Tensor ReflectionPadLeft1(IBackend backend, Tensor x)
    {
        if (x.Shape.Rank != 3 || x.Shape[2] < 2) throw new ArgumentException($"ReflectionPadLeft1 expects [B, C, T>=2], got {x.Shape}.");
        Tensor first = new(new TensorShape(x.Shape[0], x.Shape[1], 1), DType.F32);
        backend.SliceLastDim(first, x, 1);
        Tensor output = new(new TensorShape(x.Shape[0], x.Shape[1], x.Shape[2] + 1), DType.F32);
        backend.Concat(output, [first, x], dim: 2);
        first.Dispose();
        return output;
    }

    /// <summary>Element-wise <c>a + b</c> into a fresh tensor of <paramref name="a"/>'s shape.</summary>
    public static Tensor Add(IBackend backend, Tensor a, Tensor b)
    {
        if (a.ElementCount != b.ElementCount)
            throw new ArgumentException($"Add: element counts differ ({a.Shape} vs {b.Shape}).");
        Tensor output = new(a.Shape, DType.F32);
        backend.Add(output, a, b);
        return output;
    }
}
