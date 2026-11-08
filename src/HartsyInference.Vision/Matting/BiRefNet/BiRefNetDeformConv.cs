using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Vision.Annotators;

namespace HartsyInference.Vision.Matting.BiRefNet;

/// <summary>Modulated deformable convolution (DCNv2, stride 1, dilation 1) as the upstream <c>DeformableConv2d</c>
/// runs it: an offset conv and a modulator conv predict the sampling shift and mask of every kernel tap, the input is
/// bilinearly sampled at the shifted positions (zero outside the image, per corner), scaled by the mask, and the
/// samples are contracted with the regular kernel as a GEMM over chunks of output positions. The regular conv's
/// BatchNorm is folded at load, so the output is the post-BN pre-activation map.</summary>
internal sealed unsafe class BiRefNetDeformConv
{
    private const float BatchNormEps = 1e-5f;
    private const int MaxChunkElements = 4 << 20;

    private readonly int _kernel, _pad;
    private Tensor? _offsetW, _offsetB, _modulatorW, _modulatorB, _regularW, _regularB;

    public BiRefNetDeformConv(int kernel, int pad)
    {
        _kernel = kernel;
        _pad = pad;
    }

    /// <summary>Loads <c>{prefix}.atrous_conv.{offset_conv,modulator_conv,regular_conv}</c> and folds <c>{prefix}.bn</c> into the regular conv.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix, List<Tensor> owned)
    {
        _offsetW = BiRefNetOps.Own(w[$"{prefix}.atrous_conv.offset_conv.weight"], owned);
        _offsetB = BiRefNetOps.Own(w[$"{prefix}.atrous_conv.offset_conv.bias"], owned);
        _modulatorW = BiRefNetOps.Own(w[$"{prefix}.atrous_conv.modulator_conv.weight"], owned);
        _modulatorB = BiRefNetOps.Own(w[$"{prefix}.atrous_conv.modulator_conv.bias"], owned);
        (Tensor foldedW, Tensor foldedB) = ConvBnFold.Fold(w, $"{prefix}.atrous_conv.regular_conv", $"{prefix}.bn", BatchNormEps);
        owned.Add(foldedW); owned.Add(foldedB);
        _regularW = foldedW;
        _regularB = foldedB;
    }

    /// <summary>Runs the layer on <paramref name="x"/> <c>[1,C,H,W]</c>; returns <c>[1,OC,H,W]</c>.</summary>
    public Tensor Forward(IBackend backend, Tensor x)
    {
        int channels = (int)x.Shape[1], h = (int)x.Shape[2], w = (int)x.Shape[3];
        int taps = _kernel * _kernel, cols = channels * taps, outChannels = (int)_regularW!.Shape[0];
        int positions = h * w;

        Tensor offset = BiRefNetOps.Conv(backend, x, _offsetW!, _offsetB, _pad);
        Tensor modRaw = BiRefNetOps.Conv(backend, x, _modulatorW!, _modulatorB, _pad);
        Tensor modulator = BiRefNetOps.Sigmoid(backend, modRaw);
        modRaw.Dispose();
        Scale(modulator, 2f);

        Tensor weight2d = _regularW!.Reshape(new TensorShape(outChannels, cols));
        int chunk = Math.Max(1, Math.Min(positions, MaxChunkElements / cols));
        Tensor samples = new(new TensorShape(chunk, cols), DType.F32);
        Tensor chunkOut = new(new TensorShape(chunk, outChannels), DType.F32);
        Tensor output = new(new TensorShape(1, outChannels, h, w), DType.F32);

        float* xp = (float*)x.DataPointer;
        float* offp = (float*)offset.DataPointer;
        float* modp = (float*)modulator.DataPointer;
        float* op = (float*)output.DataPointer;
        for (int start = 0; start < positions; start += chunk)
        {
            int n = Math.Min(chunk, positions - start);
            // Re-acquire the GEMM buffers each chunk: on CUDA, Linear leaves its result device-resident, and the host copy
            // is synchronized only when DataPointer is read after the call.
            float* sp = (float*)samples.DataPointer;
            for (int pi = 0; pi < chunk; pi++)
            {
                float* row = sp + (long)pi * cols;
                if (pi >= n)
                {
                    new Span<float>(row, cols).Clear();
                    continue;
                }
                int p = start + pi, py = p / w, px = p % w;
                for (int tap = 0; tap < taps; tap++)
                {
                    int ky = tap / _kernel, kx = tap % _kernel;
                    float sy = py - _pad + ky + offp[(2L * tap) * positions + p];
                    float sx = px - _pad + kx + offp[(2L * tap + 1) * positions + p];
                    float mask = modp[(long)tap * positions + p];
                    for (int c = 0; c < channels; c++)
                        row[c * taps + tap] = Bilinear(xp + (long)c * positions, h, w, sy, sx) * mask;
                }
            }
            backend.Linear(chunkOut, samples, weight2d, _regularB);
            float* cop = (float*)chunkOut.DataPointer;
            for (int pi = 0; pi < n; pi++)
                for (int oc = 0; oc < outChannels; oc++)
                    op[(long)oc * positions + start + pi] = cop[(long)pi * outChannels + oc];
        }

        weight2d.Dispose();
        samples.Dispose(); chunkOut.Dispose(); offset.Dispose(); modulator.Dispose();
        return output;
    }

    private static void Scale(Tensor t, float factor)
    {
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++) p[i] *= factor;
    }

    /// <summary>Bilinear sample of one HxW plane at (y, x), matching torchvision's <c>deform_conv2d</c>: outside
    /// <c>(-1, size)</c> on either axis the sample is zero, and each of the four corners contributes only when it lies
    /// inside the image.</summary>
    private static float Bilinear(float* plane, int h, int w, float y, float x)
    {
        if (y <= -1f || y >= h || x <= -1f || x >= w) return 0f;
        int y0 = (int)MathF.Floor(y), x0 = (int)MathF.Floor(x);
        int y1 = y0 + 1, x1 = x0 + 1;
        float ly = y - y0, lx = x - x0, hy = 1f - ly, hx = 1f - lx;
        float v = 0f;
        if (y0 >= 0 && x0 >= 0) v += hy * hx * plane[(long)y0 * w + x0];
        if (y0 >= 0 && x1 <= w - 1) v += hy * lx * plane[(long)y0 * w + x1];
        if (y1 <= h - 1 && x0 >= 0) v += ly * hx * plane[(long)y1 * w + x0];
        if (y1 <= h - 1 && x1 <= w - 1) v += ly * lx * plane[(long)y1 * w + x1];
        return v;
    }
}
