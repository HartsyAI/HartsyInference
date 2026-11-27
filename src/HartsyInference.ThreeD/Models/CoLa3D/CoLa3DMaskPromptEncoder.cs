using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using static HartsyInference.ThreeD.Models.CoLa3D.CoLa3DOps;

namespace HartsyInference.ThreeD.Models.CoLa3D;

/// <summary>Query tokens from prompt 2D masks: a three-stage stride-2 CNN with GroupNorm and exact GELU, a 1×1 projection
/// and a spatial mean; the <c>cnn_centerpe</c> variant adds a sinusoidal encoding of the mask centre. Borrows its weights.</summary>
internal sealed class CoLa3DMaskPromptEncoder
{
    private const float GroupNormEps = 1e-5f;
    private const float NormEps = 1e-6f;

    private readonly int _width, _pePerAxis;
    private readonly Conv[] _convs = new Conv[4];
    private readonly Affine[] _groupNorms = new Affine[3];
    private readonly bool _centerPe;
    private readonly Affine _cnnNorm, _peNorm;
    private readonly Dense _peUp, _peDown, _fuseUp, _fuseDown;

    internal CoLa3DMaskPromptEncoder(IReadOnlyDictionary<string, Tensor> w, CoLa3DConfig config)
    {
        _width = config.HiddenDim;
        _pePerAxis = config.MaskPePerAxis;
        _centerPe = config.MaskEncoder == CoLa3DConfig.MaskEncoderKind.CnnCenterPe;
        const string e = "maskPrompt.conv";
        _convs[0] = new Conv(w[$"{e}.0.weight"], w[$"{e}.0.bias"]);
        _groupNorms[0] = new Affine(w[$"{e}.1.weight"], w[$"{e}.1.bias"]);
        _convs[1] = new Conv(w[$"{e}.3.weight"], w[$"{e}.3.bias"]);
        _groupNorms[1] = new Affine(w[$"{e}.4.weight"], w[$"{e}.4.bias"]);
        _convs[2] = new Conv(w[$"{e}.6.weight"], w[$"{e}.6.bias"]);
        _groupNorms[2] = new Affine(w[$"{e}.7.weight"], w[$"{e}.7.bias"]);
        _convs[3] = new Conv(w[$"{e}.9.weight"], w[$"{e}.9.bias"]);
        if (!_centerPe)
            return;
        _cnnNorm = LoadAffine(w, "maskPrompt.cnnNorm");
        _peUp = LoadDense(w, "maskPrompt.peUp");
        _peDown = LoadDense(w, "maskPrompt.peDown");
        _peNorm = LoadAffine(w, "maskPrompt.peNorm");
        _fuseUp = LoadDense(w, "maskPrompt.fuseUp");
        _fuseDown = LoadDense(w, "maskPrompt.fuseDown");
    }

    /// <summary>Owned query rows <c>[1, B·Q, C]</c> for masks <c>[B, Q, H, W]</c>, in batch-major order.</summary>
    internal Tensor Encode(IBackend backend, Tensor masks)
    {
        int bq = (int)(masks.Shape[0] * masks.Shape[1]);
        int h = (int)masks.Shape[2], w = (int)masks.Shape[3];
        Tensor image = masks.Reshape(new TensorShape(bq, 1, h, w));

        Tensor feat = image;
        int fh = h, fw = w;
        for (int stage = 0; stage < 3; stage++)
        {
            int nextH = Stride2(fh), nextW = Stride2(fw);
            long channels = _convs[stage].Weight.Shape[0];
            Tensor conv = Alloc(bq, channels, nextH, nextW);
            backend.Conv2D(conv, feat, _convs[stage].Weight, _convs[stage].Bias, 2, 2, 1, 1);
            if (stage > 0)
                feat.Dispose();
            Tensor normed = new(conv.Shape, DType.F32);
            backend.GroupNorm(normed, conv, _groupNorms[stage].Weight, _groupNorms[stage].Bias, 1, GroupNormEps);
            conv.Dispose();
            feat = new Tensor(normed.Shape, DType.F32);
            backend.GeluErf(feat, normed);
            normed.Dispose();
            fh = nextH;
            fw = nextW;
        }

        Tensor projected = Alloc(bq, _width, fh, fw);
        backend.Conv2D(projected, feat, _convs[3].Weight, _convs[3].Bias, 1, 1, 0, 0);
        feat.Dispose();
        Tensor cnn = Alloc(1, bq, _width);
        MeanOverSpatial(projected, cnn, fh * fw);
        projected.Dispose();

        Tensor output = _centerPe ? AddCenterPe(backend, cnn, masks, bq, h, w) : cnn;
        if (_centerPe)
            cnn.Dispose();
        return output;
    }

    private Tensor AddCenterPe(IBackend backend, Tensor cnn, Tensor masks, int bq, int h, int w)
    {
        Tensor cnnNormed = Norm(backend, cnn, _cnnNorm, NormEps);
        Tensor sinusoid = Alloc(1, bq, 2L * _pePerAxis);
        float[] cy = new float[bq];
        float[] cx = new float[bq];
        Centroids(masks, h, w, cy, cx);
        float[] rowY = new float[bq * _pePerAxis];
        float[] rowX = new float[bq * _pePerAxis];
        CoLa3DOps.SinusoidalEmbedding(cy, _pePerAxis, rowY);
        CoLa3DOps.SinusoidalEmbedding(cx, _pePerAxis, rowX);
        Span<float> emb = sinusoid.AsSpan<float>();
        for (int i = 0; i < bq; i++)
        {
            rowY.AsSpan(i * _pePerAxis, _pePerAxis).CopyTo(emb.Slice(i * 2 * _pePerAxis, _pePerAxis));
            rowX.AsSpan(i * _pePerAxis, _pePerAxis).CopyTo(emb.Slice(i * 2 * _pePerAxis + _pePerAxis, _pePerAxis));
        }

        Tensor peHidden = Linear(backend, sinusoid, _peUp, _width);
        sinusoid.Dispose();
        Tensor peAct = new(peHidden.Shape, DType.F32);
        backend.GeluErf(peAct, peHidden);
        peHidden.Dispose();
        Tensor peOut = Linear(backend, peAct, _peDown, _width);
        peAct.Dispose();
        Tensor peNormed = Norm(backend, peOut, _peNorm, NormEps);
        peOut.Dispose();

        Tensor sum = new(cnnNormed.Shape, DType.F32);
        backend.Add(sum, cnnNormed, peNormed);
        cnnNormed.Dispose();
        peNormed.Dispose();

        Tensor fuseHidden = Linear(backend, sum, _fuseUp, _width);
        sum.Dispose();
        Tensor fuseAct = new(fuseHidden.Shape, DType.F32);
        backend.GeluErf(fuseAct, fuseHidden);
        fuseHidden.Dispose();
        Tensor output = Linear(backend, fuseAct, _fuseDown, _width);
        fuseAct.Dispose();
        return output;
    }

    /// <summary>Mask centre as <c>(cy, cx)</c> in [0, 1], the <c>_mask_centroid</c> rule with mass clamped to at least 1.</summary>
    private static void Centroids(Tensor masks, int h, int w, float[] cy, float[] cx)
    {
        ReadOnlySpan<float> data = masks.AsReadOnlySpan<float>();
        int maps = cy.Length;
        for (int m = 0; m < maps; m++)
        {
            ReadOnlySpan<float> map = data.Slice(m * h * w, h * w);
            float mass = 0f, sumY = 0f, sumX = 0f;
            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    float v = map[y * w + x];
                    mass += v;
                    sumY += y * v;
                    sumX += x * v;
                }
            }

            mass = MathF.Max(mass, 1f);
            cy[m] = sumY / mass / MathF.Max(h - 1, 1);
            cx[m] = sumX / mass / MathF.Max(w - 1, 1);
        }
    }

    /// <summary>Spatial mean of <c>[n, C, S]</c> into the flat <c>[1, n, C]</c> destination.</summary>
    private static void MeanOverSpatial(Tensor src, Tensor dst, int spatial)
    {
        int channels = (int)src.Shape[1], n = (int)src.Shape[0];
        ReadOnlySpan<float> s = src.AsReadOnlySpan<float>();
        Span<float> d = dst.AsSpan<float>();
        float inv = 1f / spatial;
        for (int i = 0; i < n; i++)
        {
            for (int c = 0; c < channels; c++)
            {
                float sum = 0f;
                ReadOnlySpan<float> plane = s.Slice((i * channels + c) * spatial, spatial);
                for (int k = 0; k < spatial; k++)
                    sum += plane[k];
                d[i * channels + c] = sum * inv;
            }
        }
    }

    /// <summary>Output size of a 3×3 stride-2 convolution with padding 1.</summary>
    private static int Stride2(int size) => (size - 1) / 2 + 1;

    private readonly record struct Conv(Tensor Weight, Tensor Bias);
}
