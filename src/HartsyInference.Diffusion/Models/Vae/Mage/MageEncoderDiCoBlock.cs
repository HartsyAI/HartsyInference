using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Vae.Mage;

/// <summary>The MageVAE encoder's head block (mage_vae.py <c>_EncoderDiCoBlock</c>, L152-177) — a DiCoBlock WITHOUT
/// AdaLN and WITHOUT gating: two plain residual adds, and its channel-dim <c>norm1</c>/<c>norm2</c> are affine
/// (weight+bias, unlike the decoder trunk's affine-false norms). Runs at latent resolution on <c>[B, 768, h, w]</c>.
/// Forward: <c>x = norm1(inp); x = gelu(conv2(conv1(x))); x = x·ca(x); x = conv3(x); inp += x;
/// return inp + conv5(gelu(conv4(norm2(inp))))</c>. conv2 is depthwise (groups=768).</summary>
public sealed unsafe class MageEncoderDiCoBlock
{
    private readonly int _c;
    private Tensor? _conv1W, _conv1B, _conv2W, _conv2B, _conv3W, _conv3B, _conv4W, _conv4B, _conv5W, _conv5B;
    private Tensor? _caW, _caB;
    private Tensor? _norm1W, _norm1B, _norm2W, _norm2B;   // affine channel-dim LayerNorm2d

    public MageEncoderDiCoBlock(int channels) => _c = channels;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string p)
    {
        _conv1W = F32(w[$"{p}.conv1.weight"]); _conv1B = F32(w[$"{p}.conv1.bias"]);
        _conv2W = F32(w[$"{p}.conv2.weight"]); _conv2B = F32(w[$"{p}.conv2.bias"]);
        _conv3W = F32(w[$"{p}.conv3.weight"]); _conv3B = F32(w[$"{p}.conv3.bias"]);
        _conv4W = F32(w[$"{p}.conv4.weight"]); _conv4B = F32(w[$"{p}.conv4.bias"]);
        _conv5W = F32(w[$"{p}.conv5.weight"]); _conv5B = F32(w[$"{p}.conv5.bias"]);
        _caW = F32(w[$"{p}.ca.1.weight"]); _caB = F32(w[$"{p}.ca.1.bias"]);
        _norm1W = F32(w[$"{p}.norm1.weight"]); _norm1B = F32(w[$"{p}.norm1.bias"]);
        _norm2W = F32(w[$"{p}.norm2.weight"]); _norm2B = F32(w[$"{p}.norm2.bias"]);
    }

    private static Tensor F32(Tensor t) => t.DType == DType.F32 ? t : t.CastTo(DType.F32);

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor? t in new[] { _conv1W, _conv1B, _conv2W, _conv2B, _conv3W, _conv3B, _conv4W, _conv4B,
            _conv5W, _conv5B, _caW, _caB, _norm1W, _norm1B, _norm2W, _norm2B })
            if (t is not null) yield return t;
    }

    public Tensor Forward(IBackend backend, Tensor inp)   // [b, C, h, w]
    {
        int b = (int)inp.Shape[0], h = (int)inp.Shape[2], w = (int)inp.Shape[3], hw = h * w;
        int inter = (int)_conv4W!.Shape[0];   // 3072

        Tensor x = new(inp.Shape, DType.F32);
        ChannelLayerNormAffine(inp, x, _norm1W!, _norm1B!);
        Tensor c1 = new(inp.Shape, DType.F32);
        backend.Conv2D(c1, x, _conv1W!, _conv1B, 1, 1, 0, 0); x.Dispose();
        Tensor c2 = new(inp.Shape, DType.F32);
        backend.Conv2dDepthwise(c2, c1, _conv2W!, _conv2B, 1, 1, 1, 1); c1.Dispose();
        Tensor g = new(inp.Shape, DType.F32);
        backend.Gelu(g, c2); c2.Dispose();
        Tensor pooled = GlobalAvgPool(g, b, _c, hw);
        Tensor caConv = new(pooled.Shape, DType.F32);
        backend.Conv2D(caConv, pooled, _caW!, _caB, 1, 1, 0, 0); pooled.Dispose();
        Tensor ca = new(caConv.Shape, DType.F32);
        backend.Sigmoid(ca, caConv); caConv.Dispose();
        ScaleByChannel(g, ca, b, _c, hw); ca.Dispose();
        Tensor c3 = new(inp.Shape, DType.F32);
        backend.Conv2D(c3, g, _conv3W!, _conv3B, 1, 1, 0, 0); g.Dispose();
        Tensor mid = new(inp.Shape, DType.F32);
        backend.Add(mid, inp, c3); c3.Dispose();   // inp += x

        Tensor x2 = new(inp.Shape, DType.F32);
        ChannelLayerNormAffine(mid, x2, _norm2W!, _norm2B!);
        Tensor c4 = new(new TensorShape(b, inter, h, w), DType.F32);
        backend.Conv2D(c4, x2, _conv4W!, _conv4B, 1, 1, 0, 0); x2.Dispose();
        Tensor g2 = new(c4.Shape, DType.F32);
        backend.Gelu(g2, c4); c4.Dispose();
        Tensor c5 = new(inp.Shape, DType.F32);
        backend.Conv2D(c5, g2, _conv5W!, _conv5B, 1, 1, 0, 0); g2.Dispose();
        Tensor outp = new(inp.Shape, DType.F32);
        backend.Add(outp, mid, c5); mid.Dispose(); c5.Dispose();   // + conv5(...)
        return outp;
    }

    // Affine channel-dim LayerNorm2d: per (n, spatial) normalize over C, then y = norm·weight[c] + bias[c].
    private void ChannelLayerNormAffine(Tensor input, Tensor output, Tensor weight, Tensor bias)
    {
        int b = (int)input.Shape[0], c = _c, hw = (int)input.Shape[2] * (int)input.Shape[3];
        float* src = (float*)input.DataPointer; float* dst = (float*)output.DataPointer;
        float* gw = (float*)weight.DataPointer; float* gb = (float*)bias.DataPointer;
        for (int bi = 0; bi < b; bi++)
            for (int p = 0; p < hw; p++)
            {
                double mean = 0;
                for (int ch = 0; ch < c; ch++) mean += src[((long)(bi * c + ch) * hw) + p];
                mean /= c;
                double var = 0;
                for (int ch = 0; ch < c; ch++) { double d = src[((long)(bi * c + ch) * hw) + p] - mean; var += d * d; }
                float inv = (float)(1.0 / Math.Sqrt(var / c + 1e-6));
                for (int ch = 0; ch < c; ch++)
                {
                    long o = ((long)(bi * c + ch) * hw) + p;
                    dst[o] = (float)((src[o] - mean) * inv) * gw[ch] + gb[ch];
                }
            }
    }

    private static Tensor GlobalAvgPool(Tensor x, int b, int c, int hw)
    {
        Tensor outp = new(new TensorShape(b, c, 1, 1), DType.F32);
        float* src = (float*)x.DataPointer; float* dst = (float*)outp.DataPointer;
        for (int bi = 0; bi < b; bi++)
            for (int ch = 0; ch < c; ch++)
            {
                double sum = 0; long baseO = (long)(bi * c + ch) * hw;
                for (int s = 0; s < hw; s++) sum += src[baseO + s];
                dst[bi * c + ch] = (float)(sum / hw);
            }
        return outp;
    }

    private static void ScaleByChannel(Tensor x, Tensor ca, int b, int c, int hw)
    {
        float* p = (float*)x.DataPointer; float* g = (float*)ca.DataPointer;
        for (int bi = 0; bi < b; bi++)
            for (int ch = 0; ch < c; ch++)
            {
                float s = g[bi * c + ch]; long baseO = (long)(bi * c + ch) * hw;
                for (int k = 0; k < hw; k++) p[baseO + k] *= s;
            }
    }
}
