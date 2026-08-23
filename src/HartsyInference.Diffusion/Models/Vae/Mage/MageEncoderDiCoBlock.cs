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
        _conv1W = TensorCasts.EnsureF32(w[$"{p}.conv1.weight"]); _conv1B = TensorCasts.EnsureF32(w[$"{p}.conv1.bias"]);
        _conv2W = TensorCasts.EnsureF32(w[$"{p}.conv2.weight"]); _conv2B = TensorCasts.EnsureF32(w[$"{p}.conv2.bias"]);
        _conv3W = TensorCasts.EnsureF32(w[$"{p}.conv3.weight"]); _conv3B = TensorCasts.EnsureF32(w[$"{p}.conv3.bias"]);
        _conv4W = TensorCasts.EnsureF32(w[$"{p}.conv4.weight"]); _conv4B = TensorCasts.EnsureF32(w[$"{p}.conv4.bias"]);
        _conv5W = TensorCasts.EnsureF32(w[$"{p}.conv5.weight"]); _conv5B = TensorCasts.EnsureF32(w[$"{p}.conv5.bias"]);
        _caW = TensorCasts.EnsureF32(w[$"{p}.ca.1.weight"]); _caB = TensorCasts.EnsureF32(w[$"{p}.ca.1.bias"]);
        _norm1W = TensorCasts.EnsureF32(w[$"{p}.norm1.weight"]); _norm1B = TensorCasts.EnsureF32(w[$"{p}.norm1.bias"]);
        _norm2W = TensorCasts.EnsureF32(w[$"{p}.norm2.weight"]); _norm2B = TensorCasts.EnsureF32(w[$"{p}.norm2.bias"]);
    }

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
        MageVaeOps.ChannelLayerNormAffine(inp, x, _norm1W!, _norm1B!, b, _c, hw);
        Tensor c1 = new(inp.Shape, DType.F32);
        backend.Conv2D(c1, x, _conv1W!, _conv1B, 1, 1, 0, 0); x.Dispose();
        Tensor c2 = new(inp.Shape, DType.F32);
        backend.Conv2dDepthwise(c2, c1, _conv2W!, _conv2B, 1, 1, 1, 1); c1.Dispose();
        Tensor g = new(inp.Shape, DType.F32);
        backend.Gelu(g, c2); c2.Dispose();
        Tensor pooled = MageVaeOps.GlobalAvgPool(g, b, _c, hw);
        Tensor caConv = new(pooled.Shape, DType.F32);
        backend.Conv2D(caConv, pooled, _caW!, _caB, 1, 1, 0, 0); pooled.Dispose();
        Tensor ca = new(caConv.Shape, DType.F32);
        backend.Sigmoid(ca, caConv); caConv.Dispose();
        MageVaeOps.ScaleByChannel(g, ca, b, _c, hw); ca.Dispose();
        Tensor c3 = new(inp.Shape, DType.F32);
        backend.Conv2D(c3, g, _conv3W!, _conv3B, 1, 1, 0, 0); g.Dispose();
        Tensor mid = new(inp.Shape, DType.F32);
        backend.Add(mid, inp, c3); c3.Dispose();   // inp += x

        Tensor x2 = new(inp.Shape, DType.F32);
        MageVaeOps.ChannelLayerNormAffine(mid, x2, _norm2W!, _norm2B!, b, _c, hw);
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
}
