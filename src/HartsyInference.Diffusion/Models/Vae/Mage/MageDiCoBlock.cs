using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Vae.Mage;

/// <summary>One DiCo (Diffusion-Convolution) block from MageVAE's decoder trunk (mage_vae.py <c>DiCoBlock</c>,
/// L112-149). Runs at latent resolution on <c>[B, 384, h, w]</c>. Structure: two AdaLN-modulated branches sharing a
/// timestep-derived modulation <c>c</c> (constant at t=0):
/// <list type="bullet">
/// <item><b>conv branch:</b> modulate(LN2d(inp)) → conv1(1×1) → conv2(3×3 depthwise) → GELU → ×channel-attention →
/// conv3(1×1), gated by <c>gate_msa</c>.</item>
/// <item><b>mlp branch:</b> modulate(LN2d(·)) → conv4(1×1, 384→1536) → GELU → conv5(1×1, 1536→384), gated by
/// <c>gate_mlp</c>.</item>
/// </list>
/// <c>norm1</c>/<c>norm2</c> are channel-dim LayerNorm2d with <b>affine=false</b> (no params). <c>modulate(x)=x·(1+scale)+shift</c>
/// broadcast per-channel. Channel attention = global-avg-pool → 1×1 conv → sigmoid.</summary>
public sealed unsafe class MageDiCoBlock
{
    private readonly int _c;
    private Tensor? _conv1W, _conv1B, _conv2W, _conv2B, _conv3W, _conv3B, _conv4W, _conv4B, _conv5W, _conv5B;
    private Tensor? _caW, _caB;              // ca.1 (1×1 conv c→c)
    private Tensor? _adaW, _adaB;            // adaLN_modulation.1 (Linear c→6c)

    public MageDiCoBlock(int channels) => _c = channels;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string p)
    {
        _conv1W = F32(w[$"{p}.conv1.weight"]); _conv1B = F32(w[$"{p}.conv1.bias"]);
        _conv2W = F32(w[$"{p}.conv2.weight"]); _conv2B = F32(w[$"{p}.conv2.bias"]);
        _conv3W = F32(w[$"{p}.conv3.weight"]); _conv3B = F32(w[$"{p}.conv3.bias"]);
        _conv4W = F32(w[$"{p}.conv4.weight"]); _conv4B = F32(w[$"{p}.conv4.bias"]);
        _conv5W = F32(w[$"{p}.conv5.weight"]); _conv5B = F32(w[$"{p}.conv5.bias"]);
        _caW = F32(w[$"{p}.ca.1.weight"]); _caB = F32(w[$"{p}.ca.1.bias"]);
        _adaW = F32(w[$"{p}.adaLN_modulation.1.weight"]); _adaB = F32(w[$"{p}.adaLN_modulation.1.bias"]);
    }

    private static Tensor F32(Tensor t) => t.DType == DType.F32 ? t : t.CastTo(DType.F32);

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor? t in new[] { _conv1W, _conv1B, _conv2W, _conv2B, _conv3W, _conv3B, _conv4W, _conv4B,
            _conv5W, _conv5B, _caW, _caB, _adaW, _adaB })
            if (t is not null) yield return t;
    }

    public Tensor Forward(IBackend backend, Tensor inp, Tensor c)   // inp [b,C,h,w], c [b,C]
    {
        int b = (int)inp.Shape[0], h = (int)inp.Shape[2], w = (int)inp.Shape[3];
        int inter = (int)_conv4W!.Shape[0];   // 1536

        // adaLN: silu(c) → Linear(C→6C) → 6 per-channel modulation vectors [b, C].
        Tensor mod = new(new TensorShape(b, 6 * _c), DType.F32);
        Tensor act = new(new TensorShape(b, _c), DType.F32);
        backend.Silu(act, c);
        backend.Linear(mod, act, _adaW!, _adaB);
        act.Dispose();
        float* mp = (float*)mod.DataPointer;

        // ── conv branch ──
        Tensor x = new(inp.Shape, DType.F32);
        ChannelLayerNormAffineFalse(inp, x);
        ModulatePerChannel(x, mp, shiftOff: 0, scaleOff: _c, b, _c, h * w);   // shift_msa=chunk0, scale_msa=chunk1
        Tensor c1 = new(inp.Shape, DType.F32);
        backend.Conv2D(c1, x, _conv1W!, _conv1B, 1, 1, 0, 0); x.Dispose();
        Tensor c2 = new(inp.Shape, DType.F32);
        backend.Conv2dDepthwise(c2, c1, _conv2W!, _conv2B, 1, 1, 1, 1); c1.Dispose();
        Tensor g = new(inp.Shape, DType.F32);
        backend.Gelu(g, c2); c2.Dispose();
        // channel attention: global avg pool → 1×1 conv → sigmoid → scale.
        Tensor pooled = GlobalAvgPool(g, b, _c, h, w);            // [b,C,1,1]
        Tensor caConv = new(pooled.Shape, DType.F32);
        backend.Conv2D(caConv, pooled, _caW!, _caB, 1, 1, 0, 0); pooled.Dispose();
        Tensor ca = new(caConv.Shape, DType.F32);
        backend.Sigmoid(ca, caConv); caConv.Dispose();
        ScaleByChannel(g, ca, b, _c, h * w); ca.Dispose();
        Tensor c3 = new(inp.Shape, DType.F32);
        backend.Conv2D(c3, g, _conv3W!, _conv3B, 1, 1, 0, 0); g.Dispose();
        // inp = inp + gate_msa * c3   (gate_msa = chunk2)
        Tensor afterMsa = new(inp.Shape, DType.F32);
        AddGated(inp, c3, mp, gateOff: 2 * _c, afterMsa, b, _c, h * w); c3.Dispose();

        // ── mlp branch ──
        Tensor x2 = new(inp.Shape, DType.F32);
        ChannelLayerNormAffineFalse(afterMsa, x2);
        ModulatePerChannel(x2, mp, shiftOff: 3 * _c, scaleOff: 4 * _c, b, _c, h * w);   // shift_mlp=3, scale_mlp=4
        Tensor c4 = new(new TensorShape(b, inter, h, w), DType.F32);
        backend.Conv2D(c4, x2, _conv4W!, _conv4B, 1, 1, 0, 0); x2.Dispose();
        Tensor g2 = new(c4.Shape, DType.F32);
        backend.Gelu(g2, c4); c4.Dispose();
        Tensor c5 = new(inp.Shape, DType.F32);
        backend.Conv2D(c5, g2, _conv5W!, _conv5B, 1, 1, 0, 0); g2.Dispose();
        Tensor outp = new(inp.Shape, DType.F32);
        AddGated(afterMsa, c5, mp, gateOff: 5 * _c, outp, b, _c, h * w);   // gate_mlp = chunk5
        afterMsa.Dispose(); c5.Dispose(); mod.Dispose();
        return outp;
    }

    // LayerNorm over the channel dim (NCHW), affine=false: per (n, spatial) normalize the C-vector (mean/var over C).
    private void ChannelLayerNormAffineFalse(Tensor input, Tensor output)
    {
        int b = (int)input.Shape[0], c = _c, hw = (int)input.Shape[2] * (int)input.Shape[3];
        float* src = (float*)input.DataPointer; float* dst = (float*)output.DataPointer;
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
                    dst[o] = (float)((src[o] - mean) * inv);
                }
            }
    }

    // x[b,c,h,w] = x*(1+scale[b,c]) + shift[b,c]  (per-channel broadcast over spatial).
    private static void ModulatePerChannel(Tensor x, float* mod, int shiftOff, int scaleOff, int b, int c, int hw)
    {
        float* p = (float*)x.DataPointer;
        for (int bi = 0; bi < b; bi++)
            for (int ch = 0; ch < c; ch++)
            {
                float shift = mod[bi * 6 * c + shiftOff + ch];
                float scale = mod[bi * 6 * c + scaleOff + ch];
                long baseO = (long)(bi * c + ch) * hw;
                for (int s = 0; s < hw; s++) { long o = baseO + s; p[o] = p[o] * (1f + scale) + shift; }
            }
    }

    // out[b,c,h,w] = residual + gate[b,c] * branch.
    private static void AddGated(Tensor residual, Tensor branch, float* mod, int gateOff, Tensor output, int b, int c, int hw)
    {
        float* r = (float*)residual.DataPointer; float* br = (float*)branch.DataPointer; float* o = (float*)output.DataPointer;
        for (int bi = 0; bi < b; bi++)
            for (int ch = 0; ch < c; ch++)
            {
                float gate = mod[bi * 6 * c + gateOff + ch];
                long baseO = (long)(bi * c + ch) * hw;
                for (int s = 0; s < hw; s++) { long k = baseO + s; o[k] = r[k] + gate * br[k]; }
            }
    }

    private static Tensor GlobalAvgPool(Tensor x, int b, int c, int h, int w)
    {
        int hw = h * w;
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
        float* p = (float*)x.DataPointer; float* g = (float*)ca.DataPointer;   // ca [b,c,1,1]
        for (int bi = 0; bi < b; bi++)
            for (int ch = 0; ch < c; ch++)
            {
                float s = g[bi * c + ch]; long baseO = (long)(bi * c + ch) * hw;
                for (int k = 0; k < hw; k++) p[baseO + k] *= s;
            }
    }
}
