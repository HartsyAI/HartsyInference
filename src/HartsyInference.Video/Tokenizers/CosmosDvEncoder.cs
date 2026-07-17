using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Vae;

namespace HartsyInference.Video.Tokenizers;

/// <summary>The learned causal-conv-3D autoencoder <b>encoder</b> of Cosmos-Tokenize1-DV8x16x16-720p, recovered
/// bit-for-bit from NVIDIA's shipped <c>encoder.jit</c> (the <c>network.*</c> BF16 weights). Runs the exact
/// pipeline the tokenizer applies before FSQ:
///
/// <list type="number">
///   <item><b>Patcher3d</b> — a 2-level separable Haar wavelet (filters <c>[1/√2, 1/√2]</c>) implemented as grouped
///   depthwise stride-2 convolutions with reflect padding, band-major channel packing (<c>out = band·C + c</c>) and a
///   <c>1/√2</c> per-level renorm. Level 1 replicates the first frame 4× (causal temporal boundary); RGB 3ch → 192ch.</item>
///   <item><b>conv_in</b> (192→128) then 3 down stages (ch_mult <c>[1,2,4,4]</c>, base 128, 2 res-blocks each): down.0
///   spatiotemporal downsample, down.1 spatial-only, down.2 none.</item>
///   <item><b>mid</b> — res-block, spatial self-attention, causal <b>temporal</b> self-attention, res-block.</item>
///   <item><b>norm_out</b> → SiLU → <b>conv_out</b> (512→16) → <b>quant_conv</b> (16→6) = the pre-FSQ latent.</item>
/// </list>
///
/// <para>Downsample: right-pad H/W by 1, stride-2 3×3 conv + stride-2 avg-pool residual (spatial); replicate first
/// frame, stride-2 3×1×1 conv + stride-2 avg-pool residual (temporal); then a 1×1×1 projection. Shared conv / norm /
/// attention machinery lives in <see cref="CosmosDvModule"/>. B=1.</para></summary>
internal sealed unsafe class CosmosDvEncoder : CosmosDvModule
{
    public CosmosDvEncoder(Dictionary<string, Tensor> weightsF32) : base(weightsF32) { }

    /// <summary>Encodes RGB video <c>[1,3,T,H,W]</c> (in <c>[-1,1]</c>) to the pre-FSQ continuous latent
    /// <c>[1,6,Tl,Hl,Wl]</c>.</summary>
    public Tensor Encode(IBackend backend, Tensor video)
    {
        Tensor x = Patcher(video);
        Tensor y = FactConv(backend, x, "encoder.conv_in"); x.Dispose();
        y = Chain(backend, y, "encoder.down.0.block.0");
        y = Chain(backend, y, "encoder.down.0.block.1");
        Tensor d0 = Downsample(backend, y, "encoder.down.0.downsample", spatiotemporal: true); y.Dispose(); y = d0;
        y = Chain(backend, y, "encoder.down.1.block.0");
        y = Chain(backend, y, "encoder.down.1.block.1");
        Tensor d1 = Downsample(backend, y, "encoder.down.1.downsample", spatiotemporal: false); y.Dispose(); y = d1;
        y = Chain(backend, y, "encoder.down.2.block.0");
        y = Chain(backend, y, "encoder.down.2.block.1");
        y = Chain(backend, y, "encoder.mid.block_1");
        Tensor a0 = SpatialAttn(backend, y, "encoder.mid.attn_1.0"); y.Dispose();
        Tensor a1 = TemporalAttn(backend, a0, "encoder.mid.attn_1.1"); a0.Dispose();
        y = Chain(backend, a1, "encoder.mid.block_2"); a1.Dispose();
        Tensor n = GroupNorm(y, "encoder.norm_out", silu: true); y.Dispose();
        Tensor co = FactConv(backend, n, "encoder.conv_out"); n.Dispose();
        Tensor q = OneXOne(backend, co, "quant_conv.conv3d"); co.Dispose();
        return q;
    }

    private Tensor Chain(IBackend backend, Tensor x, string resBase)
    {
        Tensor r = ResBlock(backend, x, resBase);
        x.Dispose();
        return r;
    }

    private Tensor Downsample(IBackend backend, Tensor x, string baseName, bool spatiotemporal)
    {
        Tensor xp = PadSpatialRight1(x);
        Tensor conv1 = Conv(baseName + ".conv1.conv3d", 1, 2, 2, 0, 0, 0, false).Forward(backend, xp);
        Tensor pool = AvgPoolSpatial(xp);
        xp.Dispose();
        AddInPlace(conv1, pool); pool.Dispose();
        Tensor x0 = conv1;
        if (spatiotemporal)
        {
            Tensor conv2 = Conv(baseName + ".conv2.conv3d", 2, 1, 1, 1, 0, 0, true).Forward(backend, x0);
            Tensor x1 = ReplicateFirstFrame1(x0);
            Tensor pool2 = AvgPoolTemporal(x1);
            x1.Dispose();
            AddInPlace(conv2, pool2); pool2.Dispose();
            x0.Dispose();
            x0 = conv2;
        }
        Tensor outT = OneXOne(backend, x0, baseName + ".conv3.conv3d");
        x0.Dispose();
        return outT;
    }

    // ── wavelet patcher (2-level Haar, band-major) ─────────────────────────
    private static Tensor Patcher(Tensor video)
    {
        Tensor l1 = WaveletLevel(video, firstFrameRepeat: true);
        Tensor l2 = WaveletLevel(l1, firstFrameRepeat: false);
        l1.Dispose();
        return l2;
    }

    private static Tensor WaveletLevel(Tensor x, bool firstFrameRepeat)
    {
        Tensor xin = firstFrameRepeat ? RepeatFirstFrameInterleave4(x) : x;
        (int b, int c, int t, int h, int w) = Dims(xin);
        int ot = OutLen(t), oh = OutLen(h), ow = OutLen(w);
        Tensor outT = new(new TensorShape([(long)b, c * 8, ot, oh, ow]), DType.F32);
        float* sp = (float*)xin.DataPointer, op = (float*)outT.DataPointer;
        const float norm3 = Wav * Wav * Wav;
        float renorm = norm3 / MathF.Sqrt(2f);
        long hw = (long)h * w;
        float* cell = stackalloc float[8];
        for (int bi = 0; bi < b; bi++)
            for (int ci = 0; ci < c; ci++)
            {
                long inBase = ((long)bi * c + ci) * t * hw;
                for (int it = 0; it < ot; it++)
                {
                    int t0 = Reflect(it * 2, t), t1 = Reflect(it * 2 + 1, t);
                    for (int ih = 0; ih < oh; ih++)
                    {
                        int h0 = Reflect(ih * 2, h), h1 = Reflect(ih * 2 + 1, h);
                        for (int iw = 0; iw < ow; iw++)
                        {
                            int w0 = Reflect(iw * 2, w), w1 = Reflect(iw * 2 + 1, w);
                            cell[0] = sp[inBase + (long)t0 * hw + (long)h0 * w + w0];
                            cell[1] = sp[inBase + (long)t0 * hw + (long)h0 * w + w1];
                            cell[2] = sp[inBase + (long)t0 * hw + (long)h1 * w + w0];
                            cell[3] = sp[inBase + (long)t0 * hw + (long)h1 * w + w1];
                            cell[4] = sp[inBase + (long)t1 * hw + (long)h0 * w + w0];
                            cell[5] = sp[inBase + (long)t1 * hw + (long)h0 * w + w1];
                            cell[6] = sp[inBase + (long)t1 * hw + (long)h1 * w + w0];
                            cell[7] = sp[inBase + (long)t1 * hw + (long)h1 * w + w1];
                            for (int s = 0; s < 8; s++)
                            {
                                int sT = (s >> 2) & 1, sH = (s >> 1) & 1, sW = s & 1;
                                float acc = 0f;
                                for (int d = 0; d < 8; d++)
                                {
                                    int dt = (d >> 2) & 1, dh = (d >> 1) & 1, dw = d & 1;
                                    acc += Sgn(sT, dt) * Sgn(sH, dh) * Sgn(sW, dw) * cell[d];
                                }
                                long co = (long)s * c + ci;
                                op[(((long)bi * (c * 8) + co) * ot + it) * oh * ow + (long)ih * ow + iw] = acc * renorm;
                            }
                        }
                    }
                }
            }
        if (firstFrameRepeat) xin.Dispose();
        return outT;
    }

    private static int OutLen(int lenIn) => (lenIn - 1) / 2 + 1;   // reflect-pad-right-1 then k2s2
    private static int Reflect(int i, int len) => i < len ? i : 2 * len - 2 - i;   // torch 'reflect' pad-right

    private static Tensor PadSpatialRight1(Tensor x)   // [b,c,t,h,w] -> [b,c,t,h+1,w+1] zero right pad
    {
        (int b, int c, int t, int h, int w) = Dims(x);
        int hp = h + 1, wp = w + 1;
        Tensor o = new(new TensorShape([(long)b, c, t, hp, wp]), DType.F32);
        float* sp = (float*)x.DataPointer, op = (float*)o.DataPointer;
        new Span<float>(op, (int)o.Shape.ElementCount).Clear();
        for (int bi = 0; bi < b; bi++)
            for (int ci = 0; ci < c; ci++)
                for (int ti = 0; ti < t; ti++)
                    for (int hi = 0; hi < h; hi++)
                    {
                        long si = ((((long)bi * c + ci) * t + ti) * h + hi) * w;
                        long di = ((((long)bi * c + ci) * t + ti) * hp + hi) * wp;
                        Buffer.MemoryCopy(sp + si, op + di, (long)w * 4, (long)w * 4);
                    }
        return o;
    }

    private static Tensor AvgPoolSpatial(Tensor x)   // k(1,2,2) s(1,2,2)
    {
        (int b, int c, int t, int h, int w) = Dims(x);
        int oh = h / 2, ow = w / 2;
        Tensor o = new(new TensorShape([(long)b, c, t, oh, ow]), DType.F32);
        float* sp = (float*)x.DataPointer, op = (float*)o.DataPointer;
        for (int bi = 0; bi < b; bi++)
            for (int ci = 0; ci < c; ci++)
                for (int ti = 0; ti < t; ti++)
                {
                    long sBase = (((long)bi * c + ci) * t + ti) * h * w;
                    long dBase = (((long)bi * c + ci) * t + ti) * oh * ow;
                    for (int hi = 0; hi < oh; hi++)
                        for (int wi = 0; wi < ow; wi++)
                        {
                            float s = sp[sBase + ((long)(hi * 2) * w) + wi * 2]
                                    + sp[sBase + ((long)(hi * 2) * w) + wi * 2 + 1]
                                    + sp[sBase + ((long)(hi * 2 + 1) * w) + wi * 2]
                                    + sp[sBase + ((long)(hi * 2 + 1) * w) + wi * 2 + 1];
                            op[dBase + (long)hi * ow + wi] = s * 0.25f;
                        }
                }
        return o;
    }

    private static Tensor AvgPoolTemporal(Tensor x)   // k(2,1,1) s(2,1,1)
    {
        (int b, int c, int t, int h, int w) = Dims(x);
        int ot = t / 2; long hw = (long)h * w;
        Tensor o = new(new TensorShape([(long)b, c, ot, h, w]), DType.F32);
        float* sp = (float*)x.DataPointer, op = (float*)o.DataPointer;
        for (int bi = 0; bi < b; bi++)
            for (int ci = 0; ci < c; ci++)
                for (int ti = 0; ti < ot; ti++)
                {
                    long s0 = (((long)bi * c + ci) * t + ti * 2) * hw;
                    long s1 = (((long)bi * c + ci) * t + ti * 2 + 1) * hw;
                    long d = (((long)bi * c + ci) * ot + ti) * hw;
                    for (long p = 0; p < hw; p++) op[d + p] = (sp[s0 + p] + sp[s1 + p]) * 0.5f;
                }
        return o;
    }

    private static Tensor ReplicateFirstFrame1(Tensor x)   // prepend one copy of frame 0 along T
    {
        (int b, int c, int t, int h, int w) = Dims(x);
        long hw = (long)h * w;
        Tensor o = new(new TensorShape([(long)b, c, t + 1, h, w]), DType.F32);
        float* sp = (float*)x.DataPointer, op = (float*)o.DataPointer;
        for (int bi = 0; bi < b; bi++)
            for (int ci = 0; ci < c; ci++)
            {
                long sBase = ((long)bi * c + ci) * t * hw;
                long dBase = ((long)bi * c + ci) * (t + 1) * hw;
                Buffer.MemoryCopy(sp + sBase, op + dBase, hw * 4, hw * 4);
                Buffer.MemoryCopy(sp + sBase, op + dBase + hw, (long)t * hw * 4, (long)t * hw * 4);
            }
        return o;
    }

    private static Tensor RepeatFirstFrameInterleave4(Tensor x)   // frame0 ×4 then the rest
    {
        (int b, int c, int t, int h, int w) = Dims(x);
        long hw = (long)h * w;
        int ot = t + 3;
        Tensor o = new(new TensorShape([(long)b, c, ot, h, w]), DType.F32);
        float* sp = (float*)x.DataPointer, op = (float*)o.DataPointer;
        for (int bi = 0; bi < b; bi++)
            for (int ci = 0; ci < c; ci++)
            {
                long sBase = ((long)bi * c + ci) * t * hw;
                long dBase = ((long)bi * c + ci) * ot * hw;
                for (int r = 0; r < 4; r++) Buffer.MemoryCopy(sp + sBase, op + dBase + (long)r * hw, hw * 4, hw * 4);
                if (t > 1) Buffer.MemoryCopy(sp + sBase + hw, op + dBase + 4 * hw, (long)(t - 1) * hw * 4, (long)(t - 1) * hw * 4);
            }
        return o;
    }
}
