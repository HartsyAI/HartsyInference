using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Vision.Detection.Blocks;

namespace HartsyInference.Vision.Detection;

/// <summary>RT-DETR hybrid encoder: projects the three backbone feature maps to the transformer
/// hidden dim, runs <b>AIFI</b> (Attention-based Intra-scale Feature Interaction — one transformer
/// encoder block, with a 2D sincos position embedding, on the flattened top level S5), then fuses
/// the scales with a CNN <b>CCFM</b> (CSP-style Cross-scale Feature-fusion Module) laid out as an
/// FPN top-down + PAN bottom-up path.
/// <para>The AIFI self-attention adds the position embedding to the fused QKV input (rather than to
/// q/k only as in the reference) — a documented structural simplification for the fused-QKV path.
/// The CCFM RepBlocks are approximated by a 1×1 reduce conv over the concatenated pair; this keeps
/// the fusion shape-correct while parity against the reference RepC3 stack is deferred.</para></summary>
public sealed class RtDetrHybridEncoder
{
    private readonly RtDetrConfig _config;

    // Per-level 1x1 projection to hidden dim (input_proj). No activation (BN-folded conv only).
    private readonly ConvBnSilu _inputProj0;   // C3 -> hidden
    private readonly ConvBnSilu _inputProj1;   // C4 -> hidden
    private readonly ConvBnSilu _inputProj2;   // C5 -> hidden

    // AIFI transformer encoder block on the top level.
    private readonly RtDetrLinear _aifiQkv;     // hidden -> 3*hidden
    private readonly RtDetrLinear _aifiProj;    // hidden -> hidden
    private readonly RtDetrLayerNorm _aifiNorm1;
    private readonly RtDetrLayerNorm _aifiNorm2;
    private readonly RtDetrLinear _aifiFc1;     // hidden -> ffn
    private readonly RtDetrLinear _aifiFc2;     // ffn -> hidden

    // CCFM fusion convs. Fuse = 1x1 (2*hidden -> hidden); Down = 3x3 stride-2 (hidden -> hidden).
    private readonly ConvBnSilu _fuse4;   // up(S5) ++ S4 -> F4
    private readonly ConvBnSilu _fuse3;   // up(F4) ++ S3 -> P3
    private readonly ConvBnSilu _down3;   // P3 (stride 2)
    private readonly ConvBnSilu _fuseN4;  // down(P3) ++ F4 -> N4
    private readonly ConvBnSilu _down4;   // N4 (stride 2)
    private readonly ConvBnSilu _fuseN5;  // down(N4) ++ S5 -> N5

    /// <summary>Constructs the encoder from a config.</summary>
    public RtDetrHybridEncoder(RtDetrConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);
        _config = config;
        int h = config.HiddenDim;
        int ffn = config.FeedForwardDim;

        _inputProj0 = new ConvBnSilu(h, strideH: 1, strideW: 1, padH: 0, padW: 0, useSilu: false);
        _inputProj1 = new ConvBnSilu(h, strideH: 1, strideW: 1, padH: 0, padW: 0, useSilu: false);
        _inputProj2 = new ConvBnSilu(h, strideH: 1, strideW: 1, padH: 0, padW: 0, useSilu: false);

        _aifiQkv = new RtDetrLinear(3 * h);
        _aifiProj = new RtDetrLinear(h);
        _aifiNorm1 = new RtDetrLayerNorm(h, config.LayerNormEps);
        _aifiNorm2 = new RtDetrLayerNorm(h, config.LayerNormEps);
        _aifiFc1 = new RtDetrLinear(ffn);
        _aifiFc2 = new RtDetrLinear(h);

        _fuse4 = new ConvBnSilu(h, strideH: 1, strideW: 1, padH: 0, padW: 0);
        _fuse3 = new ConvBnSilu(h, strideH: 1, strideW: 1, padH: 0, padW: 0);
        _down3 = new ConvBnSilu(h, strideH: 2, strideW: 2, padH: 1, padW: 1);
        _fuseN4 = new ConvBnSilu(h, strideH: 1, strideW: 1, padH: 0, padW: 0);
        _down4 = new ConvBnSilu(h, strideH: 2, strideW: 2, padH: 1, padW: 1);
        _fuseN5 = new ConvBnSilu(h, strideH: 1, strideW: 1, padH: 0, padW: 0);
    }

    /// <summary>Loads encoder weights under <paramref name="prefix"/> (default <c>"encoder"</c>).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _inputProj0.LoadWeights(weights, $"{prefix}.input_proj.0.conv");
        _inputProj1.LoadWeights(weights, $"{prefix}.input_proj.1.conv");
        _inputProj2.LoadWeights(weights, $"{prefix}.input_proj.2.conv");

        _aifiQkv.LoadWeights(weights, $"{prefix}.aifi.qkv");
        _aifiProj.LoadWeights(weights, $"{prefix}.aifi.proj");
        _aifiNorm1.LoadWeights(weights, $"{prefix}.aifi.norm1");
        _aifiNorm2.LoadWeights(weights, $"{prefix}.aifi.norm2");
        _aifiFc1.LoadWeights(weights, $"{prefix}.aifi.ffn.fc1");
        _aifiFc2.LoadWeights(weights, $"{prefix}.aifi.ffn.fc2");

        _fuse4.LoadWeights(weights, $"{prefix}.ccfm.fuse4.conv");
        _fuse3.LoadWeights(weights, $"{prefix}.ccfm.fuse3.conv");
        _down3.LoadWeights(weights, $"{prefix}.ccfm.down3.conv");
        _fuseN4.LoadWeights(weights, $"{prefix}.ccfm.fuse_n4.conv");
        _down4.LoadWeights(weights, $"{prefix}.ccfm.down4.conv");
        _fuseN5.LoadWeights(weights, $"{prefix}.ccfm.fuse_n5.conv");
    }

    /// <summary>Yields every encoder weight for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _inputProj0.EnumerateWeights()) yield return t;
        foreach (Tensor t in _inputProj1.EnumerateWeights()) yield return t;
        foreach (Tensor t in _inputProj2.EnumerateWeights()) yield return t;
        foreach (Tensor t in _aifiQkv.EnumerateWeights()) yield return t;
        foreach (Tensor t in _aifiProj.EnumerateWeights()) yield return t;
        foreach (Tensor t in _aifiNorm1.EnumerateWeights()) yield return t;
        foreach (Tensor t in _aifiNorm2.EnumerateWeights()) yield return t;
        foreach (Tensor t in _aifiFc1.EnumerateWeights()) yield return t;
        foreach (Tensor t in _aifiFc2.EnumerateWeights()) yield return t;
        foreach (Tensor t in _fuse4.EnumerateWeights()) yield return t;
        foreach (Tensor t in _fuse3.EnumerateWeights()) yield return t;
        foreach (Tensor t in _down3.EnumerateWeights()) yield return t;
        foreach (Tensor t in _fuseN4.EnumerateWeights()) yield return t;
        foreach (Tensor t in _down4.EnumerateWeights()) yield return t;
        foreach (Tensor t in _fuseN5.EnumerateWeights()) yield return t;
    }

    /// <summary>Runs the encoder. Returns the three fused multi-scale memory maps <c>(P3, N4, N5)</c>
    /// each <c>[1, hidden, H_l, W_l]</c> at strides 8/16/32 — the caller owns and disposes all three.</summary>
    public (Tensor P3, Tensor N4, Tensor N5) Forward(IBackend backend, Tensor c3, Tensor c4, Tensor c5)
    {
        // Project every level to the hidden dim.
        Tensor s3 = _inputProj0.Forward(backend, c3);
        Tensor s4 = _inputProj1.Forward(backend, c4);
        Tensor s5Proj = _inputProj2.Forward(backend, c5);

        // AIFI on the top level.
        Tensor s5 = ApplyAifi(backend, s5Proj);
        s5Proj.Dispose();

        // FPN top-down: up(S5) ++ S4 -> F4 ; up(F4) ++ S3 -> P3.
        Tensor s5Up = Upsample2x(backend, s5);
        Tensor cat4 = ConcatChannel(backend, s5Up, s4);
        s5Up.Dispose(); s4.Dispose();
        Tensor f4 = _fuse4.Forward(backend, cat4); cat4.Dispose();

        Tensor f4Up = Upsample2x(backend, f4);
        Tensor cat3 = ConcatChannel(backend, f4Up, s3);
        f4Up.Dispose(); s3.Dispose();
        Tensor p3 = _fuse3.Forward(backend, cat3); cat3.Dispose();

        // PAN bottom-up: down(P3) ++ F4 -> N4 ; down(N4) ++ S5 -> N5.
        Tensor p3Down = _down3.Forward(backend, p3);
        Tensor catN4 = ConcatChannel(backend, p3Down, f4);
        p3Down.Dispose(); f4.Dispose();
        Tensor n4 = _fuseN4.Forward(backend, catN4); catN4.Dispose();

        Tensor n4Down = _down4.Forward(backend, n4);
        Tensor catN5 = ConcatChannel(backend, n4Down, s5);
        n4Down.Dispose(); s5.Dispose();
        Tensor n5 = _fuseN5.Forward(backend, catN5); catN5.Dispose();

        return (p3, n4, n5);
    }

    /// <summary>One transformer encoder layer on the flattened top level with a 2D sincos position
    /// embedding: self-attention (post-norm) + FFN. Input/output are <c>[1, hidden, H, W]</c>.</summary>
    private Tensor ApplyAifi(IBackend backend, Tensor feat)
    {
        int hidden = _config.HiddenDim;
        int height = (int)feat.Shape[2];
        int width = (int)feat.Shape[3];
        int tokens = height * width;

        // Flatten NCHW -> [1, HW, C].
        Tensor src = new Tensor(new TensorShape(1, tokens, hidden), DType.F32);
        backend.Transpose2D(src, feat, hidden, tokens);

        // Add the 2D sincos position embedding, then run fused-QKV self-attention.
        Tensor pos = BuildSinCosPositionEmbedding(height, width, hidden);
        Tensor srcPos = new Tensor(src.Shape, DType.F32);
        backend.Add(srcPos, src, pos);
        pos.Dispose();

        Tensor qkv = _aifiQkv.Forward(backend, srcPos);
        srcPos.Dispose();
        Tensor q = new Tensor(new TensorShape(1, tokens, hidden), DType.F32);
        Tensor k = new Tensor(new TensorShape(1, tokens, hidden), DType.F32);
        Tensor v = new Tensor(new TensorShape(1, tokens, hidden), DType.F32);
        backend.Split([q, k, v], qkv, dim: 2);
        qkv.Dispose();

        Tensor attn = RtDetrAttention.MultiHead(backend, q, k, v, _config.NumHeads, _config.HeadDim);
        q.Dispose(); k.Dispose(); v.Dispose();
        Tensor attnProj = _aifiProj.Forward(backend, attn);
        attn.Dispose();

        // Residual + norm1.
        Tensor res1 = new Tensor(src.Shape, DType.F32);
        backend.Add(res1, src, attnProj);
        src.Dispose(); attnProj.Dispose();
        Tensor norm1 = _aifiNorm1.Forward(backend, res1);

        // FFN: fc1 -> GELU -> fc2, residual + norm2.
        Tensor fc1 = _aifiFc1.Forward(backend, norm1);
        backend.Gelu(fc1, fc1);
        Tensor fc2 = _aifiFc2.Forward(backend, fc1);
        fc1.Dispose();
        Tensor res2 = new Tensor(norm1.Shape, DType.F32);
        backend.Add(res2, norm1, fc2);
        norm1.Dispose(); fc2.Dispose(); res1.Dispose();
        Tensor norm2 = _aifiNorm2.Forward(backend, res2);
        res2.Dispose();

        // Un-flatten [1, HW, C] -> NCHW.
        Tensor outFeat = new Tensor(new TensorShape(1, hidden, height, width), DType.F32);
        backend.Transpose2D(outFeat, norm2, tokens, hidden);
        norm2.Dispose();
        return outFeat;
    }

    /// <summary>Builds the RT-DETR 2D sincos position embedding <c>[1, H·W, hidden]</c> (token order
    /// h-major, matching the flattened NCHW layout). Each quarter of the channel axis encodes
    /// sin(x)/cos(x)/sin(y)/cos(y) at geometric frequencies (temperature 10000).</summary>
    private static unsafe Tensor BuildSinCosPositionEmbedding(int height, int width, int hidden)
    {
        int tokens = height * width;
        int posDim = hidden / 4;
        Tensor pos = new Tensor(new TensorShape(1, tokens, hidden), DType.F32);
        float* p = (float*)pos.DataPointer;
        Span<float> omega = stackalloc float[posDim];
        for (int i = 0; i < posDim; i++)
            omega[i] = 1f / MathF.Pow(10000f, (float)i / posDim);

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                long baseOff = (long)(y * width + x) * hidden;
                for (int i = 0; i < posDim; i++)
                {
                    float wx = x * omega[i];
                    float wy = y * omega[i];
                    p[baseOff + i] = MathF.Sin(wx);
                    p[baseOff + posDim + i] = MathF.Cos(wx);
                    p[baseOff + 2 * posDim + i] = MathF.Sin(wy);
                    p[baseOff + 3 * posDim + i] = MathF.Cos(wy);
                }
            }
        }
        return pos;
    }

    private static Tensor Upsample2x(IBackend backend, Tensor input)
    {
        int c = (int)input.Shape[1];
        int h = (int)input.Shape[2];
        int w = (int)input.Shape[3];
        Tensor output = new Tensor(new TensorShape(1, c, h * 2, w * 2), DType.F32);
        backend.UpsampleNearest2D(output, input, 2, 2);
        return output;
    }

    private static Tensor ConcatChannel(IBackend backend, Tensor a, Tensor b)
    {
        int c = (int)(a.Shape[1] + b.Shape[1]);
        int h = (int)a.Shape[2];
        int w = (int)a.Shape[3];
        Tensor output = new Tensor(new TensorShape(1, c, h, w), DType.F32);
        backend.Concat(output, [a, b], dim: 1);
        return output;
    }
}
