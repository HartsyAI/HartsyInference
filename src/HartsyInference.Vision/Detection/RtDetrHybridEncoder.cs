using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Detection;

/// <summary>RT-DETR hybrid encoder (CCFM): an AIFI transformer block applied to the lowest-resolution
/// feature level, then a top-down FPN and bottom-up PAN built from <see cref="CspRepLayer"/> fusion
/// blocks, lateral 1×1 convs, and stride-2 downsample convs. Input is the three encoder-input-projected
/// feature maps (all 256-channel, strides 8/16/32); output is the three fused maps at the same shapes.</summary>
public sealed class RtDetrHybridEncoder
{
    private readonly RtDetrConfig _config;
    private readonly EncoderLayer _aifi;                 // encoder.encoder.0.layers.0
    private readonly RtDetrConv[] _lateralConvs;         // encoder.lateral_convs.{0,1}
    private readonly CspRepLayer[] _fpnBlocks;           // encoder.fpn_blocks.{0,1}
    private readonly RtDetrConv[] _downsampleConvs;      // encoder.downsample_convs.{0,1}
    private readonly CspRepLayer[] _panBlocks;           // encoder.pan_blocks.{0,1}

    /// <summary>Builds the hybrid encoder from a config.</summary>
    public RtDetrHybridEncoder(RtDetrConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);
        _config = config;
        _aifi = new EncoderLayer(config);
        _lateralConvs = [new RtDetrConv(1, 1, 0, 0, RtDetrActivation.Silu), new RtDetrConv(1, 1, 0, 0, RtDetrActivation.Silu)];
        _fpnBlocks = [new CspRepLayer(config), new CspRepLayer(config)];
        _downsampleConvs = [new RtDetrConv(2, 2, 1, 1, RtDetrActivation.Silu), new RtDetrConv(2, 2, 1, 1, RtDetrActivation.Silu)];
        _panBlocks = [new CspRepLayer(config), new CspRepLayer(config)];
    }

    /// <summary>Loads encoder weights. <paramref name="prefix"/> is <c>model.encoder</c>.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _aifi.LoadWeights(weights, $"{prefix}.encoder.0.layers.0");
        for (int i = 0; i < 2; i++)
        {
            _lateralConvs[i].LoadWeights(weights, $"{prefix}.lateral_convs.{i}.conv");
            _fpnBlocks[i].LoadWeights(weights, $"{prefix}.fpn_blocks.{i}");
            _downsampleConvs[i].LoadWeights(weights, $"{prefix}.downsample_convs.{i}.conv");
            _panBlocks[i].LoadWeights(weights, $"{prefix}.pan_blocks.{i}");
        }
    }

    /// <summary>Yields every encoder weight for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _aifi.EnumerateWeights()) yield return t;
        for (int i = 0; i < 2; i++)
        {
            foreach (Tensor t in _lateralConvs[i].EnumerateWeights()) yield return t;
            foreach (Tensor t in _fpnBlocks[i].EnumerateWeights()) yield return t;
            foreach (Tensor t in _downsampleConvs[i].EnumerateWeights()) yield return t;
            foreach (Tensor t in _panBlocks[i].EnumerateWeights()) yield return t;
        }
    }

    /// <summary>Runs the hybrid encoder over the three projected feature maps (all owned by the caller,
    /// not consumed). Returns the three fused PAN maps (strides 8/16/32). Caller owns all three.</summary>
    public (Tensor P0, Tensor P1, Tensor P2) Forward(IBackend backend, Tensor f0, Tensor f1, Tensor f2)
    {
        // AIFI on the top level (index 2).
        Tensor aifi2 = _aifi.Forward(backend, f2);

        // Top-down FPN.  fpn = [aifi2]; lateral+upsample+concat+fpn_block, then reverse.
        Tensor lat0 = _lateralConvs[0].Forward(backend, aifi2);
        aifi2.Dispose();
        Tensor up0 = NearestUpsample2x(lat0);
        Tensor cat0 = ConcatChannels(backend, up0, f1);
        up0.Dispose();
        Tensor p1Fpn = _fpnBlocks[0].Forward(backend, cat0);   // 40x40
        cat0.Dispose();

        Tensor lat1 = _lateralConvs[1].Forward(backend, p1Fpn);
        p1Fpn.Dispose();
        Tensor up1 = NearestUpsample2x(lat1);
        Tensor cat1 = ConcatChannels(backend, up1, f0);
        up1.Dispose();
        Tensor p0Fpn = _fpnBlocks[1].Forward(backend, cat1);   // 80x80
        cat1.Dispose();

        // fpn_feature_maps reversed = [p0Fpn(80), lat1(40), lat0(20)].
        // Bottom-up PAN.  pan = [p0Fpn]; downsample+concat+pan_block.
        Tensor out0 = p0Fpn;   // stride 8

        Tensor down0 = _downsampleConvs[0].Forward(backend, out0);
        Tensor pcat0 = ConcatChannels(backend, down0, lat1);
        down0.Dispose(); lat1.Dispose();
        Tensor out1 = _panBlocks[0].Forward(backend, pcat0);   // stride 16
        pcat0.Dispose();

        Tensor down1 = _downsampleConvs[1].Forward(backend, out1);
        Tensor pcat1 = ConcatChannels(backend, down1, lat0);
        down1.Dispose(); lat0.Dispose();
        Tensor out2 = _panBlocks[1].Forward(backend, pcat1);   // stride 32
        pcat1.Dispose();

        return (out0, out1, out2);
    }

    /// <summary>Nearest-neighbour ×2 upsample of a <c>[1,C,H,W]</c> map. Owns the returned tensor.</summary>
    internal static unsafe Tensor NearestUpsample2x(Tensor input)
    {
        int n = (int)input.Shape[0], c = (int)input.Shape[1], h = (int)input.Shape[2], w = (int)input.Shape[3];
        int oH = h * 2, oW = w * 2;
        Tensor output = new Tensor(new TensorShape(n, c, oH, oW), DType.F32);
        float* ip = (float*)input.DataPointer, op = (float*)output.DataPointer;
        for (long plane = 0; plane < (long)n * c; plane++)
        {
            float* src = ip + plane * h * w;
            float* dst = op + plane * oH * oW;
            for (int oy = 0; oy < oH; oy++)
                for (int ox = 0; ox < oW; ox++)
                    dst[oy * oW + ox] = src[(oy / 2) * w + (ox / 2)];
        }
        return output;
    }

    /// <summary>Concatenates two <c>[1,C,H,W]</c> maps along the channel axis. Owns the returned tensor.</summary>
    internal static Tensor ConcatChannels(IBackend backend, Tensor a, Tensor b)
    {
        int ca = (int)a.Shape[1], cb = (int)b.Shape[1];
        int h = (int)a.Shape[2], w = (int)a.Shape[3];
        Tensor output = new Tensor(new TensorShape(1, ca + cb, h, w), DType.F32);
        backend.Concat(output, [a, b], dim: 1);
        return output;
    }

    /// <summary>RepVGG block: parallel 3×3 and 1×1 convs summed, then SiLU.</summary>
    private sealed class RepVgg
    {
        private readonly RtDetrConv _conv1;   // 3x3
        private readonly RtDetrConv _conv2;   // 1x1

        public RepVgg()
        {
            _conv1 = new RtDetrConv(1, 1, 1, 1, RtDetrActivation.None);
            _conv2 = new RtDetrConv(1, 1, 0, 0, RtDetrActivation.None);
        }

        public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
        {
            _conv1.LoadWeights(weights, $"{prefix}.conv1.conv");
            _conv2.LoadWeights(weights, $"{prefix}.conv2.conv");
        }

        public IEnumerable<Tensor> EnumerateWeights()
        {
            foreach (Tensor t in _conv1.EnumerateWeights()) yield return t;
            foreach (Tensor t in _conv2.EnumerateWeights()) yield return t;
        }

        public Tensor Forward(IBackend backend, Tensor x)
        {
            Tensor a = _conv1.Forward(backend, x);
            Tensor b = _conv2.Forward(backend, x);
            Span<float> sa = a.AsSpan<float>();
            ReadOnlySpan<float> sb = b.AsSpan<float>();
            for (int i = 0; i < sa.Length; i++)
            {
                float v = sa[i] + sb[i];
                sa[i] = v / (1f + MathF.Exp(-v));   // SiLU
            }
            b.Dispose();
            return a;
        }
    }

    /// <summary>CSP RepVGG fusion layer: two 1×1 entry convs, three RepVGG bottlenecks on one branch,
    /// their sum, then a 1×1 exit conv (all SiLU).</summary>
    private sealed class CspRepLayer
    {
        private readonly RtDetrConv _conv1;
        private readonly RtDetrConv _conv2;
        private readonly RepVgg[] _bottlenecks;
        private readonly RtDetrConv _conv3;

        public CspRepLayer(RtDetrConfig config)
        {
            _conv1 = new RtDetrConv(1, 1, 0, 0, RtDetrActivation.Silu);
            _conv2 = new RtDetrConv(1, 1, 0, 0, RtDetrActivation.Silu);
            _bottlenecks = [new RepVgg(), new RepVgg(), new RepVgg()];
            _conv3 = new RtDetrConv(1, 1, 0, 0, RtDetrActivation.Silu);
        }

        public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
        {
            _conv1.LoadWeights(weights, $"{prefix}.conv1.conv");
            _conv2.LoadWeights(weights, $"{prefix}.conv2.conv");
            for (int i = 0; i < _bottlenecks.Length; i++)
                _bottlenecks[i].LoadWeights(weights, $"{prefix}.bottlenecks.{i}");
            _conv3.LoadWeights(weights, $"{prefix}.conv3.conv");
        }

        public IEnumerable<Tensor> EnumerateWeights()
        {
            foreach (Tensor t in _conv1.EnumerateWeights()) yield return t;
            foreach (Tensor t in _conv2.EnumerateWeights()) yield return t;
            foreach (RepVgg b in _bottlenecks)
                foreach (Tensor t in b.EnumerateWeights()) yield return t;
            foreach (Tensor t in _conv3.EnumerateWeights()) yield return t;
        }

        public Tensor Forward(IBackend backend, Tensor x)
        {
            Tensor h1 = _conv1.Forward(backend, x);
            foreach (RepVgg b in _bottlenecks)
            {
                Tensor next = b.Forward(backend, h1);
                h1.Dispose();
                h1 = next;
            }
            Tensor h2 = _conv2.Forward(backend, x);
            Span<float> s1 = h1.AsSpan<float>();
            ReadOnlySpan<float> s2 = h2.AsSpan<float>();
            for (int i = 0; i < s1.Length; i++)
                s1[i] += s2[i];
            h2.Dispose();
            Tensor outp = _conv3.Forward(backend, h1);
            h1.Dispose();
            return outp;
        }
    }

    /// <summary>The AIFI transformer encoder layer (post-norm): self-attention with the 2D sincos
    /// position embedding added to queries and keys, then a GELU feed-forward, each with a residual +
    /// LayerNorm.</summary>
    private sealed class EncoderLayer
    {
        private readonly RtDetrConfig _config;
        private readonly RtDetrLinear _qProj, _kProj, _vProj, _outProj;
        private readonly RtDetrLayerNorm _selfAttnLayerNorm;
        private readonly RtDetrLinear _fc1, _fc2;
        private readonly RtDetrLayerNorm _finalLayerNorm;

        public EncoderLayer(RtDetrConfig config)
        {
            _config = config;
            int d = config.HiddenDim;
            _qProj = new RtDetrLinear(d);
            _kProj = new RtDetrLinear(d);
            _vProj = new RtDetrLinear(d);
            _outProj = new RtDetrLinear(d);
            _selfAttnLayerNorm = new RtDetrLayerNorm(d, config.LayerNormEps);
            _fc1 = new RtDetrLinear(config.FeedForwardDim);
            _fc2 = new RtDetrLinear(d);
            _finalLayerNorm = new RtDetrLayerNorm(d, config.LayerNormEps);
        }

        public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
        {
            _qProj.LoadWeights(weights, $"{prefix}.self_attn.q_proj");
            _kProj.LoadWeights(weights, $"{prefix}.self_attn.k_proj");
            _vProj.LoadWeights(weights, $"{prefix}.self_attn.v_proj");
            _outProj.LoadWeights(weights, $"{prefix}.self_attn.out_proj");
            _selfAttnLayerNorm.LoadWeights(weights, $"{prefix}.self_attn_layer_norm");
            _fc1.LoadWeights(weights, $"{prefix}.fc1");
            _fc2.LoadWeights(weights, $"{prefix}.fc2");
            _finalLayerNorm.LoadWeights(weights, $"{prefix}.final_layer_norm");
        }

        public IEnumerable<Tensor> EnumerateWeights()
        {
            foreach (Tensor t in _qProj.EnumerateWeights()) yield return t;
            foreach (Tensor t in _kProj.EnumerateWeights()) yield return t;
            foreach (Tensor t in _vProj.EnumerateWeights()) yield return t;
            foreach (Tensor t in _outProj.EnumerateWeights()) yield return t;
            foreach (Tensor t in _selfAttnLayerNorm.EnumerateWeights()) yield return t;
            foreach (Tensor t in _fc1.EnumerateWeights()) yield return t;
            foreach (Tensor t in _fc2.EnumerateWeights()) yield return t;
            foreach (Tensor t in _finalLayerNorm.EnumerateWeights()) yield return t;
        }

        /// <summary>Runs AIFI over a <c>[1,C,H,W]</c> feature map: flatten to tokens, self-attend with the
        /// sincos pos-embed, feed-forward, then reshape back to <c>[1,C,H,W]</c>. Owns the returned tensor;
        /// does not consume <paramref name="featureMap"/>.</summary>
        public Tensor Forward(IBackend backend, Tensor featureMap)
        {
            int c = (int)featureMap.Shape[1], h = (int)featureMap.Shape[2], w = (int)featureMap.Shape[3];
            int s = h * w;
            Tensor x = FlattenToTokens(featureMap);                                   // [1, S, C]
            Tensor pos = RtDetrPositionEmbedding.Build2dSincos(h, w, c, _config.PositionalEncodingTemperature);  // [1, S, C]

            Tensor qkIn = AddTensors(x, pos);
            Tensor q = _qProj.Forward(backend, qkIn);
            Tensor k = _kProj.Forward(backend, qkIn);
            qkIn.Dispose();
            Tensor v = _vProj.Forward(backend, x);
            Tensor attn = RtDetrAttention.MultiHead(backend, q, k, v, _config.NumHeads, _config.HeadDim);
            q.Dispose(); k.Dispose(); v.Dispose();
            Tensor attnOut = _outProj.Forward(backend, attn);
            attn.Dispose();

            AddInto(x, attnOut);        // residual
            attnOut.Dispose();
            Tensor normed = _selfAttnLayerNorm.Forward(backend, x);
            x.Dispose();
            pos.Dispose();

            Tensor ff1 = _fc1.Forward(backend, normed);
            Tensor act = new Tensor(ff1.Shape, DType.F32);
            backend.GeluErf(act, ff1);
            ff1.Dispose();
            Tensor ff2 = _fc2.Forward(backend, act);
            act.Dispose();
            AddInto(normed, ff2);       // residual
            ff2.Dispose();
            Tensor outTokens = _finalLayerNorm.Forward(backend, normed);
            normed.Dispose();

            Tensor result = TokensToFeatureMap(outTokens, c, h, w);
            outTokens.Dispose();
            return result;
        }

        private static unsafe Tensor FlattenToTokens(Tensor featureMap)
        {
            int c = (int)featureMap.Shape[1], h = (int)featureMap.Shape[2], w = (int)featureMap.Shape[3];
            int s = h * w;
            Tensor tokens = new Tensor(new TensorShape(1, s, c), DType.F32);
            float* src = (float*)featureMap.DataPointer, dst = (float*)tokens.DataPointer;
            for (int ci = 0; ci < c; ci++)
                for (int p = 0; p < s; p++)
                    dst[(long)p * c + ci] = src[(long)ci * s + p];
            return tokens;
        }

        private static unsafe Tensor TokensToFeatureMap(Tensor tokens, int c, int h, int w)
        {
            int s = h * w;
            Tensor fm = new Tensor(new TensorShape(1, c, h, w), DType.F32);
            float* src = (float*)tokens.DataPointer, dst = (float*)fm.DataPointer;
            for (int p = 0; p < s; p++)
                for (int ci = 0; ci < c; ci++)
                    dst[(long)ci * s + p] = src[(long)p * c + ci];
            return fm;
        }

        private static Tensor AddTensors(Tensor a, Tensor b)
        {
            Tensor outp = new Tensor(a.Shape, DType.F32);
            Span<float> o = outp.AsSpan<float>();
            ReadOnlySpan<float> sa = a.AsSpan<float>(), sb = b.AsSpan<float>();
            for (int i = 0; i < o.Length; i++)
                o[i] = sa[i] + sb[i];
            return outp;
        }

        private static void AddInto(Tensor acc, Tensor add)
        {
            Span<float> a = acc.AsSpan<float>();
            ReadOnlySpan<float> b = add.AsSpan<float>();
            for (int i = 0; i < a.Length; i++)
                a[i] += b[i];
        }
    }
}
