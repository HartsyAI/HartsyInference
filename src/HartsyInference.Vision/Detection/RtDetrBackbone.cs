using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Detection;

/// <summary>RT-DETR ResNet-vd CNN backbone (the <c>PekingU/rtdetr_r18vd</c> ResNet-18vd variant): a
/// deep 3-conv stem + max-pool, then four residual stages of <see cref="RtDetrConfig.BackboneDepths"/>
/// basic blocks each. The "vd" downsample replaces the stride-2 1×1 shortcut with an avg-pool + 1×1
/// conv. Every conv has its BatchNorm folded in (see the checkpoint converter). Returns the last three
/// stage feature maps (strides 8/16/32, channels 128/256/512).</summary>
public sealed class RtDetrBackbone
{
    private readonly RtDetrConfig _config;
    private readonly RtDetrConv[] _stem;        // deep stem: 3 conv-bn-relu
    private readonly BasicLayer[][] _stages;    // [stage][layer]

    /// <summary>Builds the backbone from a config.</summary>
    public RtDetrBackbone(RtDetrConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);
        _config = config;
        _stem =
        [
            new RtDetrConv(2, 2, 1, 1, RtDetrActivation.Relu),   // num_channels -> emb/2, /2
            new RtDetrConv(1, 1, 1, 1, RtDetrActivation.Relu),   // emb/2 -> emb/2
            new RtDetrConv(1, 1, 1, 1, RtDetrActivation.Relu),   // emb/2 -> emb
        ];

        _stages = new BasicLayer[4][];
        for (int s = 0; s < 4; s++)
        {
            int depth = config.BackboneDepths[s];
            int stride = s == 0 ? 1 : 2;   // stage 0 keeps resolution; stages 1-3 downsample ×2
            _stages[s] = new BasicLayer[depth];
            for (int l = 0; l < depth; l++)
                _stages[s][l] = new BasicLayer(firstLayer: l == 0, stride: l == 0 ? stride : 1);
        }
    }

    /// <summary>Loads backbone weights. <paramref name="prefix"/> is <c>model.backbone.model</c>.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        for (int i = 0; i < 3; i++)
            _stem[i].LoadWeights(weights, $"{prefix}.embedder.embedder.{i}.convolution");
        for (int s = 0; s < _stages.Length; s++)
            for (int l = 0; l < _stages[s].Length; l++)
                _stages[s][l].LoadWeights(weights, $"{prefix}.encoder.stages.{s}.layers.{l}");
    }

    /// <summary>Yields every backbone weight for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (RtDetrConv c in _stem)
            foreach (Tensor t in c.EnumerateWeights()) yield return t;
        foreach (BasicLayer[] stage in _stages)
            foreach (BasicLayer layer in stage)
                foreach (Tensor t in layer.EnumerateWeights()) yield return t;
    }

    /// <summary>Runs the backbone; returns the three feature maps (C3 @ /8, C4 @ /16, C5 @ /32). Caller owns all three.</summary>
    public (Tensor C3, Tensor C4, Tensor C5) ForwardFeatures(IBackend backend, Tensor input)
    {
        Tensor x = _stem[0].Forward(backend, input);
        for (int i = 1; i < 3; i++)
        {
            Tensor next = _stem[i].Forward(backend, x);
            x.Dispose();
            x = next;
        }
        Tensor pooled = MaxPool3x3Stride2(x);
        x.Dispose();
        x = pooled;

        Tensor c3 = null!, c4 = null!, c5 = null!;
        for (int s = 0; s < _stages.Length; s++)
        {
            for (int l = 0; l < _stages[s].Length; l++)
            {
                Tensor next = _stages[s][l].Forward(backend, x);
                x.Dispose();
                x = next;
            }
            if (s == 1) c3 = Clone(x);
            else if (s == 2) c4 = Clone(x);
            else if (s == 3) c5 = x;
        }
        return (c3, c4, c5);
    }

    private static Tensor Clone(Tensor t)
    {
        Tensor c = new Tensor(t.Shape, DType.F32);
        t.AsSpan<float>().CopyTo(c.AsSpan<float>());
        return c;
    }

    /// <summary>Max pool, kernel 3, stride 2, padding 1 (the stem pooler), zero-free (out-of-bounds excluded).</summary>
    private static unsafe Tensor MaxPool3x3Stride2(Tensor input)
    {
        int n = (int)input.Shape[0], c = (int)input.Shape[1], h = (int)input.Shape[2], w = (int)input.Shape[3];
        int oH = (h + 2 - 3) / 2 + 1, oW = (w + 2 - 3) / 2 + 1;
        Tensor output = new Tensor(new TensorShape(n, c, oH, oW), DType.F32);
        float* ip = (float*)input.DataPointer, op = (float*)output.DataPointer;
        for (int ni = 0; ni < n; ni++)
            for (int ci = 0; ci < c; ci++)
            {
                float* plane = ip + ((long)ni * c + ci) * h * w;
                float* outPlane = op + ((long)ni * c + ci) * oH * oW;
                for (int oy = 0; oy < oH; oy++)
                    for (int ox = 0; ox < oW; ox++)
                    {
                        float best = float.NegativeInfinity;
                        for (int ky = 0; ky < 3; ky++)
                        {
                            int iy = oy * 2 - 1 + ky;
                            if (iy < 0 || iy >= h) continue;
                            for (int kx = 0; kx < 3; kx++)
                            {
                                int ix = ox * 2 - 1 + kx;
                                if (ix < 0 || ix >= w) continue;
                                float v = plane[iy * w + ix];
                                if (v > best) best = v;
                            }
                        }
                        outPlane[oy * oW + ox] = best;
                    }
            }
        return output;
    }

    /// <summary>Avg pool, kernel 2, stride 2, padding 0, ceil-mode (the "vd" shortcut downsampler).</summary>
    internal static unsafe Tensor AvgPool2x2Stride2(Tensor input)
    {
        int n = (int)input.Shape[0], c = (int)input.Shape[1], h = (int)input.Shape[2], w = (int)input.Shape[3];
        int oH = (int)Math.Ceiling((h - 2) / 2.0) + 1, oW = (int)Math.Ceiling((w - 2) / 2.0) + 1;
        Tensor output = new Tensor(new TensorShape(n, c, oH, oW), DType.F32);
        float* ip = (float*)input.DataPointer, op = (float*)output.DataPointer;
        for (int ni = 0; ni < n; ni++)
            for (int ci = 0; ci < c; ci++)
            {
                float* plane = ip + ((long)ni * c + ci) * h * w;
                float* outPlane = op + ((long)ni * c + ci) * oH * oW;
                for (int oy = 0; oy < oH; oy++)
                    for (int ox = 0; ox < oW; ox++)
                    {
                        float sum = 0f;
                        int cnt = 0;
                        for (int ky = 0; ky < 2; ky++)
                        {
                            int iy = oy * 2 + ky;
                            if (iy >= h) continue;
                            for (int kx = 0; kx < 2; kx++)
                            {
                                int ix = ox * 2 + kx;
                                if (ix >= w) continue;
                                sum += plane[iy * w + ix];
                                cnt++;
                            }
                        }
                        outPlane[oy * oW + ox] = cnt > 0 ? sum / cnt : 0f;
                    }
            }
        return output;
    }

    /// <summary>One ResNet-vd basic residual block: two 3×3 convs (ReLU then none), a residual add + ReLU,
    /// and — for the first block of a stage — a projection shortcut (plain 1×1 conv, or avg-pool + 1×1 conv
    /// when the stage downsamples).</summary>
    private sealed class BasicLayer
    {
        private readonly bool _firstLayer;
        private readonly bool _downsample;   // stride == 2 → vd avg-pool shortcut
        private readonly RtDetrConv _conv0;
        private readonly RtDetrConv _conv1;
        private RtDetrConv? _shortcut;       // only when _firstLayer

        public BasicLayer(bool firstLayer, int stride)
        {
            _firstLayer = firstLayer;
            _downsample = stride == 2;
            _conv0 = new RtDetrConv(stride, stride, 1, 1, RtDetrActivation.Relu);
            _conv1 = new RtDetrConv(1, 1, 1, 1, RtDetrActivation.None);
            if (firstLayer)
                _shortcut = new RtDetrConv(1, 1, 0, 0, RtDetrActivation.None);
        }

        public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
        {
            _conv0.LoadWeights(weights, $"{prefix}.layer.0.convolution");
            _conv1.LoadWeights(weights, $"{prefix}.layer.1.convolution");
            if (_firstLayer)
            {
                // Stage 0 (stride 1): plain shortcut "shortcut.convolution"; stages 1-3 (stride 2):
                // avg-pool + conv "shortcut.1.convolution".
                string scKey = _downsample ? $"{prefix}.shortcut.1.convolution" : $"{prefix}.shortcut.convolution";
                _shortcut!.LoadWeights(weights, scKey);
            }
        }

        public IEnumerable<Tensor> EnumerateWeights()
        {
            foreach (Tensor t in _conv0.EnumerateWeights()) yield return t;
            foreach (Tensor t in _conv1.EnumerateWeights()) yield return t;
            if (_shortcut is not null)
                foreach (Tensor t in _shortcut.EnumerateWeights()) yield return t;
        }

        public unsafe Tensor Forward(IBackend backend, Tensor input)
        {
            Tensor h0 = _conv0.Forward(backend, input);
            Tensor h1 = _conv1.Forward(backend, h0);
            h0.Dispose();

            Tensor residual;
            if (_shortcut is not null)
            {
                if (_downsample)
                {
                    Tensor pooled = AvgPool2x2Stride2(input);
                    residual = _shortcut.Forward(backend, pooled);
                    pooled.Dispose();
                }
                else
                {
                    residual = _shortcut.Forward(backend, input);
                }
            }
            else
            {
                residual = input;
            }

            Span<float> hs = h1.AsSpan<float>();
            ReadOnlySpan<float> rs = residual.AsSpan<float>();
            for (int i = 0; i < hs.Length; i++)
            {
                float v = hs[i] + rs[i];
                hs[i] = v < 0f ? 0f : v;
            }
            if (_shortcut is not null)
                residual.Dispose();
            return h1;
        }
    }
}
