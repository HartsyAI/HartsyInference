using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.ControlFoley;

/// <summary>Synchformer's visual feature extractor (<c>lib/synchformer</c>, <c>vfeat_extractor.*</c> keys) for inference:
/// a divided space-time ViT over 16-frame segments followed by a one-layer spatial transformer encoder that pools each
/// temporal token with a CLS token. <see cref="Encode"/> is <c>FeaturesUtils.encode_video_with_sync</c>.</summary>
public sealed unsafe class ControlFoleySynchformer : IDisposable
{
    private const string Prefix = "vfeat_extractor.";

    private readonly ControlFoleySynchformerConfig _config;
    private readonly List<Tensor> _owned = [];
    private Block[]? _blocks;
    private Tensor? _patchW, _patchB, _cls, _pos, _temp, _normW, _normB;
    private Agg? _agg;

    private sealed record Attn(Tensor QkvW, Tensor QkvB, Tensor ProjW, Tensor ProjB);

    private sealed record Block(Tensor Norm1W, Tensor Norm1B, Tensor Norm2W, Tensor Norm2B, Tensor Norm3W, Tensor Norm3B, Attn Space,
        Attn Time, Tensor Fc1W, Tensor Fc1B, Tensor Fc2W, Tensor Fc2B);

    private sealed record Agg(Tensor Cls, Tensor InW, Tensor InB, Tensor OutW, Tensor OutB, Tensor Norm1W, Tensor Norm1B, Tensor Norm2W,
        Tensor Norm2B, Tensor L1W, Tensor L1B, Tensor L2W, Tensor L2B);

    /// <summary>Creates an unloaded extractor; call <see cref="LoadWeights"/> before encoding.</summary>
    public ControlFoleySynchformer(ControlFoleySynchformerConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);
        if (config.EmbedDim % config.Heads != 0 || config.InputSize % config.PatchSize != 0 || config.SegmentFrames % config.TemporalPatch != 0)
        {
            throw new ArgumentException("Invalid Synchformer configuration.", nameof(config));
        }

        _config = config;
    }

    /// <summary>Geometry this instance was built for.</summary>
    public ControlFoleySynchformerConfig Config => _config;

    /// <summary>Binds <c>synchformer_state_dict.pth</c> (keys under <c>vfeat_extractor.</c>; other keys are ignored).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        ArgumentNullException.ThrowIfNull(weights);
        ControlFoleySynchformerConfig c = _config;
        int d = c.EmbedDim, hidden = d * c.MlpRatio, p = c.PatchSize, tp = c.TemporalPatch;
        Tensor conv = ControlFoleyVitOps.Load(weights, Prefix + "patch_embed_3d.proj.weight", _owned, d, 3, tp, p, p);
        _patchW = conv.Reshape(new TensorShape(d, 3 * tp * p * p));
        _owned.Add(_patchW);
        _patchB = L(weights, "patch_embed_3d.proj.bias", d);
        _cls = L(weights, "cls_token", 1, 1, d);
        _pos = L(weights, "pos_embed", 1, c.SpatialTokens + 1, d);
        _temp = L(weights, "temp_embed", 1, c.TemporalTokens, d);
        _normW = L(weights, "norm.weight", d);
        _normB = L(weights, "norm.bias", d);
        _blocks = new Block[c.Depth];
        for (int i = 0; i < c.Depth; i++)
        {
            string b = $"blocks.{i}";
            _blocks[i] = new Block(
                L(weights, $"{b}.norm1.weight", d), L(weights, $"{b}.norm1.bias", d),
                L(weights, $"{b}.norm2.weight", d), L(weights, $"{b}.norm2.bias", d),
                L(weights, $"{b}.norm3.weight", d), L(weights, $"{b}.norm3.bias", d),
                new Attn(L(weights, $"{b}.attn.qkv.weight", 3 * d, d), L(weights, $"{b}.attn.qkv.bias", 3 * d),
                    L(weights, $"{b}.attn.proj.weight", d, d), L(weights, $"{b}.attn.proj.bias", d)),
                new Attn(L(weights, $"{b}.timeattn.qkv.weight", 3 * d, d), L(weights, $"{b}.timeattn.qkv.bias", 3 * d),
                    L(weights, $"{b}.timeattn.proj.weight", d, d), L(weights, $"{b}.timeattn.proj.bias", d)),
                L(weights, $"{b}.mlp.fc1.weight", hidden, d), L(weights, $"{b}.mlp.fc1.bias", hidden),
                L(weights, $"{b}.mlp.fc2.weight", d, hidden), L(weights, $"{b}.mlp.fc2.bias", d));
        }

        const string a = "spatial_attn_agg.";
        _agg = new Agg(L(weights, a + "cls_token", 1, 1, d), L(weights, a + "self_attn.in_proj_weight", 3 * d, d),
            L(weights, a + "self_attn.in_proj_bias", 3 * d), L(weights, a + "self_attn.out_proj.weight", d, d),
            L(weights, a + "self_attn.out_proj.bias", d), L(weights, a + "norm1.weight", d), L(weights, a + "norm1.bias", d),
            L(weights, a + "norm2.weight", d), L(weights, a + "norm2.bias", d), L(weights, a + "linear1.weight", hidden, d),
            L(weights, a + "linear1.bias", hidden), L(weights, a + "linear2.weight", d, hidden), L(weights, a + "linear2.bias", d));
    }

    private Tensor L(IReadOnlyDictionary<string, Tensor> w, string key, params long[] shape)
        => ControlFoleyVitOps.Load(w, Prefix + key, _owned, shape);

    /// <summary>Number of segments <c>(T - SegmentFrames) / StrideFrames + 1</c> for <paramref name="frameCount"/> frames.</summary>
    public int SegmentCount(int frameCount) => frameCount < _config.SegmentFrames ? 0 : (frameCount - _config.SegmentFrames) / _config.StrideFrames + 1;

    /// <summary>Encodes frames <c>[T, 3, S, S]</c> (S = <see cref="ControlFoleySynchformerConfig.InputSize"/>, normalised with
    /// mean/std 0.5) into <c>[segments * TemporalTokens, EmbedDim]</c> sync tokens.</summary>
    public float[] Encode(IBackend backend, float[] frames)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(frames);
        ControlFoleySynchformerConfig c = _config;
        long perFrame = 3L * c.InputSize * c.InputSize;
        if (frames.Length == 0 || frames.Length % perFrame != 0)
        {
            throw new ArgumentException($"Expected a multiple of 3x{c.InputSize}x{c.InputSize} floats, got {frames.Length}.", nameof(frames));
        }

        int count = (int)(frames.Length / perFrame), segments = SegmentCount(count);
        if (segments <= 0)
        {
            throw new ArgumentException($"{count} frames are fewer than one {c.SegmentFrames}-frame segment.", nameof(frames));
        }

        int rows = c.TemporalTokens * c.EmbedDim;
        float[] result = new float[segments * rows];
        for (int s = 0; s < segments; s++)
        {
            EncodeSegment(backend, frames, s * c.StrideFrames * perFrame).CopyTo(result, s * rows);
        }

        return result;
    }

    /// <summary>One segment starting at float offset <paramref name="offset"/>; returns <c>[TemporalTokens, EmbedDim]</c>.</summary>
    private float[] EncodeSegment(IBackend backend, float[] frames, long offset)
    {
        if (_blocks is null)
        {
            throw new InvalidOperationException("Call LoadWeights before encoding.");
        }

        ControlFoleySynchformerConfig c = _config;
        int d = c.EmbedDim, n = c.SpatialTokens, t = c.TemporalTokens, total = 1 + t * n;
        Tensor x = new(new TensorShape(1, total, d), DType.F32);
        using (Tensor patches = Patchify(frames, offset))
        using (Tensor embedded = ControlFoleyOps.Linear(backend, patches, _patchW!, _patchB))
        {
            float* xp = (float*)x.DataPointer, e = (float*)embedded.DataPointer;
            float* cls = (float*)_cls!.DataPointer, pos = (float*)_pos!.DataPointer, temp = (float*)_temp!.DataPointer;
            for (int k = 0; k < d; k++)
            {
                xp[k] = cls[k] + pos[k];
            }

            for (int ti = 0; ti < t; ti++)
            {
                for (int pi = 0; pi < n; pi++)
                {
                    long row = 1 + (long)ti * n + pi;
                    for (int k = 0; k < d; k++)
                    {
                        xp[row * d + k] = e[(row - 1) * d + k] + pos[(1 + pi) * d + k] + temp[ti * d + k];
                    }
                }
            }
        }

        foreach (Block b in _blocks)
        {
            RunBlock(backend, x, b);
        }

        using (x)
        {
            return Aggregate(backend, x);
        }
    }

    private Tensor Patchify(float[] frames, long offset)
    {
        ControlFoleySynchformerConfig c = _config;
        int size = c.InputSize, p = c.PatchSize, tp = c.TemporalPatch, grid = c.GridSize, plane = size * size;
        int patchDim = 3 * tp * p * p;
        Tensor o = new(new TensorShape(1, c.TemporalTokens * grid * grid, patchDim), DType.F32);
        float* dst = (float*)o.DataPointer;
        for (int ti = 0; ti < c.TemporalTokens; ti++)
        {
            for (int gy = 0; gy < grid; gy++)
            {
                for (int gx = 0; gx < grid; gx++)
                {
                    float* row = dst + (((long)ti * grid + gy) * grid + gx) * patchDim;
                    for (int ch = 0; ch < 3; ch++)
                    {
                        for (int dt = 0; dt < tp; dt++)
                        {
                            long frameBase = offset + ((long)(ti * tp + dt) * 3 + ch) * plane;
                            for (int ky = 0; ky < p; ky++)
                            {
                                new ReadOnlySpan<float>(frames, (int)(frameBase + (long)(gy * p + ky) * size + gx * p), p)
                                    .CopyTo(new Span<float>(row + ((ch * tp + dt) * p + ky) * p, p));
                            }
                        }
                    }
                }
            }
        }

        return o;
    }

    private void RunBlock(IBackend backend, Tensor x, Block b)
    {
        float eps = _config.LayerNormEps;
        using (Tensor normed = ControlFoleyVitOps.LayerNorm(backend, x, b.Norm3W, b.Norm3B, eps))
        using (Tensor attn = DividedAttention(backend, normed, b.Time, time: true))
        {
            ControlFoleyVitOps.AddInPlace(x, attn);
        }

        using (Tensor normed = ControlFoleyVitOps.LayerNorm(backend, x, b.Norm1W, b.Norm1B, eps))
        using (Tensor attn = DividedAttention(backend, normed, b.Space, time: false))
        {
            ControlFoleyVitOps.AddInPlace(x, attn);
        }

        using (Tensor normed = ControlFoleyVitOps.LayerNorm(backend, x, b.Norm2W, b.Norm2B, eps))
        using (Tensor ff = ControlFoleyVitOps.Mlp(backend, normed, b.Fc1W, b.Fc1B, b.Fc2W, b.Fc2B))
        {
            ControlFoleyVitOps.AddInPlace(x, ff);
        }
    }

    /// <summary>Port of <c>DividedAttention</c>: the CLS token attends to every token; patch tokens attend to the CLS key/value
    /// plus the tokens of their own frame (space) or their own spatial position (time).</summary>
    private Tensor DividedAttention(IBackend backend, Tensor x, Attn w, bool time)
    {
        ControlFoleySynchformerConfig c = _config;
        int heads = c.Heads, d = c.EmbedDim, hd = d / heads, n = c.SpatialTokens, frames = c.TemporalTokens, total = 1 + frames * n;
        int groups = time ? n : frames, len = time ? frames : n;
        float scale = 1f / MathF.Sqrt(hd);

        using Tensor qkv = ControlFoleyOps.Linear(backend, x, w.QkvW, w.QkvB);
        (Tensor qAll, Tensor kAll, Tensor vAll) = ControlFoleyVitOps.SplitQkv(qkv, heads);
        using Tensor qa = qAll, ka = kAll, va = vAll;

        using Tensor clsQ = new(new TensorShape(1, heads, 1, hd), DType.F32);
        using Tensor clsCtx = new(clsQ.Shape, DType.F32);
        float* qa0 = (float*)qa.DataPointer, ka0 = (float*)ka.DataPointer, va0 = (float*)va.DataPointer;
        for (int h = 0; h < heads; h++)
        {
            new ReadOnlySpan<float>(qa0 + (long)h * total * hd, hd).CopyTo(new Span<float>((float*)clsQ.DataPointer + h * hd, hd));
        }

        backend.ScaledDotProductAttention(clsCtx, clsQ, ka, va, null, scale);

        using Tensor q = new(new TensorShape(groups, heads, len, hd), DType.F32);
        using Tensor k = new(new TensorShape(groups, heads, len + 1, hd), DType.F32);
        using Tensor v = new(k.Shape, DType.F32);
        float* qp = (float*)q.DataPointer, kp = (float*)k.DataPointer, vp = (float*)v.DataPointer;
        for (int g = 0; g < groups; g++)
        {
            for (int h = 0; h < heads; h++)
            {
                long headBase = (long)h * total * hd;
                long kBase = ((long)g * heads + h) * (len + 1) * hd;
                new ReadOnlySpan<float>(ka0 + headBase, hd).CopyTo(new Span<float>(kp + kBase, hd));
                new ReadOnlySpan<float>(va0 + headBase, hd).CopyTo(new Span<float>(vp + kBase, hd));
                for (int i = 0; i < len; i++)
                {
                    long token = time ? 1 + (long)i * n + g : 1 + (long)g * n + i;
                    long src = headBase + token * hd;
                    new ReadOnlySpan<float>(qa0 + src, hd).CopyTo(new Span<float>(qp + (((long)g * heads + h) * len + i) * hd, hd));
                    new ReadOnlySpan<float>(ka0 + src, hd).CopyTo(new Span<float>(kp + kBase + (i + 1) * hd, hd));
                    new ReadOnlySpan<float>(va0 + src, hd).CopyTo(new Span<float>(vp + kBase + (i + 1) * hd, hd));
                }
            }
        }

        using Tensor ctx = new(q.Shape, DType.F32);
        backend.ScaledDotProductAttention(ctx, q, k, v, null, scale);

        using Tensor merged = new(new TensorShape(1, total, d), DType.F32);
        float* mp = (float*)merged.DataPointer, cp = (float*)ctx.DataPointer;
        for (int h = 0; h < heads; h++)
        {
            new ReadOnlySpan<float>((float*)clsCtx.DataPointer + h * hd, hd).CopyTo(new Span<float>(mp + h * hd, hd));
        }

        for (int g = 0; g < groups; g++)
        {
            for (int h = 0; h < heads; h++)
            {
                for (int i = 0; i < len; i++)
                {
                    long token = time ? 1 + (long)i * n + g : 1 + (long)g * n + i;
                    new ReadOnlySpan<float>(cp + (((long)g * heads + h) * len + i) * hd, hd)
                        .CopyTo(new Span<float>(mp + token * d + h * hd, hd));
                }
            }
        }

        return ControlFoleyOps.Linear(backend, merged, w.ProjW, w.ProjB);
    }

    /// <summary>Final norm on the patch tokens, then per temporal token the spatial encoder layer keeps its CLS row
    /// (only that query is computed; keys and values still cover every token).</summary>
    private float[] Aggregate(IBackend backend, Tensor x)
    {
        ControlFoleySynchformerConfig c = _config;
        Agg a = _agg!;
        int d = c.EmbedDim, n = c.SpatialTokens, t = c.TemporalTokens, heads = c.Heads, hd = d / heads, seq = n + 1;
        float eps = c.LayerNormEps;
        using Tensor seqIn = new(new TensorShape(t, seq, d), DType.F32);
        using (Tensor patches = new(new TensorShape(1, t * n, d), DType.F32))
        {
            new ReadOnlySpan<float>((float*)x.DataPointer + d, t * n * d).CopyTo(new Span<float>((float*)patches.DataPointer, t * n * d));
            using Tensor normed = ControlFoleyVitOps.LayerNorm(backend, patches, _normW!, _normB!, eps);
            float* sp = (float*)seqIn.DataPointer, np = (float*)normed.DataPointer, cls = (float*)a.Cls.DataPointer;
            for (int ti = 0; ti < t; ti++)
            {
                new ReadOnlySpan<float>(cls, d).CopyTo(new Span<float>(sp + (long)ti * seq * d, d));
                new ReadOnlySpan<float>(np + (long)ti * n * d, n * d).CopyTo(new Span<float>(sp + ((long)ti * seq + 1) * d, n * d));
            }
        }

        using Tensor normed1 = ControlFoleyVitOps.LayerNorm(backend, seqIn, a.Norm1W, a.Norm1B, eps);
        using Tensor kv = new(new TensorShape(t, seq, 2 * d), DType.F32);
        backend.LinearWeightRows(kv, normed1, a.InW, a.InB, d, 2 * d);
        using Tensor clsNormed = new(new TensorShape(t, 1, d), DType.F32);
        for (int ti = 0; ti < t; ti++)
        {
            new ReadOnlySpan<float>((float*)normed1.DataPointer + (long)ti * seq * d, d)
                .CopyTo(new Span<float>((float*)clsNormed.DataPointer + (long)ti * d, d));
        }

        using Tensor qProj = new(new TensorShape(t, 1, d), DType.F32);
        backend.LinearWeightRows(qProj, clsNormed, a.InW, a.InB, 0, d);

        TensorShape qShape = new(t, heads, 1, hd), kvShape = new(t, heads, seq, hd);
        using Tensor q = new(qShape, DType.F32);
        using Tensor k = new(kvShape, DType.F32);
        using Tensor v = new(kvShape, DType.F32);
        float* kvp = (float*)kv.DataPointer;
        for (int ti = 0; ti < t; ti++)
        {
            for (int h = 0; h < heads; h++)
            {
                new ReadOnlySpan<float>((float*)qProj.DataPointer + (long)ti * d + h * hd, hd)
                    .CopyTo(new Span<float>((float*)q.DataPointer + ((long)ti * heads + h) * hd, hd));
                for (int s = 0; s < seq; s++)
                {
                    float* row = kvp + ((long)ti * seq + s) * 2 * d;
                    long dst = (((long)ti * heads + h) * seq + s) * hd;
                    new ReadOnlySpan<float>(row + h * hd, hd).CopyTo(new Span<float>((float*)k.DataPointer + dst, hd));
                    new ReadOnlySpan<float>(row + d + h * hd, hd).CopyTo(new Span<float>((float*)v.DataPointer + dst, hd));
                }
            }
        }

        using Tensor ctx = new(qShape, DType.F32);
        backend.ScaledDotProductAttention(ctx, q, k, v, null, 1f / MathF.Sqrt(hd));
        using Tensor merged = ControlFoleyVitOps.MergeHeads(ctx);
        using Tensor attn = ControlFoleyOps.Linear(backend, merged, a.OutW, a.OutB);
        using Tensor y = new(new TensorShape(t, 1, d), DType.F32);
        float* yp = (float*)y.DataPointer, ap = (float*)attn.DataPointer, clsRaw = (float*)a.Cls.DataPointer;
        for (int ti = 0; ti < t; ti++)
        {
            for (int e = 0; e < d; e++)
            {
                yp[ti * d + e] = clsRaw[e] + ap[ti * d + e];
            }
        }

        using Tensor normed2 = ControlFoleyVitOps.LayerNorm(backend, y, a.Norm2W, a.Norm2B, eps);
        using Tensor ff = ControlFoleyVitOps.Mlp(backend, normed2, a.L1W, a.L1B, a.L2W, a.L2B);
        ControlFoleyVitOps.AddInPlace(y, ff);
        return new ReadOnlySpan<float>(yp, t * d).ToArray();
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        foreach (Tensor t in _owned)
        {
            t.Dispose();
        }

        _owned.Clear();
    }
}
