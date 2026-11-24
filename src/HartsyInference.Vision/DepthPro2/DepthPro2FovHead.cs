using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Vision.DinoVit;

namespace HartsyInference.Vision.DepthPro2;

/// <summary>
/// The focal estimator's head (upstream <c>FOVHead</c>): each of the four tapped token sets is LayerNormed, the four are concatenated and projected to
/// <c>dim</c>, the patch grid is average-pooled (2 x 2, <c>ceil_mode</c>) and given a 2D sin/cos code scaled by the longer side, and a learned query token,
/// the prefix tokens (plus a learned embedding) and the pooled patches go through two plain ViT blocks (fused qkv with bias, no LayerScale, LayerNorm eps
/// 1e-6). The query's final-normed state maps to two logits, horizontal and vertical field of view. F32; weights are borrowed from the caller.
/// </summary>
public sealed class DepthPro2FovHead
{
    private const float TapNormEps = 1e-5f;     // nn.LayerNorm's default, as the tap norms and the final norm use
    private const float BlockNormEps = 1e-6f;   // timm's Block default
    private const int PoolStride = 2;

    private readonly IReadOnlyDictionary<string, Tensor> _weights;
    private readonly string _prefix;
    private readonly int _embed, _taps, _numPrefix, _dim;
    private readonly DinoViTLayer[] _blocks;

    /// <param name="weights">The tensors by upstream name (<c>head.*</c> in <c>apple/DepthPro2-focal</c>), F32.</param>
    /// <param name="prefix">The head's prefix, <c>head.</c> in the released checkpoint.</param>
    /// <param name="embedDim">The trunk width.</param>
    /// <param name="numTaps">How many trunk blocks are tapped.</param>
    /// <param name="numPrefix">CLS plus storage tokens of the trunk.</param>
    /// <param name="dim">The head width.</param>
    /// <param name="depth">The number of ViT blocks.</param>
    /// <param name="heads">The attention heads of each block.</param>
    /// <exception cref="InvalidOperationException">A weight is missing, not F32 or of the wrong shape.</exception>
    public DepthPro2FovHead(IReadOnlyDictionary<string, Tensor> weights, string prefix = "head.", int embedDim = 384, int numTaps = 4, int numPrefix = 5,
        int dim = 248, int depth = 2, int heads = 4)
    {
        ArgumentNullException.ThrowIfNull(weights);
        ArgumentNullException.ThrowIfNull(prefix);
        if (embedDim <= 0 || numTaps <= 0 || numPrefix < 0 || dim <= 0 || depth <= 0 || heads <= 0 || dim % heads != 0 || dim % 4 != 0)
            throw new ArgumentException($"Invalid head geometry: embed {embedDim}, taps {numTaps}, prefix {numPrefix}, dim {dim}, depth {depth}, heads {heads}.");
        _weights = weights;
        _prefix = prefix;
        _embed = embedDim;
        _taps = numTaps;
        _numPrefix = numPrefix;
        _dim = dim;
        for (int t = 0; t < numTaps; t++)
        {
            W($"tap_norms.{t}.weight", embedDim);
            W($"tap_norms.{t}.bias", embedDim);
        }

        W("proj.weight", dim, embedDim * numTaps);
        W("proj.bias", dim);
        W("query", 1, 1, dim);
        W("prefix_embed", 1, numPrefix, dim);
        W("norm.weight", dim);
        W("norm.bias", dim);
        W("out.weight", 2, dim);
        W("out.bias", 2);
        _blocks = new DinoViTLayer[depth];
        for (int i = 0; i < depth; i++)
        {
            _blocks[i] = new DinoViTLayer(dim, heads, 4 * dim, BlockNormEps);
            _blocks[i].LoadWeights(BlockWeights(i), "b");
        }
    }

    /// <summary>The head's tensors.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (KeyValuePair<string, Tensor> entry in _weights)
            if (entry.Key.StartsWith(_prefix, StringComparison.Ordinal))
                yield return entry.Value;
    }

    /// <summary>
    /// The two field-of-view logits <c>[1, 2]</c> (horizontal, vertical) from the trunk taps concatenated on channels, <c>[1, numPrefix + gridH·gridW,
    /// numTaps·embedDim]</c>. <paramref name="observe"/> sees <c>fov_proj</c>, <c>fov_pooled</c>, <c>fov_sincos</c>, <c>fov_seq</c> and <c>fov_block{i}</c>.
    /// </summary>
    public Tensor Forward(IBackend backend, Tensor taps, int gridH, int gridW, Action<string, Tensor>? observe = null)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(taps);
        int tokens = _numPrefix + gridH * gridW, width = _taps * _embed;
        if (taps.DType != DType.F32 || taps.Shape.Rank != 3 || taps.Shape[0] != 1 || taps.Shape[1] != tokens || taps.Shape[2] != width)
            throw new ArgumentException($"taps must be F32 [1, {tokens}, {width}]; got {taps.DType} {taps.Shape}.", nameof(taps));

        using Tensor normed = NormTaps(backend, taps, tokens);
        using Tensor proj = new(new TensorShape(1, tokens, _dim), DType.F32);
        backend.Linear(proj, normed, W("proj.weight", _dim, width), W("proj.bias", _dim));
        observe?.Invoke("fov_proj", proj);

        int poolH = (gridH + PoolStride - 1) / PoolStride, poolW = (gridW + PoolStride - 1) / PoolStride, pooledCount = poolH * poolW;
        using Tensor pooled = new(new TensorShape(1, _dim, poolH, poolW), DType.F32);
        float[] channelsFirst = new float[_dim * gridH * gridW];
        ReadOnlySpan<float> projSpan = proj.AsReadOnlySpan<float>();
        for (int n = 0; n < gridH * gridW; n++)
            for (int c = 0; c < _dim; c++)
                channelsFirst[c * gridH * gridW + n] = projSpan[(_numPrefix + n) * _dim + c];
        AvgPoolCeil(channelsFirst, _dim, gridH, gridW, PoolStride, pooled.AsSpan<float>());
        observe?.Invoke("fov_pooled", pooled);

        float[] code = SinCos2d(poolH, poolW, _dim);
        if (observe is not null)
        {
            using Tensor codeTensor = new(new TensorShape(pooledCount, _dim), DType.F32);
            code.CopyTo(codeTensor.AsSpan<float>());
            observe("fov_sincos", codeTensor);
        }

        Tensor seq = new(new TensorShape(1, 1 + _numPrefix + pooledCount, _dim), DType.F32);
        try
        {
            Span<float> s = seq.AsSpan<float>();
            W("query", 1, 1, _dim).AsReadOnlySpan<float>().CopyTo(s);
            ReadOnlySpan<float> prefixEmbed = W("prefix_embed", 1, _numPrefix, _dim).AsReadOnlySpan<float>();
            for (int i = 0; i < _numPrefix * _dim; i++) s[_dim + i] = projSpan[i] + prefixEmbed[i];
            ReadOnlySpan<float> pooledSpan = pooled.AsReadOnlySpan<float>();
            int patchBase = (1 + _numPrefix) * _dim;
            for (int n = 0; n < pooledCount; n++)
                for (int c = 0; c < _dim; c++)
                    s[patchBase + n * _dim + c] = pooledSpan[c * pooledCount + n] + code[n * _dim + c];
            observe?.Invoke("fov_seq", seq);

            for (int i = 0; i < _blocks.Length; i++)
            {
                Tensor next = _blocks[i].Forward(backend, seq);
                seq.Dispose();
                seq = next;
                observe?.Invoke($"fov_block{i}", seq);
            }

            using Tensor query = new(new TensorShape(1, _dim), DType.F32);
            seq.AsReadOnlySpan<float>()[.._dim].CopyTo(query.AsSpan<float>());
            using Tensor queryNormed = new(query.Shape, DType.F32);
            backend.LayerNorm(queryNormed, query, W("norm.weight", _dim), W("norm.bias", _dim), TapNormEps);
            Tensor logits = new(new TensorShape(1, 2), DType.F32);
            backend.Linear(logits, queryNormed, W("out.weight", 2, _dim), W("out.bias", 2));
            return logits;
        }
        finally
        {
            seq.Dispose();
        }
    }

    /// <summary><c>F.avg_pool2d(x, stride, ceil_mode=True)</c> on <c>[channels, h, w]</c> (no padding): a window cut by the edge averages the cells it covers.</summary>
    public static void AvgPoolCeil(ReadOnlySpan<float> src, int channels, int h, int w, int stride, Span<float> dst)
    {
        if (channels <= 0 || h <= 0 || w <= 0 || stride <= 0) throw new ArgumentOutOfRangeException(nameof(stride), "Sizes and the stride must be positive.");
        int oh = (h + stride - 1) / stride, ow = (w + stride - 1) / stride;
        if (src.Length < channels * h * w || dst.Length < channels * oh * ow)
            throw new ArgumentException($"Buffers too small for [{channels}, {h}, {w}] pooled to [{channels}, {oh}, {ow}].");
        for (int c = 0; c < channels; c++)
            for (int y = 0; y < oh; y++)
                for (int x = 0; x < ow; x++)
                {
                    int y0 = y * stride, y1 = Math.Min(y0 + stride, h), x0 = x * stride, x1 = Math.Min(x0 + stride, w);
                    float sum = 0f;
                    for (int yy = y0; yy < y1; yy++)
                        for (int xx = x0; xx < x1; xx++)
                            sum += src[(c * h + yy) * w + xx];
                    dst[(c * oh + y) * ow + x] = sum / ((y1 - y0) * (x1 - x0));
                }
    }

    /// <summary>
    /// Upstream <c>sincos_2d</c>: <c>[gh·gw, dim]</c>, per cell <c>[sin(y·f), cos(y·f), sin(x·f), cos(x·f)]</c> over <c>dim/4</c> frequencies
    /// <c>f = π·2^linspace(0, 7)</c>, with centred coordinates divided by the longer side (so the code carries the aspect ratio). Float32, as torch computes it.
    /// </summary>
    public static float[] SinCos2d(int gh, int gw, int dim)
    {
        if (gh <= 0 || gw <= 0 || dim <= 0 || dim % 4 != 0) throw new ArgumentOutOfRangeException(nameof(dim), "Grid sides must be positive and dim a multiple of 4.");
        int n = dim / 4;
        float g = Math.Max(gh, gw);
        float[] freq = new float[n];
        float step = n > 1 ? (7f - 0f) / (n - 1) : 0f;
        for (int i = 0; i < n; i++)
        {
            // torch.linspace fills the first half from the start and the second half back from the end.
            float t = i < n / 2 ? 0f + step * i : 7f - step * (n - i - 1);
            freq[i] = MathF.PI * MathF.Pow(2f, t);
        }

        float[] code = new float[gh * gw * dim];
        for (int y = 0; y < gh; y++)
        {
            float cy = (y + 0.5f - gh / 2f) / g;
            for (int x = 0; x < gw; x++)
            {
                float cx = (x + 0.5f - gw / 2f) / g;
                int row = (y * gw + x) * dim;
                for (int i = 0; i < n; i++)
                {
                    float ay = cy * freq[i], ax = cx * freq[i];
                    code[row + i] = MathF.Sin(ay);
                    code[row + n + i] = MathF.Cos(ay);
                    code[row + 2 * n + i] = MathF.Sin(ax);
                    code[row + 3 * n + i] = MathF.Cos(ax);
                }
            }
        }

        return code;
    }

    // Each tap's channel slice LayerNormed with its own weights, written back side by side: [tokens, numTaps * embed].
    private Tensor NormTaps(IBackend backend, Tensor taps, int tokens)
    {
        int width = _taps * _embed;
        Tensor normed = new(new TensorShape(1, tokens, width), DType.F32);
        try
        {
            using Tensor slice = new(new TensorShape(1, tokens, _embed), DType.F32);
            using Tensor sliceNormed = new(slice.Shape, DType.F32);
            ReadOnlySpan<float> src = taps.AsReadOnlySpan<float>();
            Span<float> dst = normed.AsSpan<float>();
            for (int t = 0; t < _taps; t++)
            {
                Span<float> s = slice.AsSpan<float>();
                for (int r = 0; r < tokens; r++) src.Slice(r * width + t * _embed, _embed).CopyTo(s.Slice(r * _embed, _embed));
                backend.LayerNorm(sliceNormed, slice, W($"tap_norms.{t}.weight", _embed), W($"tap_norms.{t}.bias", _embed), TapNormEps);
                ReadOnlySpan<float> o = sliceNormed.AsReadOnlySpan<float>();
                for (int r = 0; r < tokens; r++) o.Slice(r * _embed, _embed).CopyTo(dst.Slice(r * width + t * _embed, _embed));
            }

            return normed;
        }
        catch
        {
            normed.Dispose();
            throw;
        }
    }

    // timm Block (norm1, attn.qkv fused with bias, attn.proj, norm2, mlp.fc1/fc2) under the HF ViT names DinoViTLayer reads; the qkv slices are views.
    private Dictionary<string, Tensor> BlockWeights(int index)
    {
        string p = $"blocks.{index}.";
        int d = _dim, inner = 4 * _dim;
        Tensor qkvW = W(p + "attn.qkv.weight", 3 * d, d), qkvB = W(p + "attn.qkv.bias", 3 * d);
        return new Dictionary<string, Tensor>(StringComparer.Ordinal)
        {
            ["b.layernorm_before.weight"] = W(p + "norm1.weight", d),
            ["b.layernorm_before.bias"] = W(p + "norm1.bias", d),
            ["b.attention.attention.query.weight"] = qkvW.SliceRows(0, d),
            ["b.attention.attention.key.weight"] = qkvW.SliceRows(d, d),
            ["b.attention.attention.value.weight"] = qkvW.SliceRows(2L * d, d),
            ["b.attention.attention.query.bias"] = qkvB.SliceRows(0, d),
            ["b.attention.attention.key.bias"] = qkvB.SliceRows(d, d),
            ["b.attention.attention.value.bias"] = qkvB.SliceRows(2L * d, d),
            ["b.attention.output.dense.weight"] = W(p + "attn.proj.weight", d, d),
            ["b.attention.output.dense.bias"] = W(p + "attn.proj.bias", d),
            ["b.layernorm_after.weight"] = W(p + "norm2.weight", d),
            ["b.layernorm_after.bias"] = W(p + "norm2.bias", d),
            ["b.intermediate.dense.weight"] = W(p + "mlp.fc1.weight", inner, d),
            ["b.intermediate.dense.bias"] = W(p + "mlp.fc1.bias", inner),
            ["b.output.dense.weight"] = W(p + "mlp.fc2.weight", d, inner),
            ["b.output.dense.bias"] = W(p + "mlp.fc2.bias", d),
        };
    }

    private Tensor W(string name, params long[] shape)
    {
        string key = _prefix + name;
        if (!_weights.TryGetValue(key, out Tensor? tensor))
            throw new InvalidOperationException($"Depth Pro 2 focal weight '{key}' is missing.");
        if (tensor.DType != DType.F32)
            throw new InvalidOperationException($"Depth Pro 2 focal weight '{key}' is {tensor.DType}; the head takes F32 weights.");
        if (!tensor.Shape.Equals(new TensorShape(shape)))
            throw new InvalidOperationException($"Depth Pro 2 focal weight '{key}' has shape {tensor.Shape}, expected {new TensorShape(shape)}.");
        return tensor;
    }
}
