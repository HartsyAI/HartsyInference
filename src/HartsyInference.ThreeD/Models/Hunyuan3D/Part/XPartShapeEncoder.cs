using HartsyInference.Core.Backends;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Geometry.Ops;

namespace HartsyInference.ThreeD.Models.Hunyuan3D.Part;

/// <summary>Shape of the X-Part ShapeVAE encoder (<c>PointCrossAttentionEncoder</c> of <c>partgen/models/autoencoders</c>, <c>infer.yaml</c>): width 1024, 16 heads, 8 self-attention
/// blocks, 8 Fourier frequencies without pi, 4 point features (the normal and one more channel), q/k LayerNorm over the head dimension.</summary>
public sealed record XPartShapeEncoderConfig(int Width = 1024, int Heads = 16, int Layers = 8, int NumFreqs = 8, int PointFeats = 4)
{
    /// <summary>Elements per Fourier-embedded point: <c>3 (2 NumFreqs + 1)</c>.</summary>
    public int FourierWidth => 3 * (2 * NumFreqs + 1);

    /// <summary>Elements per point after concatenating the point features (the input of <c>input_proj</c>).</summary>
    public int InputWidth => FourierWidth + PointFeats;

    /// <summary>Per-head width.</summary>
    public int HeadDim => Width / Heads;
}

/// <summary>The X-Part ShapeVAE encoder: surface points with features become <c>[1, queries, Width]</c> latent tokens. Every point is a key; the queries are a subset of the points (upstream
/// takes them by farthest-point sampling from a random start, so the caller chooses the indices, and the numbers here match upstream for the same indices): Fourier embedding plus
/// <c>input_proj</c>, one residual cross-attention (queries over all points), the self-attention blocks and a final LayerNorm. Used for both conditioner encoders (object and part in
/// its box) and, with <c>encode_shape</c> semantics, nothing else of the VAE (no KL head, decoder transformer or <c>geo_decoder</c>).</summary>
public sealed class XPartShapeEncoder : IDisposable
{
    private const float Eps = 1e-6f, PostEps = 1e-5f;

    private readonly IBackend _backend;
    private readonly XPartShapeEncoderConfig _config;
    private readonly FourierEmbedSpec _fourier;
    private readonly List<Tensor> _owned = [];
    private Tensor _fourierWeight = null!, _featureWeight = null!, _inputBias = null!, _postW = null!, _postB = null!;
    private CrossBlock _cross = null!;
    private SelfBlock[] _blocks = [];
    private bool _disposed;

    /// <summary>Creates an encoder with no weights; call <see cref="LoadWeights"/>.</summary>
    public XPartShapeEncoder(IBackend backend, XPartShapeEncoderConfig? config = null)
    {
        ArgumentNullException.ThrowIfNull(backend);
        _backend = backend;
        _config = config ?? new XPartShapeEncoderConfig();
        _fourier = new FourierEmbedSpec(InputDims: 3, _config.NumFreqs, IncludeInput: true);
    }

    /// <summary>Token width of the output.</summary>
    public int Width => _config.Width;

    /// <summary>Reads the <c>encoder.*</c> tensors under <paramref name="prefix"/> (<c>""</c> for <c>shapevae.safetensors</c>, <c>geo_encoder.local_encoder</c> or <c>obj_encoder</c> in the
    /// conditioner). BF16 and F16 tensors widen to F32; the caller keeps ownership of <paramref name="weights"/>.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix = "")
    {
        ArgumentNullException.ThrowIfNull(weights);
        string p = prefix.Length == 0 ? "encoder" : prefix + ".encoder";
        int width = _config.Width, heads = _config.Heads, headDim = _config.HeadDim;
        Tensor inputProj = Own(F32(weights, $"{p}.input_proj.weight", width, _config.InputWidth));
        _fourierWeight = Own(SliceColumns(inputProj, 0, _config.FourierWidth));
        _featureWeight = Own(SliceColumns(inputProj, _config.FourierWidth, _config.PointFeats));
        _inputBias = Own(F32(weights, $"{p}.input_proj.bias", width));
        _cross = new CrossBlock(this, weights, $"{p}.cross_attn", width, heads, headDim);
        _blocks = new SelfBlock[_config.Layers];
        for (int i = 0; i < _blocks.Length; i++)
            _blocks[i] = new SelfBlock(this, weights, $"{p}.self_attn.resblocks.{i}", width, heads, headDim);
        _postW = Own(F32(weights, $"{p}.ln_post.weight", width));
        _postB = Own(F32(weights, $"{p}.ln_post.bias", width));
    }

    /// <summary>Encodes <paramref name="points"/> (length <c>3 N</c>) with <paramref name="features"/> (length <c>PointFeats N</c>), using the points at <paramref name="queryIndices"/> as queries.
    /// Returns <c>[1, queryIndices.Length, Width]</c>; the caller disposes it.</summary>
    public Tensor Encode(ReadOnlySpan<float> points, ReadOnlySpan<float> features, ReadOnlySpan<int> queryIndices)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        int count = points.Length / 3, feats = _config.PointFeats;
        if (_blocks.Length == 0 && _config.Layers > 0)
            throw new InvalidOperationException("Call LoadWeights first.");
        if (count == 0 || points.Length != 3 * count || features.Length != feats * count || queryIndices.Length == 0)
            throw new ArgumentException($"The encoder needs 3 floats and {feats} features per point and at least one query; got {points.Length}, {features.Length} and {queryIndices.Length} queries.");
        float[] queryPoints = new float[3 * queryIndices.Length], queryFeats = new float[feats * queryIndices.Length];
        for (int i = 0; i < queryIndices.Length; i++)
        {
            int q = queryIndices[i];
            if ((uint)q >= (uint)count)
                throw new ArgumentOutOfRangeException(nameof(queryIndices), $"Query index {q} is outside the {count} points.");
            points.Slice(3 * q, 3).CopyTo(queryPoints.AsSpan(3 * i, 3));
            features.Slice(feats * q, feats).CopyTo(queryFeats.AsSpan(feats * i, feats));
        }

        using Tensor data = Embed(points, features, count);
        Tensor x = _cross.Forward(Embed(queryPoints, queryFeats, queryIndices.Length), data);
        foreach (SelfBlock block in _blocks)
            x = block.Forward(x);
        Tensor result = LayerNorm(x, _postW, _postB, PostEps);
        x.Dispose();
        return result;
    }

    /// <summary>As <see cref="Encode(ReadOnlySpan{float}, ReadOnlySpan{float}, ReadOnlySpan{int})"/> with <paramref name="queryCount"/> farthest-point samples as queries, starting from
    /// <paramref name="startIndex"/> (upstream starts from a random point).</summary>
    public Tensor Encode(ReadOnlySpan<float> points, ReadOnlySpan<float> features, int queryCount, int startIndex = 0) =>
        Encode(points, features, SurfaceSampler.FarthestPointIndices(points, queryCount, startIndex));

    /// <summary>Releases the converted weights.</summary>
    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;
        foreach (Tensor t in _owned)
            t.Dispose();
    }

    private Tensor Embed(ReadOnlySpan<float> points, ReadOnlySpan<float> features, int count)
    {
        int width = _config.Width;
        using Tensor coords = FromFloats(points, new TensorShape(count, 3));
        using Tensor fourier = new(new TensorShape(1, count, _fourier.Width), DType.F32);
        _backend.FourierEmbed(fourier, coords, _fourier);
        using Tensor projected = new(new TensorShape(1, count, width), DType.F32);
        _backend.Linear(projected, fourier, _fourierWeight, _inputBias);
        using Tensor featTensor = FromFloats(features, new TensorShape(1, count, _config.PointFeats));
        using Tensor featProjected = new(projected.Shape, DType.F32);
        _backend.Linear(featProjected, featTensor, _featureWeight, null);
        return Add(projected, featProjected);
    }

    private Tensor Add(Tensor a, Tensor b)
    {
        Tensor sum = new(a.Shape, DType.F32);
        _backend.Add(sum, a, b);
        return sum;
    }

    private Tensor LayerNorm(Tensor input, Tensor weight, Tensor bias, float eps)
    {
        Tensor output = new(input.Shape, DType.F32);
        _backend.LayerNorm(output, input, weight, bias, eps);
        return output;
    }

    // nn.GELU() (erf) over c_fc, then c_proj.
    private Tensor Mlp(Tensor input, Tensor fcW, Tensor fcB, Tensor projW, Tensor projB)
    {
        using Tensor up = new(new TensorShape(input.Shape[0], input.Shape[1], fcW.Shape[0]), DType.F32);
        _backend.Linear(up, input, fcW, fcB);
        using Tensor activated = new(up.Shape, DType.F32);
        _backend.GeluErf(activated, up);
        Tensor output = new(input.Shape, DType.F32);
        _backend.Linear(output, activated, projW, projB);
        return output;
    }

    // q_norm / k_norm: LayerNorm over the head dimension, with bias.
    private Tensor HeadNorm(Tensor perHead, Tensor weight, Tensor bias)
    {
        Tensor normed = new(perHead.Shape, DType.F32);
        _backend.LayerNorm(normed, perHead, weight, bias, Eps);
        return normed;
    }

    private Tensor Own(Tensor tensor)
    {
        _owned.Add(tensor);
        return tensor;
    }

    private Tensor F32(IReadOnlyDictionary<string, Tensor> weights, string key, params long[] shape)
    {
        if (!weights.TryGetValue(key, out Tensor? source))
            throw new KeyNotFoundException($"The X-Part encoder tensor '{key}' is missing.");
        if (!source.Shape.Equals(new TensorShape(shape)))
            throw new InvalidDataException($"The X-Part encoder tensor '{key}' has shape {source.Shape}, expected {new TensorShape(shape)}.");
        Tensor widened = Hunyuan3DShapeVae.F32(source);
        return ReferenceEquals(widened, source) ? source : Own(widened);
    }

    private static Tensor FromFloats(ReadOnlySpan<float> values, TensorShape shape)
    {
        Tensor tensor = new(shape, DType.F32);
        values.CopyTo(tensor.AsSpan<float>());
        return tensor;
    }

    private static Tensor SliceColumns(Tensor matrix, int start, int count)
    {
        int rows = (int)matrix.Shape[0], columns = (int)matrix.Shape[1];
        Tensor slice = new(new TensorShape(rows, count), DType.F32);
        ReadOnlySpan<float> source = matrix.AsSpan<float>();
        Span<float> target = slice.AsSpan<float>();
        for (int r = 0; r < rows; r++)
            source.Slice(r * columns + start, count).CopyTo(target.Slice(r * count, count));
        return slice;
    }

    // ResidualCrossAttentionBlock: x = x + c_proj(attn(q_norm(c_q(ln_1 x)), k_norm(c_kv(ln_2 data)), v)); x = x + mlp(ln_3 x).
    private sealed class CrossBlock
    {
        private readonly XPartShapeEncoder _o;
        private readonly int _heads, _headDim;
        private readonly Tensor _ln1W, _ln1B, _ln2W, _ln2B, _ln3W, _ln3B, _qnW, _qnB, _knW, _knB, _fcW, _fcB, _fcpW, _fcpB, _cq, _ck, _cv;
        private readonly AttentionLayer _attention;

        public CrossBlock(XPartShapeEncoder o, IReadOnlyDictionary<string, Tensor> w, string p, int width, int heads, int headDim)
        {
            _o = o; _heads = heads; _headDim = headDim;
            _ln1W = o.F32(w, $"{p}.ln_1.weight", width); _ln1B = o.F32(w, $"{p}.ln_1.bias", width);
            _ln2W = o.F32(w, $"{p}.ln_2.weight", width); _ln2B = o.F32(w, $"{p}.ln_2.bias", width);
            _ln3W = o.F32(w, $"{p}.ln_3.weight", width); _ln3B = o.F32(w, $"{p}.ln_3.bias", width);
            _qnW = o.F32(w, $"{p}.attn.attention.q_norm.weight", headDim); _qnB = o.F32(w, $"{p}.attn.attention.q_norm.bias", headDim);
            _knW = o.F32(w, $"{p}.attn.attention.k_norm.weight", headDim); _knB = o.F32(w, $"{p}.attn.attention.k_norm.bias", headDim);
            _cq = o.F32(w, $"{p}.attn.c_q.weight", width, width);
            (_ck, _cv, _) = Hunyuan3DVaeOps.SplitInterleavedWeight(o.F32(w, $"{p}.attn.c_kv.weight", 2 * width, width), 2, heads, headDim);
            o.Own(_ck); o.Own(_cv);
            _attention = new AttentionLayer(
                new AttentionLayerConfig { Heads = heads, KvHeads = heads, HeadDim = headDim, ComputeDType = DType.F32 },
                new AttentionLayerWeights { Query = _cq, Key = _ck, Value = _cv, Output = o.F32(w, $"{p}.attn.c_proj.weight", width, width), OutputBias = o.F32(w, $"{p}.attn.c_proj.bias", width) });
            _fcW = o.F32(w, $"{p}.mlp.c_fc.weight", 4 * width, width); _fcB = o.F32(w, $"{p}.mlp.c_fc.bias", 4 * width);
            _fcpW = o.F32(w, $"{p}.mlp.c_proj.weight", width, 4 * width); _fcpB = o.F32(w, $"{p}.mlp.c_proj.bias", width);
        }

        public Tensor Forward(Tensor queries, Tensor data)
        {
            IBackend b = _o._backend;
            int nq = (int)queries.Shape[1], nd = (int)data.Shape[1], width = _heads * _headDim;
            using Tensor n1 = _o.LayerNorm(queries, _ln1W, _ln1B, Eps);
            using Tensor n2 = _o.LayerNorm(data, _ln2W, _ln2B, Eps);
            TensorShape queryHeads = new(1, nq, _heads, _headDim), dataHeads = new(1, nd, _heads, _headDim);
            using Tensor q = new(queryHeads, DType.F32);
            b.Linear(q, n1, _cq, null);
            using Tensor k = new(dataHeads, DType.F32);
            b.Linear(k, n2, _ck, null);
            using Tensor vh = new(dataHeads, DType.F32);
            b.Linear(vh, n2, _cv, null);
            using Tensor qn = _o.HeadNorm(q, _qnW, _qnB);
            using Tensor kn = _o.HeadNorm(k, _knW, _knB);
            using Tensor attended = _attention.Attend(b, qn, kn, vh, new TensorShape(1, nq, width), default);
            using Tensor projected = _attention.ProjectOutput(b, attended);
            Tensor x = _o.Add(queries, projected);
            queries.Dispose();
            using Tensor n3 = _o.LayerNorm(x, _ln3W, _ln3B, Eps);
            using Tensor mlp = _o.Mlp(n3, _fcW, _fcB, _fcpW, _fcpB);
            Tensor result = _o.Add(x, mlp);
            x.Dispose();
            return result;
        }
    }

    // ResidualAttentionBlock: x = x + attn(ln_1 x) with q/k LayerNorm; x = x + mlp(ln_2 x).
    private sealed class SelfBlock
    {
        private readonly XPartShapeEncoder _o;
        private readonly int _heads, _headDim;
        private readonly Tensor _ln1W, _ln1B, _ln2W, _ln2B, _qnW, _qnB, _knW, _knB, _fcW, _fcB, _fcpW, _fcpB;
        private readonly AttentionLayer _attention;

        public SelfBlock(XPartShapeEncoder o, IReadOnlyDictionary<string, Tensor> w, string p, int width, int heads, int headDim)
        {
            _o = o; _heads = heads; _headDim = headDim;
            _ln1W = o.F32(w, $"{p}.ln_1.weight", width); _ln1B = o.F32(w, $"{p}.ln_1.bias", width);
            _ln2W = o.F32(w, $"{p}.ln_2.weight", width); _ln2B = o.F32(w, $"{p}.ln_2.bias", width);
            _qnW = o.F32(w, $"{p}.attn.attention.q_norm.weight", headDim); _qnB = o.F32(w, $"{p}.attn.attention.q_norm.bias", headDim);
            _knW = o.F32(w, $"{p}.attn.attention.k_norm.weight", headDim); _knB = o.F32(w, $"{p}.attn.attention.k_norm.bias", headDim);
            (Tensor q, Tensor k, Tensor v) = Hunyuan3DVaeOps.SplitInterleavedWeight(o.F32(w, $"{p}.attn.c_qkv.weight", 3 * width, width), 3, heads, headDim);
            o.Own(q); o.Own(k); o.Own(v);
            _attention = new AttentionLayer(
                new AttentionLayerConfig { Heads = heads, KvHeads = heads, HeadDim = headDim, ComputeDType = DType.F32 },
                new AttentionLayerWeights { Query = q, Key = k, Value = v, Output = o.F32(w, $"{p}.attn.c_proj.weight", width, width), OutputBias = o.F32(w, $"{p}.attn.c_proj.bias", width) });
            _fcW = o.F32(w, $"{p}.mlp.c_fc.weight", 4 * width, width); _fcB = o.F32(w, $"{p}.mlp.c_fc.bias", 4 * width);
            _fcpW = o.F32(w, $"{p}.mlp.c_proj.weight", width, 4 * width); _fcpB = o.F32(w, $"{p}.mlp.c_proj.bias", width);
        }

        public Tensor Forward(Tensor x)
        {
            IBackend b = _o._backend;
            int n = (int)x.Shape[1], width = _heads * _headDim;
            using Tensor n1 = _o.LayerNorm(x, _ln1W, _ln1B, Eps);
            (Tensor q, Tensor k, Tensor v) = _attention.ProjectSelf(b, n1, releaseInput: false);
            using Tensor qn = _o.HeadNorm(q, _qnW, _qnB);
            using Tensor kn = _o.HeadNorm(k, _knW, _knB);
            q.Dispose();
            k.Dispose();
            using Tensor attended = _attention.Attend(b, qn, kn, v, new TensorShape(1, n, width), default);
            v.Dispose();
            using Tensor projected = _attention.ProjectOutput(b, attended);
            Tensor x1 = _o.Add(x, projected);
            x.Dispose();
            using Tensor n2 = _o.LayerNorm(x1, _ln2W, _ln2B, Eps);
            using Tensor mlp = _o.Mlp(n2, _fcW, _fcB, _fcpW, _fcpB);
            Tensor x2 = _o.Add(x1, mlp);
            x1.Dispose();
            return x2;
        }
    }
}
