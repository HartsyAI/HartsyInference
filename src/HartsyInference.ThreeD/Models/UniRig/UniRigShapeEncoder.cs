using HartsyInference.Core.Backends;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Geometry.Ops;
using HartsyInference.ThreeD.Models.Hunyuan3D;

namespace HartsyInference.ThreeD.Models.UniRig;

/// <summary>
/// The point encoder of both UniRig stages: Fourier-embedded points with normals, a cross-attention from farthest-point
/// queries to every input point, then residual self-attention blocks and a final LayerNorm. Built from the architecture
/// as shipped in the checkpoint (see <c>docs/Research/UNIRIG_ARCHITECTURE.md</c>); shares only the head-interleaved
/// projection layout with Hunyuan3D's ShapeVAE, not its modules.
/// </summary>
public sealed class UniRigShapeEncoder : IDisposable
{
    private const float LayerNormEps = 1e-5f;
    private const int FpsPoolFactor = 4;

    private readonly IBackend _backend;
    private readonly UniRigShapeEncoderConfig _config;
    private readonly FourierEmbedSpec _fourier;
    private readonly List<Tensor> _owned = [];
    private readonly Tensor _inputFourierWeight;
    private readonly Tensor _inputFeatureWeight;
    private readonly Tensor _inputBias;
    private readonly Norm _crossQueryNorm;
    private readonly Norm _crossDataNorm;
    private readonly Norm _crossMlpNorm;
    private readonly AttentionLayer _crossAttention;
    private readonly Mlp _crossMlp;
    private readonly Block[] _blocks;
    private readonly Norm _postNorm;
    private bool _disposed;

    /// <param name="weights">Checkpoint tensors by module path; the tensors stay owned by the caller and must outlive this encoder.</param>
    /// <param name="prefix">Path of the encoder module, e.g. <c>mesh_encoder.encoder</c> or <c>global_encoder.encoder</c>.</param>
    public UniRigShapeEncoder(IBackend backend, UniRigShapeEncoderConfig config, IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(weights);
        config.Validate();
        _backend = backend;
        _config = config;
        int width = config.Width;
        int headDim = width / config.Heads;
        int fourierWidth = 3 * (2 * config.NumFreqs + 1);
        _fourier = new FourierEmbedSpec(InputDims: 3, config.NumFreqs, IncludeInput: true);

        Tensor inputProj = UniRigCheckpoint.Get(weights, $"{prefix}.input_proj.weight", width, config.InputWidth);
        _inputFourierWeight = Own(SliceColumns(inputProj, 0, fourierWidth));
        _inputFeatureWeight = Own(SliceColumns(inputProj, fourierWidth, config.PointFeats));
        _inputBias = UniRigCheckpoint.Get(weights, $"{prefix}.input_proj.bias", width);

        string cross = $"{prefix}.cross_attn";
        _crossQueryNorm = LoadNorm(weights, $"{cross}.ln_1", width);
        _crossDataNorm = LoadNorm(weights, $"{cross}.ln_2", width);
        _crossMlpNorm = LoadNorm(weights, $"{cross}.ln_3", width);
        Tensor cq = UniRigCheckpoint.Get(weights, $"{cross}.attn.c_q.weight", width, width);
        (Tensor ck, Tensor cv, _) = Hunyuan3DVaeOps.SplitInterleavedWeight(
            UniRigCheckpoint.Get(weights, $"{cross}.attn.c_kv.weight", 2 * width, width), 2, config.Heads, headDim);
        Own(ck);
        Own(cv);
        _crossAttention = Attention(config, cq, ck, cv, weights, $"{cross}.attn.c_proj");
        _crossMlp = LoadMlp(weights, $"{cross}.mlp", width);

        _blocks = new Block[config.Layers];
        for (int i = 0; i < _blocks.Length; i++)
        {
            string block = $"{prefix}.self_attn.resblocks.{i}";
            (Tensor q, Tensor k, Tensor v) = Hunyuan3DVaeOps.SplitInterleavedWeight(
                UniRigCheckpoint.Get(weights, $"{block}.attn.c_qkv.weight", 3 * width, width), 3, config.Heads, headDim);
            Own(q);
            Own(k);
            Own(v);
            _blocks[i] = new Block(
                LoadNorm(weights, $"{block}.ln_1", width),
                Attention(config, q, k, v, weights, $"{block}.attn.c_proj"),
                LoadNorm(weights, $"{block}.ln_2", width),
                LoadMlp(weights, $"{block}.mlp", width));
        }

        _postNorm = LoadNorm(weights, $"{prefix}.ln_post", width);
    }

    /// <summary>Token width of <see cref="Encode"/>'s output.</summary>
    public int Width => _config.Width;

    /// <summary>Output token count.</summary>
    public int TokenNum => _config.TokenNum;

    /// <summary>
    /// Encodes a point cloud into <c>[1, TokenNum, Width]</c> tokens. Query points are <see cref="UniRigShapeEncoderConfig.TokenNum"/>
    /// farthest-point samples of <c>4 * TokenNum</c> points drawn at random with <paramref name="seed"/>; keys are all points.
    /// The caller owns the result.
    /// </summary>
    /// <param name="points">Positions in the normalized cube, length <c>3 * N</c>.</param>
    /// <param name="normals">Per-point features, length <c>PointFeats * N</c>.</param>
    public Tensor Encode(ReadOnlySpan<float> points, ReadOnlySpan<float> normals, int seed)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        int count = points.Length / 3;
        if (count == 0 || points.Length != 3 * count || normals.Length != _config.PointFeats * count)
        {
            throw new ArgumentException(
                $"UniRig encoder needs 3 floats and {_config.PointFeats} features per point; got {points.Length} and {normals.Length}.");
        }

        int[] pool = UniRigPointSelection.Choose(count, FpsPoolFactor * _config.TokenNum, seed);
        float[] poolPoints = new float[3 * pool.Length];
        float[] poolFeats = new float[_config.PointFeats * pool.Length];
        for (int i = 0; i < pool.Length; i++)
        {
            points.Slice(3 * pool[i], 3).CopyTo(poolPoints.AsSpan(3 * i, 3));
            normals.Slice(_config.PointFeats * pool[i], _config.PointFeats).CopyTo(poolFeats.AsSpan(_config.PointFeats * i, _config.PointFeats));
        }

        int[] picked = SurfaceSampler.FarthestPointIndices(poolPoints, _config.TokenNum);
        float[] queryPoints = new float[3 * picked.Length];
        float[] queryFeats = new float[_config.PointFeats * picked.Length];
        for (int i = 0; i < picked.Length; i++)
        {
            poolPoints.AsSpan(3 * picked[i], 3).CopyTo(queryPoints.AsSpan(3 * i, 3));
            poolFeats.AsSpan(_config.PointFeats * picked[i], _config.PointFeats).CopyTo(queryFeats.AsSpan(_config.PointFeats * i,
                _config.PointFeats));
        }

        using Tensor data = EmbedPoints(points, normals, count);
        Tensor queries = EmbedPoints(queryPoints, queryFeats, picked.Length);
        Tensor x = CrossAttend(queries, data);
        for (int i = 0; i < _blocks.Length; i++)
        {
            x = _blocks[i].Forward(this, x);
        }

        Tensor result = _postNorm.Apply(_backend, x);
        x.Dispose();
        return result;
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        foreach (Tensor tensor in _owned)
        {
            tensor.Dispose();
        }
    }

    private Tensor EmbedPoints(ReadOnlySpan<float> points, ReadOnlySpan<float> feats, int count)
    {
        int width = _config.Width;
        using Tensor coords = FromFloats(points, new TensorShape(count, 3));
        using Tensor fourier = new(new TensorShape(1, count, _fourier.Width), DType.F32);
        _backend.FourierEmbed(fourier, coords, _fourier);
        Tensor projected = new(new TensorShape(1, count, width), DType.F32);
        _backend.Linear(projected, fourier, _inputFourierWeight, _inputBias);
        if (_config.PointFeats == 0)
        {
            return projected;
        }

        using Tensor featTensor = FromFloats(feats, new TensorShape(1, count, _config.PointFeats));
        using Tensor featProjected = new(projected.Shape, DType.F32);
        _backend.Linear(featProjected, featTensor, _inputFeatureWeight, null);
        Tensor sum = new(projected.Shape, DType.F32);
        _backend.Add(sum, projected, featProjected);
        projected.Dispose();
        return sum;
    }

    private Tensor CrossAttend(Tensor queries, Tensor data)
    {
        using Tensor normedQueries = _crossQueryNorm.Apply(_backend, queries);
        using Tensor normedData = _crossDataNorm.Apply(_backend, data);
        using Tensor attended = _crossAttention.Forward(_backend, normedQueries, normedData, default);
        Tensor x = Sum(queries, attended);
        queries.Dispose();
        using Tensor normedX = _crossMlpNorm.Apply(_backend, x);
        using Tensor mlp = _crossMlp.Apply(_backend, normedX);
        Tensor result = Sum(x, mlp);
        x.Dispose();
        return result;
    }

    private Tensor Sum(Tensor a, Tensor b)
    {
        Tensor sum = new(a.Shape, DType.F32);
        _backend.Add(sum, a, b);
        return sum;
    }

    private AttentionLayer Attention(UniRigShapeEncoderConfig config, Tensor query, Tensor key, Tensor value,
        IReadOnlyDictionary<string, Tensor> weights, string projection)
    {
        int width = config.Width;
        return new AttentionLayer(
            new AttentionLayerConfig { Heads = config.Heads, KvHeads = config.Heads, HeadDim = width / config.Heads, ComputeDType = DType.F32 },
            new AttentionLayerWeights
            {
                Query = query,
                Key = key,
                Value = value,
                Output = UniRigCheckpoint.Get(weights, $"{projection}.weight", width, width),
                OutputBias = UniRigCheckpoint.Get(weights, $"{projection}.bias", width),
            });
    }

    private static Norm LoadNorm(IReadOnlyDictionary<string, Tensor> weights, string path, int width) =>
        new(UniRigCheckpoint.Get(weights, $"{path}.weight", width), UniRigCheckpoint.Get(weights, $"{path}.bias", width));

    private static Mlp LoadMlp(IReadOnlyDictionary<string, Tensor> weights, string path, int width) =>
        new(UniRigCheckpoint.Get(weights, $"{path}.c_fc.weight", 4 * width, width), UniRigCheckpoint.Get(weights, $"{path}.c_fc.bias", 4 * width),
            UniRigCheckpoint.Get(weights, $"{path}.c_proj.weight", width, 4 * width), UniRigCheckpoint.Get(weights, $"{path}.c_proj.bias", width));

    private Tensor Own(Tensor tensor)
    {
        _owned.Add(tensor);
        return tensor;
    }

    private static Tensor FromFloats(ReadOnlySpan<float> values, TensorShape shape)
    {
        Tensor tensor = new(shape, DType.F32);
        values.CopyTo(tensor.AsSpan<float>());
        return tensor;
    }

    private static Tensor SliceColumns(Tensor matrix, int start, int count)
    {
        int rows = (int)matrix.Shape[0];
        int columns = (int)matrix.Shape[1];
        Tensor slice = new(new TensorShape(rows, count), DType.F32);
        ReadOnlySpan<float> source = matrix.AsSpan<float>();
        Span<float> target = slice.AsSpan<float>();
        for (int r = 0; r < rows; r++)
        {
            source.Slice(r * columns + start, count).CopyTo(target.Slice(r * count, count));
        }

        return slice;
    }

    private sealed class Norm(Tensor weight, Tensor bias)
    {
        public Tensor Apply(IBackend backend, Tensor input)
        {
            Tensor output = new(input.Shape, DType.F32);
            backend.LayerNorm(output, input, weight, bias, LayerNormEps);
            return output;
        }
    }

    private sealed class Mlp(Tensor fcWeight, Tensor fcBias, Tensor projWeight, Tensor projBias)
    {
        public Tensor Apply(IBackend backend, Tensor input)
        {
            long hidden = fcWeight.Shape[0];
            using Tensor up = new(new TensorShape(input.Shape[0], input.Shape[1], hidden), DType.F32);
            backend.Linear(up, input, fcWeight, fcBias);
            using Tensor activated = new(up.Shape, DType.F32);
            backend.GeluErf(activated, up);
            Tensor output = new(input.Shape, DType.F32);
            backend.Linear(output, activated, projWeight, projBias);
            return output;
        }
    }

    private sealed class Block(Norm norm1, AttentionLayer attention, Norm norm2, Mlp mlp)
    {
        public Tensor Forward(UniRigShapeEncoder owner, Tensor x)
        {
            using Tensor normed1 = norm1.Apply(owner._backend, x);
            using Tensor attended = attention.Forward(owner._backend, normed1, normed1, default);
            Tensor x1 = owner.Sum(x, attended);
            x.Dispose();
            using Tensor normed2 = norm2.Apply(owner._backend, x1);
            using Tensor mlpOut = mlp.Apply(owner._backend, normed2);
            Tensor x2 = owner.Sum(x1, mlpOut);
            x1.Dispose();
            return x2;
        }
    }
}
