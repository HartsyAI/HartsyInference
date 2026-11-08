using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Segmentation.Sam3;

/// <summary>One SAM 3 ViT block: pre-norm RoPE attention (windowed or global), then a pre-norm exact-GELU MLP.</summary>
/// <remarks>Runs in F32. The upstream fused bf16 fc1 path (GOTCHA G11) is not reproduced.</remarks>
public sealed class Sam3VitBlock
{
    private readonly int _dim;
    private readonly int _heads;
    private readonly int _headDim;
    private readonly int _mlpHidden;
    private readonly int _tokens;
    private readonly int _windowCount;
    private readonly int _windowTokens;
    private readonly float _eps;
    private readonly float _scale;
    private readonly bool _globalAttention;
    private readonly int[] _windowGather;
    private readonly int[] _windowScatter;
    private readonly Tensor _norm1Weight;
    private readonly Tensor _norm1Bias;
    private readonly Tensor _qkvWeight;
    private readonly Tensor _qkvBias;
    private readonly Tensor _projWeight;
    private readonly Tensor _projBias;
    private readonly Tensor _norm2Weight;
    private readonly Tensor _norm2Bias;
    private readonly Tensor _fc1Weight;
    private readonly Tensor _fc1Bias;
    private readonly Tensor _fc2Weight;
    private readonly Tensor _fc2Bias;

    /// <summary>Loads one block's weights and fixes its attention layout.</summary>
    /// <param name="config">Trunk hyperparameters.</param>
    /// <param name="globalAttention">True to attend over the whole grid; false for window attention.</param>
    /// <param name="windowGather">Window-major row to token index, required for windowed blocks.</param>
    /// <param name="windowScatter">Inverse of <paramref name="windowGather"/>, required for windowed blocks.</param>
    /// <param name="weights">Trunk weights keyed relative to the trunk prefix.</param>
    /// <param name="prefix">Key prefix of this block, for example <c>blocks.3.</c>.</param>
    public Sam3VitBlock(
        Sam3Config config,
        bool globalAttention,
        int[]? windowGather,
        int[]? windowScatter,
        IReadOnlyDictionary<string, Tensor> weights,
        string prefix)
    {
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(weights);
        ArgumentNullException.ThrowIfNull(prefix);

        int grid = config.PatchGrid;
        _dim = config.VitEmbedDim;
        _heads = config.VitNumHeads;
        _headDim = _dim / _heads;
        _mlpHidden = config.VitMlpHidden;
        _tokens = grid * grid;
        _eps = config.LayerNormEps;
        _scale = 1f / MathF.Sqrt(_headDim);
        _globalAttention = globalAttention;

        if (globalAttention)
        {
            _windowGather = [];
            _windowScatter = [];
            _windowTokens = _tokens;
            _windowCount = 1;
        }
        else
        {
            ArgumentNullException.ThrowIfNull(windowGather);
            ArgumentNullException.ThrowIfNull(windowScatter);
            if (windowGather.Length != _tokens || windowScatter.Length != _tokens)
                throw new ArgumentException($"Window permutations must cover all {_tokens} tokens.");

            _windowGather = windowGather;
            _windowScatter = windowScatter;
            _windowTokens = config.VitWindowSize * config.VitWindowSize;
            _windowCount = _tokens / _windowTokens;
        }

        _norm1Weight = Sam3WeightAccess.Require(weights, prefix + "norm1.weight", _dim);
        _norm1Bias = Sam3WeightAccess.Require(weights, prefix + "norm1.bias", _dim);
        _qkvWeight = Sam3WeightAccess.Require(weights, prefix + "attn.qkv.weight", 3 * _dim, _dim);
        _qkvBias = Sam3WeightAccess.Require(weights, prefix + "attn.qkv.bias", 3 * _dim);
        _projWeight = Sam3WeightAccess.Require(weights, prefix + "attn.proj.weight", _dim, _dim);
        _projBias = Sam3WeightAccess.Require(weights, prefix + "attn.proj.bias", _dim);
        _norm2Weight = Sam3WeightAccess.Require(weights, prefix + "norm2.weight", _dim);
        _norm2Bias = Sam3WeightAccess.Require(weights, prefix + "norm2.bias", _dim);
        _fc1Weight = Sam3WeightAccess.Require(weights, prefix + "mlp.fc1.weight", _mlpHidden, _dim);
        _fc1Bias = Sam3WeightAccess.Require(weights, prefix + "mlp.fc1.bias", _mlpHidden);
        _fc2Weight = Sam3WeightAccess.Require(weights, prefix + "mlp.fc2.weight", _dim, _mlpHidden);
        _fc2Bias = Sam3WeightAccess.Require(weights, prefix + "mlp.fc2.bias", _dim);
    }

    /// <summary>Runs the block on token-major tokens and returns a new owned <c>[1, tokens, dim]</c> tensor.</summary>
    /// <param name="backend">Backend that runs the ops.</param>
    /// <param name="tokens">F32 <c>[1, tokens, dim]</c>; not modified and not disposed.</param>
    /// <param name="rope">RoPE tables shared by every block.</param>
    public Tensor Forward(IBackend backend, Tensor tokens, Sam3Rope rope)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(tokens);
        ArgumentNullException.ThrowIfNull(rope);
        if (tokens.DType != DType.F32 || tokens.Shape.Rank != 3 || tokens.Shape[0] != 1
            || tokens.Shape[1] != _tokens || tokens.Shape[2] != _dim)
        {
            throw new ArgumentException($"ViT block expects F32 tokens [1,{_tokens},{_dim}]; got {tokens.DType} {tokens.Shape}.", nameof(tokens));
        }

        using Tensor normed = new Tensor(tokens.Shape, DType.F32);
        backend.LayerNorm(normed, tokens, _norm1Weight, _norm1Bias, _eps);
        using Tensor attention = Attention(backend, normed, rope);
        using Tensor afterAttention = new Tensor(tokens.Shape, DType.F32);
        backend.Add(afterAttention, tokens, attention);

        using Tensor normedMlp = new Tensor(tokens.Shape, DType.F32);
        backend.LayerNorm(normedMlp, afterAttention, _norm2Weight, _norm2Bias, _eps);
        using Tensor hidden = new Tensor(new TensorShape(1, _tokens, _mlpHidden), DType.F32);
        backend.Linear(hidden, normedMlp, _fc1Weight, _fc1Bias);
        backend.GeluErf(hidden, hidden);
        using Tensor mlp = new Tensor(tokens.Shape, DType.F32);
        backend.Linear(mlp, hidden, _fc2Weight, _fc2Bias);

        Tensor output = new Tensor(tokens.Shape, DType.F32);
        backend.Add(output, afterAttention, mlp);
        return output;
    }

    private Tensor Attention(IBackend backend, Tensor normed, Sam3Rope rope)
    {
        using Tensor qkv = new Tensor(new TensorShape(1, _tokens, 3 * _dim), DType.F32);
        backend.Linear(qkv, normed, _qkvWeight, _qkvBias);
        using Tensor packed = qkv.Reshape(new TensorShape(1, _tokens, 3, _dim));
        using Tensor q = new Tensor(new TensorShape(1, _tokens, 1, _dim), DType.F32);
        using Tensor k = new Tensor(new TensorShape(1, _tokens, 1, _dim), DType.F32);
        using Tensor v = new Tensor(new TensorShape(1, _tokens, 1, _dim), DType.F32);
        Span<Tensor> parts = [q, k, v];
        backend.Split(parts, packed, 2);

        Tensor queryTokens = q.Reshape(new TensorShape(1, _tokens, _heads, _headDim));
        Tensor keyTokens = k.Reshape(new TensorShape(1, _tokens, _heads, _headDim));
        Tensor valueTokens = v.Reshape(new TensorShape(1, _tokens, _heads, _headDim));
        rope.Apply(backend, queryTokens, keyTokens, _globalAttention);

        int batch = _globalAttention ? 1 : _windowCount;
        int seq = _globalAttention ? _tokens : _windowTokens;
        using Tensor queryHeads = ToHeadMajor(backend, queryTokens, batch, seq);
        using Tensor keyHeads = ToHeadMajor(backend, keyTokens, batch, seq);
        using Tensor valueHeads = ToHeadMajor(backend, valueTokens, batch, seq);
        using Tensor attended = new Tensor(new TensorShape(batch, _heads, seq, _headDim), DType.F32);
        backend.ScaledDotProductAttention(attended, queryHeads, keyHeads, valueHeads, null, _scale);

        using Tensor attendedTokens = new Tensor(new TensorShape(batch, seq, _heads, _headDim), DType.F32);
        backend.Permute0213(attendedTokens, attended, _heads, seq, _headDim);
        return Project(backend, attendedTokens);
    }

    // Window-major gather happens after RoPE, so the table only needs in-window positions.
    private Tensor ToHeadMajor(IBackend backend, Tensor tokens, int batch, int seq)
    {
        Tensor headMajor = new Tensor(new TensorShape(batch, _heads, seq, _headDim), DType.F32);
        if (_globalAttention)
        {
            backend.Permute0213(headMajor, tokens.Reshape(new TensorShape(batch, seq, _heads, _headDim)), seq, _heads, _headDim);
            return headMajor;
        }

        using Tensor gathered = new Tensor(new TensorShape(_tokens, _dim), DType.F32);
        backend.GatherRows(gathered, tokens.Reshape(new TensorShape(_tokens, _dim)), _windowGather);
        backend.Permute0213(headMajor, gathered.Reshape(new TensorShape(batch, seq, _heads, _headDim)), seq, _heads, _headDim);
        return headMajor;
    }

    private Tensor Project(IBackend backend, Tensor attendedTokens)
    {
        Tensor output = new Tensor(new TensorShape(1, _tokens, _dim), DType.F32);
        if (_globalAttention)
        {
            backend.Linear(output, attendedTokens.Reshape(new TensorShape(1, _tokens, _dim)), _projWeight, _projBias);
            return output;
        }

        using Tensor restored = new Tensor(new TensorShape(_tokens, _dim), DType.F32);
        backend.GatherRows(restored, attendedTokens.Reshape(new TensorShape(_tokens, _dim)), _windowScatter);
        backend.Linear(output, restored.Reshape(new TensorShape(1, _tokens, _dim)), _projWeight, _projBias);
        return output;
    }
}
