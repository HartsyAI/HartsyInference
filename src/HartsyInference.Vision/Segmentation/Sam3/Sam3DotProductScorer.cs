using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Vision.Segmentation.Sam2;

namespace HartsyInference.Vision.Segmentation.Sam3;

/// <summary>SAM 3 dot-product scorer: refines the prompt tokens, takes their masked mean and scores each query by a
/// scaled, clamped dot product with that pooled prompt.</summary>
public sealed class Sam3DotProductScorer
{
    private const string PromptMlpPrefix = "prompt_mlp.";

    private readonly SamMlpHead _promptMlp;
    private readonly Tensor _outNormWeight;
    private readonly Tensor _outNormBias;
    private readonly Tensor _promptProjWeight;
    private readonly Tensor _promptProjBias;
    private readonly Tensor _hsProjWeight;
    private readonly Tensor _hsProjBias;
    private readonly int _channels;
    private readonly float _eps;
    private readonly float _scale;
    private readonly float _clamp;

    /// <summary>Loads the prompt MLP, output norm and the two projections; <paramref name="weights"/> is relative to dot_prod_scoring.</summary>
    public Sam3DotProductScorer(Sam3Config config, IReadOnlyDictionary<string, Tensor> weights)
    {
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(weights);
        int c = config.DModel;
        int hidden = config.ScorerHidden;
        Sam3WeightAccess.Require(weights, $"{PromptMlpPrefix}layers.0.weight", hidden, c);
        Sam3WeightAccess.Require(weights, $"{PromptMlpPrefix}layers.0.bias", hidden);
        Sam3WeightAccess.Require(weights, $"{PromptMlpPrefix}layers.1.weight", c, hidden);
        Sam3WeightAccess.Require(weights, $"{PromptMlpPrefix}layers.1.bias", c);
        if (weights.ContainsKey($"{PromptMlpPrefix}layers.2.weight"))
        {
            throw new InvalidDataException("SAM 3 prompt MLP has more than two layers.");
        }

        _promptMlp = new SamMlpHead();
        _promptMlp.LoadWeights(weights, PromptMlpPrefix);

        _outNormWeight = Sam3WeightAccess.Require(weights, $"{PromptMlpPrefix}out_norm.weight", c);
        _outNormBias = Sam3WeightAccess.Require(weights, $"{PromptMlpPrefix}out_norm.bias", c);
        _promptProjWeight = Sam3WeightAccess.Require(weights, "prompt_proj.weight", c, c);
        _promptProjBias = Sam3WeightAccess.Require(weights, "prompt_proj.bias", c);
        _hsProjWeight = Sam3WeightAccess.Require(weights, "hs_proj.weight", c, c);
        _hsProjBias = Sam3WeightAccess.Require(weights, "hs_proj.bias", c);

        _channels = c;
        _eps = config.LayerNormEps;
        _scale = 1f / MathF.Sqrt(c);
        _clamp = config.ScorerClamp;
    }

    /// <summary>Returns class logits <c>[1, Q]</c> from query states <c>[1, Q, C]</c> and prompt tokens <c>[1, S, C]</c>; the caller owns the result.</summary>
    public Tensor Forward(IBackend backend, Tensor queryHidden, Tensor prompt, bool[] paddingMask)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(paddingMask);
        if (queryHidden.DType != DType.F32 || queryHidden.Shape.Rank != 3 || queryHidden.Shape[0] != 1 || queryHidden.Shape[2] != _channels)
        {
            throw new ArgumentException($"Scorer queries must be F32 [1, Q, {_channels}]; got {queryHidden.DType} {queryHidden.Shape}.", nameof(queryHidden));
        }

        if (prompt.DType != DType.F32 || prompt.Shape.Rank != 3 || prompt.Shape[0] != 1 || prompt.Shape[2] != _channels)
        {
            throw new ArgumentException($"Scorer prompt must be F32 [1, S, {_channels}]; got {prompt.DType} {prompt.Shape}.", nameof(prompt));
        }

        int seq = (int)prompt.Shape[1];
        int numQueries = (int)queryHidden.Shape[1];
        if (paddingMask.Length != seq)
        {
            throw new ArgumentException($"Scorer padding mask has {paddingMask.Length} entries for {seq} prompt tokens.", nameof(paddingMask));
        }

        int valid = 0;
        for (int i = 0; i < seq; i++)
        {
            if (!paddingMask[i])
            {
                valid++;
            }
        }

        if (valid == 0)
        {
            throw new ArgumentException("Scorer needs at least one non-padding prompt token.", nameof(paddingMask));
        }

        Tensor promptRows = prompt.Reshape(new TensorShape(seq, _channels));
        Tensor refined = _promptMlp.Forward(backend, promptRows);
        backend.Add(refined, refined, promptRows);
        Tensor normed = new Tensor(refined.Shape, DType.F32);
        backend.LayerNorm(normed, refined, _outNormWeight, _outNormBias, _eps);
        refined.Dispose();

        Tensor poolWeights = new Tensor(new TensorShape(1, 1, seq), DType.F32);
        Span<float> weightValues = poolWeights.AsSpan<float>();
        float inverseValid = 1f / valid;
        for (int i = 0; i < seq; i++)
        {
            weightValues[i] = paddingMask[i] ? 0f : inverseValid;
        }

        Tensor pooled = new Tensor(new TensorShape(1, 1, _channels), DType.F32);
        backend.BatchedMatMul(pooled, poolWeights, normed.Reshape(new TensorShape(1, seq, _channels)));
        poolWeights.Dispose();
        normed.Dispose();

        Tensor pooledProj = new Tensor(new TensorShape(1, 1, _channels), DType.F32);
        backend.Linear(pooledProj, pooled, _promptProjWeight, _promptProjBias);
        pooled.Dispose();

        Tensor hsProj = new Tensor(new TensorShape(1, numQueries, _channels), DType.F32);
        backend.Linear(hsProj, queryHidden, _hsProjWeight, _hsProjBias);

        // The GEMM writes an owned [1,Q,1] result; the [1,Q] view is taken afterwards, so Scale and Clamp read the computed scores.
        Tensor logits3 = new Tensor(new TensorShape(1, numQueries, 1), DType.F32);
        backend.BatchedMatMul(logits3, hsProj, pooledProj.Reshape(new TensorShape(1, _channels, 1)));
        hsProj.Dispose();
        pooledProj.Dispose();
        Tensor logits = logits3.Reshape(new TensorShape(1, numQueries));

        backend.Scale(logits, logits, _scale);
        backend.Clamp(logits, logits, -_clamp, _clamp);
        return logits;
    }
}
