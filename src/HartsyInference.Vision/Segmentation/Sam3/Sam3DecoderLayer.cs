using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Segmentation.Sam3;

/// <summary>Post-norm decoder layer with a prepended presence token, text cross-attention and boxRPB-biased image cross-attention.</summary>
public sealed class Sam3DecoderLayer
{
    private const int Norm1 = 0;
    private const int Norm2 = 1;
    private const int Norm3 = 2;
    private const int CatextNorm = 3;

    private readonly int _dModel;
    private readonly int _heads;
    private readonly int _ffnWidth;
    private readonly float _eps;
    private readonly Sam3MultiheadAttention _selfAttention;
    private readonly Sam3MultiheadAttention _textAttention;
    private readonly Sam3MultiheadAttention _crossAttention;
    private readonly Sam3Mlp _feedForward = new();
    private Tensor[] _normWeights = [];
    private Tensor[] _normBiases = [];

    /// <summary>Creates a decoder layer from the configuration.</summary>
    public Sam3DecoderLayer(Sam3Config config)
    {
        ArgumentNullException.ThrowIfNull(config);
        _dModel = config.DModel;
        _heads = config.DecoderHeads;
        _ffnWidth = config.DecoderFfn;
        _eps = config.LayerNormEps;
        _selfAttention = new Sam3MultiheadAttention(config.DModel, config.DecoderHeads);
        _textAttention = new Sam3MultiheadAttention(config.DModel, config.DecoderHeads);
        _crossAttention = new Sam3MultiheadAttention(config.DModel, config.DecoderHeads);
    }

    /// <summary>Loads <c>cross_attn.*</c>, <c>ca_text.*</c>, <c>self_attn.*</c>, <c>linear1/linear2</c>, <c>norm1..norm3</c>
    /// and <c>catext_norm</c> under <paramref name="prefix"/>.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        ArgumentNullException.ThrowIfNull(weights);
        ArgumentNullException.ThrowIfNull(prefix);
        int d = _dModel;
        _selfAttention.LoadWeights(weights, prefix + "self_attn.");
        _textAttention.LoadWeights(weights, prefix + "ca_text.");
        _crossAttention.LoadWeights(weights, prefix + "cross_attn.");

        Tensor linear1Weight = Sam3WeightAccess.Require(weights, prefix + "linear1.weight", _ffnWidth, d);
        Tensor linear1Bias = Sam3WeightAccess.Require(weights, prefix + "linear1.bias", _ffnWidth);
        Tensor linear2Weight = Sam3WeightAccess.Require(weights, prefix + "linear2.weight", d, _ffnWidth);
        Tensor linear2Bias = Sam3WeightAccess.Require(weights, prefix + "linear2.bias", d);
        _feedForward.Bind([linear1Weight, linear2Weight], [linear1Bias, linear2Bias]);

        string[] normNames = ["norm1", "norm2", "norm3", "catext_norm"];
        Tensor[] normWeights = new Tensor[normNames.Length];
        Tensor[] normBiases = new Tensor[normNames.Length];
        for (int i = 0; i < normNames.Length; i++)
        {
            normWeights[i] = Sam3WeightAccess.Require(weights, $"{prefix}{normNames[i]}.weight", d);
            normBiases[i] = Sam3WeightAccess.Require(weights, $"{prefix}{normNames[i]}.bias", d);
        }
        _normWeights = normWeights;
        _normBiases = normBiases;
    }

    /// <summary>Runs one layer. <paramref name="queries"/> <c>[1, Nq, D]</c> and <paramref name="presence"/> <c>[1, 1, D]</c>
    /// are the layer inputs; <paramref name="queryPos"/> is the positional query <c>[1, Nq, D]</c>; <paramref name="crossBias"/>
    /// is the additive image bias <c>[1, heads, Nq + 1, HW]</c> whose first query row is the presence row. Returns new
    /// <c>(Queries [1, Nq, D], Presence [1, 1, D])</c> tensors the caller owns; the inputs are not disposed.</summary>
    public (Tensor Queries, Tensor Presence) Forward(
        IBackend backend,
        Tensor queries,
        Tensor presence,
        Tensor queryPos,
        Tensor promptFeatures,
        bool[] promptPadding,
        Tensor memory,
        Tensor memoryPos,
        Tensor crossBias)
    {
        ArgumentNullException.ThrowIfNull(backend);
        if (_normWeights.Length == 0)
            throw new InvalidOperationException("SAM 3 decoder layer weights are not loaded.");
        Sam3Ops.RequireSequence(queries, "decoder queries", _dModel);
        Sam3Ops.RequireSequence(presence, "decoder presence token", _dModel);
        Sam3Ops.RequireSequence(queryPos, "decoder query position", _dModel);
        Sam3Ops.RequireSequence(promptFeatures, "decoder prompt", _dModel);
        Sam3Ops.RequireSequence(memory, "decoder memory", _dModel);
        Sam3Ops.RequireSequence(memoryPos, "decoder memory position", _dModel);
        if (presence.Shape[1] != 1 || queryPos.Shape[1] != queries.Shape[1])
            throw new ArgumentException("Presence must be one row and the query position must match the queries.");
        if (promptPadding.Length != promptFeatures.Shape[1])
            throw new ArgumentException($"Prompt padding has {promptPadding.Length} entries for {promptFeatures.Shape[1]} prompt rows.", nameof(promptPadding));

        long nq = queries.Shape[1];
        long rows = nq + 1;

        Tensor zeroPresencePos = Sam3Ops.NewF32(1, 1, _dModel);
        Tensor tgt = Sam3Ops.ConcatRows(backend, presence, queries);
        Tensor pos = Sam3Ops.ConcatRows(backend, zeroPresencePos, queryPos);
        zeroPresencePos.Dispose();

        Tensor selfQk = Sam3Ops.Add(backend, tgt, pos);
        Tensor selfOut = _selfAttention.Forward(backend, selfQk, selfQk, tgt, null);
        selfQk.Dispose();
        Tensor afterSelf = Sam3Ops.Add(backend, tgt, selfOut);
        selfOut.Dispose();
        tgt.Dispose();
        Tensor normSelf = Sam3Ops.LayerNorm(backend, afterSelf, _normWeights[Norm2], _normBiases[Norm2], _eps);
        afterSelf.Dispose();

        Tensor textQuery = Sam3Ops.Add(backend, normSelf, pos);
        Tensor textBias = Sam3MultiheadAttention.KeyPaddingBias(promptPadding, (int)rows, _heads);
        Tensor textOut = _textAttention.Forward(backend, textQuery, promptFeatures, promptFeatures, textBias);
        textQuery.Dispose();
        textBias.Dispose();
        Tensor afterText = Sam3Ops.Add(backend, normSelf, textOut);
        normSelf.Dispose();
        textOut.Dispose();
        Tensor normText = Sam3Ops.LayerNorm(backend, afterText, _normWeights[CatextNorm], _normBiases[CatextNorm], _eps);
        afterText.Dispose();

        Tensor imageQuery = Sam3Ops.Add(backend, normText, pos);
        Tensor imageKeys = Sam3Ops.Add(backend, memory, memoryPos);
        Tensor imageOut = _crossAttention.Forward(backend, imageQuery, imageKeys, memory, crossBias);
        imageQuery.Dispose();
        imageKeys.Dispose();
        Tensor afterImage = Sam3Ops.Add(backend, normText, imageOut);
        normText.Dispose();
        imageOut.Dispose();
        Tensor normImage = Sam3Ops.LayerNorm(backend, afterImage, _normWeights[Norm1], _normBiases[Norm1], _eps);
        afterImage.Dispose();

        Tensor ffn = _feedForward.Forward(backend, normImage);
        Tensor afterFfn = Sam3Ops.Add(backend, normImage, ffn);
        normImage.Dispose();
        ffn.Dispose();
        Tensor normOut = Sam3Ops.LayerNorm(backend, afterFfn, _normWeights[Norm3], _normBiases[Norm3], _eps);
        afterFfn.Dispose();
        pos.Dispose();

        Tensor outPresence = Sam3Ops.NewF32(1, 1, _dModel);
        Tensor outQueries = Sam3Ops.NewF32(1, nq, _dModel);
        Tensor[] parts = [outPresence, outQueries];
        backend.Split(parts, normOut, 1);
        normOut.Dispose();
        return (outQueries, outPresence);
    }
}
