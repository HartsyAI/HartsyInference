using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Segmentation.Sam2;

/// <summary>One layer of SAM's two-way transformer. In order: token self-attention, token→image
/// cross-attention, an MLP on the tokens, then image→token cross-attention — each with a residual add and a
/// LayerNorm. The point-positional encoding is re-added to the query (and the image positional encoding to
/// the keys) before every attention, matching the reference. The first layer skips adding the PE to the
/// self-attention query (<paramref name="skipFirstLayerPe"/>).</summary>
public sealed class TwoWayAttentionBlock
{
    private readonly float _eps;
    private readonly bool _skipFirstLayerPe;

    private readonly DecoderAttention _selfAttn;
    private readonly DecoderAttention _crossTokenToImage;
    private readonly DecoderAttention _crossImageToToken;

    private Tensor? _norm1W, _norm1B;
    private Tensor? _norm2W, _norm2B;
    private Tensor? _norm3W, _norm3B;
    private Tensor? _norm4W, _norm4B;
    private Tensor? _mlpLin1W, _mlpLin1B;   // [hidden, dim]
    private Tensor? _mlpLin2W, _mlpLin2B;   // [dim, hidden]

    /// <summary>Creates the block; per-attention dims and the MLP hidden size come from the loaded weights.</summary>
    public TwoWayAttentionBlock(int heads, bool skipFirstLayerPe, float eps)
    {
        _eps = eps;
        _skipFirstLayerPe = skipFirstLayerPe;
        _selfAttn = new DecoderAttention(heads);
        _crossTokenToImage = new DecoderAttention(heads);
        _crossImageToToken = new DecoderAttention(heads);
    }

    /// <summary>Loads the block's three attentions, four norms, and two-layer MLP under <paramref name="prefix"/>.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _selfAttn.LoadWeights(weights, $"{prefix}self_attn.");
        _crossTokenToImage.LoadWeights(weights, $"{prefix}cross_attn_token_to_image.");
        _crossImageToToken.LoadWeights(weights, $"{prefix}cross_attn_image_to_token.");

        _norm1W = F32(weights, $"{prefix}norm1.weight");
        _norm1B = F32(weights, $"{prefix}norm1.bias");
        _norm2W = F32(weights, $"{prefix}norm2.weight");
        _norm2B = F32(weights, $"{prefix}norm2.bias");
        _norm3W = F32(weights, $"{prefix}norm3.weight");
        _norm3B = F32(weights, $"{prefix}norm3.bias");
        _norm4W = F32(weights, $"{prefix}norm4.weight");
        _norm4B = F32(weights, $"{prefix}norm4.bias");
        // SAM names the token MLP's two linears lin1/lin2 (MLPBlock), distinct from the layers.N heads.
        _mlpLin1W = F32(weights, $"{prefix}mlp.lin1.weight");
        _mlpLin1B = F32(weights, $"{prefix}mlp.lin1.bias");
        _mlpLin2W = F32(weights, $"{prefix}mlp.lin2.weight");
        _mlpLin2B = F32(weights, $"{prefix}mlp.lin2.bias");
    }

    /// <summary>Yields every loaded weight tensor.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _selfAttn.EnumerateWeights()) yield return t;
        foreach (Tensor t in _crossTokenToImage.EnumerateWeights()) yield return t;
        foreach (Tensor t in _crossImageToToken.EnumerateWeights()) yield return t;
        foreach (Tensor? t in new[] { _norm1W, _norm1B, _norm2W, _norm2B, _norm3W, _norm3B, _norm4W, _norm4B,
                                      _mlpLin1W, _mlpLin1B, _mlpLin2W, _mlpLin2B })
        {
            if (t is not null) yield return t;
        }
    }

    /// <summary>Runs one two-way layer; returns the updated (queries, keys).</summary>
    public (Tensor queries, Tensor keys) Forward(IBackend backend, Tensor queries, Tensor keys, Tensor queryPe, Tensor keyPe)
    {
        if (_norm1W is null)
            throw new InvalidOperationException("TwoWayAttentionBlock weights not loaded.");

        // Self-attention on tokens.
        if (_skipFirstLayerPe)
        {
            Tensor attnOut = _selfAttn.Forward(backend, queries, queries, queries);
            queries = SamDecoderOps.AddNew(backend, queries, attnOut);
        }
        else
        {
            Tensor q = SamDecoderOps.AddNew(backend, queries, queryPe);
            Tensor attnOut = _selfAttn.Forward(backend, q, q, queries);
            queries = SamDecoderOps.AddNew(backend, queries, attnOut);
        }
        queries = SamDecoderOps.LayerNormNew(backend, queries, _norm1W!, _norm1B!, _eps);

        // Cross-attention: tokens attend to the image.
        Tensor qc = SamDecoderOps.AddNew(backend, queries, queryPe);
        Tensor kc = SamDecoderOps.AddNew(backend, keys, keyPe);
        Tensor crossOut = _crossTokenToImage.Forward(backend, qc, kc, keys);
        queries = SamDecoderOps.AddNew(backend, queries, crossOut);
        queries = SamDecoderOps.LayerNormNew(backend, queries, _norm2W!, _norm2B!, _eps);

        // Token MLP (ReLU inner activation, via clamp to [0, +inf)).
        Tensor mlpOut = Mlp(backend, queries);
        queries = SamDecoderOps.AddNew(backend, queries, mlpOut);
        queries = SamDecoderOps.LayerNormNew(backend, queries, _norm3W!, _norm3B!, _eps);

        // Cross-attention: image attends to the tokens (note swapped q/k).
        Tensor qi = SamDecoderOps.AddNew(backend, queries, queryPe);
        Tensor ki = SamDecoderOps.AddNew(backend, keys, keyPe);
        Tensor crossOut2 = _crossImageToToken.Forward(backend, ki, qi, queries);
        keys = SamDecoderOps.AddNew(backend, keys, crossOut2);
        keys = SamDecoderOps.LayerNormNew(backend, keys, _norm4W!, _norm4B!, _eps);

        return (queries, keys);
    }

    private Tensor Mlp(IBackend backend, Tensor x)
    {
        int b = (int)x.Shape[0];
        int n = (int)x.Shape[1];
        int dim = (int)x.Shape[2];
        int hidden = (int)_mlpLin1W!.Shape[0];

        Tensor h = new Tensor(new TensorShape(b, n, hidden), DType.F32);
        backend.Linear(h, x, _mlpLin1W!, _mlpLin1B);
        backend.Clamp(h, h, 0f, float.MaxValue); // ReLU
        Tensor o = new Tensor(new TensorShape(b, n, dim), DType.F32);
        backend.Linear(o, h, _mlpLin2W!, _mlpLin2B);
        return o;
    }

    private static Tensor F32(IReadOnlyDictionary<string, Tensor> weights, string key)
    {
        if (!weights.TryGetValue(key, out Tensor? t))
            throw new KeyNotFoundException($"TwoWayAttentionBlock: missing weight '{key}'.");
        return t.DType != DType.F32 ? t.CastTo(DType.F32) : t;
    }
}
