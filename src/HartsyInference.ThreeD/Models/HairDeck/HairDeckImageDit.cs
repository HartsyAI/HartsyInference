using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters;

namespace HartsyInference.ThreeD.Models.HairDeck;

/// <summary>The image-conditioned card DiT (<c>hairdeck/models/imgmodel.py</c>): the set DiT whose context is the projected
/// DINOv3 tokens of the 448 px crop plus a learned per-token position.</summary>
public sealed unsafe class HairDeckImageDit
{
    private readonly HairDeckImageCheckpointConverter.Weights _w;

    public HairDeckImageDit(HairDeckImageCheckpointConverter.Weights weights)
    {
        ArgumentNullException.ThrowIfNull(weights);
        _w = weights;
        Core = new HairDeckSetDit(weights.Dit);
        if (weights.CtxPos.Shape[0] != weights.Dit.ContextTokens)
            throw new ArgumentException("HairDeck image context positions do not match the context length.", nameof(weights));
    }

    /// <summary>The set DiT that runs on the card tokens.</summary>
    public HairDeckSetDit Core { get; }

    /// <summary>Number of DINOv3 context tokens (CLS, storage and patch tokens).</summary>
    public int ContextTokens => _w.Dit.ContextTokens;

    /// <summary>DINOv3 token width the context projection reads.</summary>
    public int DinoWidth => _w.DinoWidth;

    /// <summary>Encodes DINOv3 tokens <c>[ContextTokens, DinoWidth]</c> to the context <c>[ContextTokens, Dim]</c>; the caller disposes it.</summary>
    public Tensor EncodeContext(IBackend backend, Tensor dinoTokens)
    {
        if (dinoTokens.Shape.Rank != 2 || dinoTokens.Shape[0] != ContextTokens || dinoTokens.Shape[1] != _w.DinoWidth)
            throw new ArgumentException($"HairDeck DINOv3 tokens must be [{ContextTokens}, {_w.DinoWidth}], got {dinoTokens.Shape}.", nameof(dinoTokens));

        Tensor normed = HairDeckNumerics.LayerNorm(backend, dinoTokens, _w.CondNormWeight, _w.CondNormBias, 1e-5f);
        Tensor projected = HairDeckNumerics.Project(backend, normed, _w.CondProj);
        normed.Dispose();
        Tensor context = new(projected.Shape, DType.F32);
        backend.Add(context, projected, _w.CtxPos);
        projected.Dispose();
        return context;
    }
}
