using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters;

namespace HartsyInference.ThreeD.Models.HairDeck;

/// <summary>The spring-bone chain DiT (<c>hairdeck/spring/chaindit.py</c>): the set DiT over 48 chain slots, conditioned on
/// the 192 card tokens through LayerNorm, a projection and one position row shared by all slots.</summary>
public sealed unsafe class HairDeckChainDit
{
    private readonly HairDeckChainCheckpointConverter.Weights _w;

    public HairDeckChainDit(HairDeckChainCheckpointConverter.Weights weights)
    {
        ArgumentNullException.ThrowIfNull(weights);
        _w = weights;
        Core = new HairDeckSetDit(weights.Dit);
    }

    /// <summary>The set DiT that runs on the chain tokens.</summary>
    public HairDeckSetDit Core { get; }

    /// <summary>Number of card slots of the condition.</summary>
    public int CardSlots => _w.Dit.ContextTokens;

    /// <summary>Encodes the card tokens <c>[CardSlots, 40]</c> (clean tokens, no sign rule) to <c>[CardSlots, Dim]</c>; the caller disposes it.</summary>
    public Tensor EncodeContext(IBackend backend, Tensor cards)
    {
        if (cards.Shape.Rank != 2 || cards.Shape[0] != CardSlots || cards.Shape[1] != _w.CardWidth)
            throw new ArgumentException($"HairDeck card tokens must be [{CardSlots}, {_w.CardWidth}], got {cards.Shape}.", nameof(cards));

        Tensor normed = HairDeckNumerics.LayerNorm(backend, cards, _w.CardNormWeight, _w.CardNormBias, 1e-5f);
        Tensor projected = HairDeckNumerics.Project(backend, normed, _w.CardProj);
        normed.Dispose();
        HairDeckNumerics.AddRowInPlace(projected, _w.CardPos);
        return projected;
    }
}
