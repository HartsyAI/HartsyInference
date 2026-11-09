using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Geometry.HairRig;
using HartsyInference.ThreeD.Models.HairDeck;

namespace HartsyInference.ThreeD.Io;

/// <summary>Writes one HairDeck sample as a GLB: the hair cards (material <c>hair</c>) and, when asked, the spring chains
/// as static tubes (material <c>spring_chains</c>). Chains are visualisation geometry, not a rig.</summary>
public static class HairDeckGlbExport
{
    private const float HairRoughness = 0.75f;
    private const float ChainRoughness = 0.5f;

    /// <summary>Serialises <paramref name="result"/> to a GLB. <paramref name="hairColorSrgb"/> is the base colour used when the cards
    /// carry no per-card colour (sRGB in [0, 1]).</summary>
    /// <exception cref="InvalidOperationException">The sample has no hair cards.</exception>
    public static byte[] Write(HairDeckResult result, double[] hairColorSrgb, bool withChains)
    {
        ArgumentNullException.ThrowIfNull(result);
        ArgumentNullException.ThrowIfNull(hairColorSrgb);
        if (hairColorSrgb.Length != 3)
            throw new ArgumentException("The hair colour must be three sRGB values.", nameof(hairColorSrgb));

        float[] hair = [(float)hairColorSrgb[0], (float)hairColorSrgb[1], (float)hairColorSrgb[2]];
        HairCardMesh? cards = HairDeckGeometry.BuildCards(result, hair);
        if (cards is null)
            throw new InvalidOperationException("No hair card has a triangle; there is nothing to export.");

        List<HairGlbNode> nodes =
        [
            new("hair_cards", cards.Mesh, new HairGlbMaterial("hair",
                [cards.BaseColorLinear[0], cards.BaseColorLinear[1], cards.BaseColorLinear[2], 1f],
                0f, HairRoughness, DoubleSided: true)),
        ];

        if (withChains && HairDeckGeometry.BuildChains(result) is { } chains)
        {
            nodes.Add(new HairGlbNode("spring_chains", chains, new HairGlbMaterial("spring_chains",
                [0.8f, 0.8f, 0.8f, 1f], 0f, ChainRoughness, DoubleSided: false)));
        }

        return HairCardGlbWriter.Write(nodes);
    }

    /// <summary>Serialises <paramref name="result"/> as a rigged GLB: the hair cards and the chain tubes are skinned to joints
    /// along the spring chains, and four clips (turn, nod, jump, sway) move the head and the chains as the reference's
    /// spring-bone simulation does. Needs at least one chain that binds a card.</summary>
    /// <exception cref="InvalidOperationException">The sample has no hair cards, or no chain binds a card.</exception>
    public static byte[] WriteRigged(HairDeckResult result, double[] hairColorSrgb)
    {
        ArgumentNullException.ThrowIfNull(result);
        ArgumentNullException.ThrowIfNull(hairColorSrgb);
        if (hairColorSrgb.Length != 3)
            throw new ArgumentException("The hair colour must be three sRGB values.", nameof(hairColorSrgb));

        float[] hair = [(float)hairColorSrgb[0], (float)hairColorSrgb[1], (float)hairColorSrgb[2]];
        List<HairCard> cards = HairDeckGeometry.ToCards(result);
        HairCardMesh? cardMesh = HairCardMesher.Build(cards, hair);
        if (cardMesh is null)
            throw new InvalidOperationException("No hair card has a triangle; there is nothing to export.");

        Mesh? tubes = HairDeckGeometry.BuildChains(result, out int[] tubeOwners);
        HairRigResult rig = HairDeckRig.Build(cards, cardMesh.Mesh, result.Chains, tubes, tubeOwners);

        List<HairGlbNode> nodes =
        [
            new("hair_cards", cardMesh.Mesh, new HairGlbMaterial("hair",
                [cardMesh.BaseColorLinear[0], cardMesh.BaseColorLinear[1], cardMesh.BaseColorLinear[2], 1f],
                0f, HairRoughness, DoubleSided: true), rig.HairSkin),
        ];
        if (tubes is not null && rig.ChainSkin is not null)
        {
            nodes.Add(new HairGlbNode("spring_chains", tubes, new HairGlbMaterial("spring_chains",
                [0.8f, 0.8f, 0.8f, 1f], 0f, ChainRoughness, DoubleSided: false), rig.ChainSkin));
        }

        return HairCardGlbWriter.Write(nodes, rig.Rig);
    }
}
