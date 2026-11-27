namespace HartsyInference.ThreeD.Models.Utonia;

/// <summary>Serialization of one point set: a sort permutation per curve and its inverse.</summary>
public sealed class UtoniaSerialized
{
    public UtoniaSerialized(int depth, int[][] order, int[][] inverse)
    {
        Depth = depth;
        Order = order;
        Inverse = inverse;
    }

    /// <summary>Coordinate depth in bits, from the largest grid coordinate.</summary>
    public int Depth { get; }

    /// <summary>Row k holds the point index at each serialized position under curve k.</summary>
    public int[][] Order { get; }

    /// <summary>Row k holds the serialized position of each point under curve k.</summary>
    public int[][] Inverse { get; }
}
