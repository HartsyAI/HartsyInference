namespace HartsyInference.ThreeD.Pipelines;

/// <summary>
/// A scene prompt for a CoLa3D prompt checkpoint: one query per object, given either as a 2D mask in one view or as a 3D
/// point. Give exactly one of <see cref="MaskCoverage"/> and <see cref="Points"/>.
/// </summary>
public sealed record CoLa3DPrompt
{
    /// <summary>Objects named by the prompt, one query each.</summary>
    public required int Queries { get; init; }

    /// <summary>Mask coverage, 0 (outside) to 255 (inside), row-major <c>[Queries, MaskHeight, MaskWidth]</c>; null for a point prompt.</summary>
    public byte[]? MaskCoverage { get; init; }

    /// <summary>Pixel height of each mask.</summary>
    public int MaskHeight { get; init; }

    /// <summary>Pixel width of each mask.</summary>
    public int MaskWidth { get; init; }

    /// <summary>One <c>(x, y, z)</c> per query in the scene mesh's units, flat <c>[Queries, 3]</c>; null for a mask prompt.</summary>
    public float[]? Points { get; init; }

    /// <summary>Checks that exactly one prompt kind is set and that its buffers match the query count.</summary>
    public void Validate()
    {
        if (Queries < 1)
            throw new ArgumentOutOfRangeException(nameof(Queries), Queries, "A prompt names at least one object.");
        if ((MaskCoverage is null) == (Points is null))
            throw new ArgumentException("A prompt gives masks or points, not both and not neither.", nameof(Points));
        if (MaskCoverage is not null)
        {
            if (MaskHeight < 1)
                throw new ArgumentOutOfRangeException(nameof(MaskHeight), MaskHeight, "Mask height must be positive.");
            if (MaskWidth < 1)
                throw new ArgumentOutOfRangeException(nameof(MaskWidth), MaskWidth, "Mask width must be positive.");
            int plane = MaskHeight * MaskWidth;
            long expected = (long)Queries * plane;
            if (MaskCoverage.LongLength != expected)
                throw new ArgumentException($"Mask coverage holds {MaskCoverage.Length} bytes; expected {expected}.", nameof(MaskCoverage));
            // An empty mask has no centroid, so the decomposer would place its query at the origin.
            for (int q = 0; q < Queries; q++)
            {
                if (MaskCoverage.AsSpan(q * plane, plane).IndexOfAnyExcept((byte)0) < 0)
                    throw new ArgumentException($"Mask {q} has no coverage; every object needs a non-empty mask.", nameof(MaskCoverage));
            }

            return;
        }

        if (Points!.Length != Queries * 3)
            throw new ArgumentException($"Prompt points hold {Points.Length} values; expected {Queries * 3}.", nameof(Points));
        foreach (float value in Points)
        {
            if (!float.IsFinite(value))
                throw new ArgumentException("Prompt points must be finite.", nameof(Points));
        }
    }
}
