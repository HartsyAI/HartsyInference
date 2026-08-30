namespace HartsyInference.Core.Backends;

/// <summary>Generation-scoped, model-independent sparse token layout consumed by a backend VSA session.</summary>
/// <remarks><see cref="BlockOffsets"/> indexes <see cref="SourceIndices"/>. Tokens omitted from
/// <see cref="SourceIndices"/> are padding and never participate in routing or softmax.</remarks>
public sealed record VideoSparseAttentionPlan
{
    /// <summary>Versioned routing contract bound to the checkpoint profile.</summary>
    public required VideoSparseAttentionProfileKind Profile { get; init; }

    /// <summary>Logical packed sequence length, including omitted padding rows.</summary>
    public required int SequenceLength { get; init; }

    /// <summary>Exclusive prefix-sum offsets into <see cref="SourceIndices"/>; length is block count plus one.</summary>
    public required int[] BlockOffsets { get; init; }

    /// <summary>Live packed-sequence row for each block-local token.</summary>
    public required int[] SourceIndices { get; init; }

    /// <summary>Segment class for every block. Prefix blocks with different classes are never merged.</summary>
    public required int[] SegmentClasses { get; init; }

    /// <summary>Number of leading segment-pure prefix blocks that are always retained as attention sinks.</summary>
    public required int PrefixSinkBlocks { get; init; }

    /// <summary>Fraction of routeable blocks retained before forced sinks and neighbours.</summary>
    public float KeepFraction { get; init; } = 0.10f;

    /// <summary>Validates structural invariants before any device allocation occurs.</summary>
    public void Validate()
    {
        if (SequenceLength <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(SequenceLength));
        }
        if (BlockOffsets is null || BlockOffsets.Length < 2 || BlockOffsets[0] != 0)
        {
            throw new ArgumentException("Sparse block offsets must start at zero and contain at least one block.", nameof(BlockOffsets));
        }
        if (SourceIndices is null || BlockOffsets[^1] != SourceIndices.Length)
        {
            throw new ArgumentException("The final sparse block offset must equal the source-index count.", nameof(SourceIndices));
        }
        int blockCount = BlockOffsets.Length - 1;
        if (SegmentClasses is null || SegmentClasses.Length != blockCount)
        {
            throw new ArgumentException("Sparse segment classes must contain one value per block.", nameof(SegmentClasses));
        }
        if (PrefixSinkBlocks < 0 || PrefixSinkBlocks > blockCount)
        {
            throw new ArgumentOutOfRangeException(nameof(PrefixSinkBlocks));
        }
        if (!(KeepFraction > 0f && KeepFraction <= 1f) || !float.IsFinite(KeepFraction))
        {
            throw new ArgumentOutOfRangeException(nameof(KeepFraction));
        }
        bool[] seen = new bool[SequenceLength];
        for (int block = 0; block < blockCount; block++)
        {
            int start = BlockOffsets[block];
            int stop = BlockOffsets[block + 1];
            if (start > stop || start < 0 || stop > SourceIndices.Length)
            {
                throw new ArgumentException($"Sparse block {block} has invalid offsets [{start},{stop}).", nameof(BlockOffsets));
            }
            if (start == stop)
            {
                throw new ArgumentException($"Sparse block {block} has no live tokens.", nameof(BlockOffsets));
            }
            for (int i = start; i < stop; i++)
            {
                int source = SourceIndices[i];
                if ((uint)source >= (uint)SequenceLength)
                {
                    throw new ArgumentException($"Sparse source index {source} is outside [0,{SequenceLength}).", nameof(SourceIndices));
                }
                if (seen[source])
                {
                    throw new ArgumentException($"Sparse source index {source} appears in more than one block.", nameof(SourceIndices));
                }
                seen[source] = true;
            }
        }
    }
}
