namespace HartsyInference.Cuda;

/// <summary>One INT8 v2 call's workspace and output copied back for a test: logical layouts per <see cref="Int8AttentionV2Layout"/>.</summary>
internal sealed record Int8V2Readback
{
    /// <summary>The layout the call ran.</summary>
    public required Int8AttentionV2Layout Layout { get; init; }

    /// <summary>The output tensor's bytes.</summary>
    public required byte[] Output { get; init; }

    /// <summary>Q bytes <c>[B·H, Sq, 128]</c>.</summary>
    public required sbyte[] Q8 { get; init; }

    /// <summary>Q scales <c>[B·H, Sq]</c>.</summary>
    public required float[] QScale { get; init; }

    /// <summary>K bytes <c>[B·H, kvPad, 128]</c>.</summary>
    public required sbyte[] K8 { get; init; }

    /// <summary>K scales <c>[B·H, tiles, 4]</c>.</summary>
    public required float[] KScale { get; init; }

    /// <summary>V bytes <c>[B·H, 128, kvPad]</c>, keys permuted within 32-key chunks; empty for F16 PV.</summary>
    public required sbyte[] V8 { get; init; }

    /// <summary>V scales <c>[B·H, 128]</c> or <c>[B·H, tiles, 128]</c>; empty for F16 PV.</summary>
    public required float[] VScale { get; init; }

    /// <summary>K channel means <c>[B·H, 128]</c>.</summary>
    public required float[] KMean { get; init; }

    /// <summary>Traced INT32 QK products <c>[BR, kvPad]</c>; empty without a trace.</summary>
    public required int[] Dots { get; init; }

    /// <summary>Traced P bytes <c>[BR, kvPad]</c>.</summary>
    public required byte[] Probabilities { get; init; }

    /// <summary>Traced tile score maxima <c>[BR, tiles]</c>.</summary>
    public required float[] TileMax { get; init; }

    /// <summary>Traced byte sums <c>[BR, tiles]</c>.</summary>
    public required int[] Sums { get; init; }

    /// <summary>Traced INT32 PV partials <c>[BR, tiles, 128]</c>.</summary>
    public required int[] Partials { get; init; }
}
