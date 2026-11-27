namespace HartsyInference.LLM.Transformer.Compression;

/// <summary>Per-sequence cache of one HCA layer: a ring of the newest rotated key/value rows, the rotated compressed latents, and the
/// compressor's partial group. Rows are F32.</summary>
public sealed class HcaAttentionState
{
    /// <param name="capacity">Longest sequence this state will hold.</param>
    /// <param name="settings">The layer's shape.</param>
    public HcaAttentionState(int capacity, HcaAttentionSettings settings)
    {
        ArgumentNullException.ThrowIfNull(settings);
        ArgumentOutOfRangeException.ThrowIfLessThan(capacity, 1);
        settings.Validate();
        Capacity = capacity;
        Window = new float[settings.Window * settings.HeadDim];
        Compressed = new float[capacity / settings.CompressRatio * settings.HeadDim];
    }

    /// <summary>Longest sequence the state holds.</summary>
    public int Capacity { get; }

    /// <summary>Positions committed so far.</summary>
    public int Length { get; internal set; }

    /// <summary>Ring of the newest <c>Window</c> rows, position <c>p</c> at slot <c>p % Window</c>.</summary>
    internal float[] Window { get; }

    /// <summary>Completed compressed latents in group order, rotated at each group's first position.</summary>
    internal float[] Compressed { get; }

    /// <summary>The compressor's carried partial group; created by the layer on first use, since it holds the layer's bias and norm.</summary>
    internal KvCompressorState? Compressor { get; set; }

    /// <summary>Forgets the sequence.</summary>
    public void Reset()
    {
        Length = 0;
        Compressor?.Reset();
    }
}
