namespace HartsyInference.Diffusion.Streaming;

/// <summary>One chunk of a chunked sequence: frames <c>[Start, End)</c>.</summary>
public readonly record struct ChunkSpan(int Index, int Start, int End)
{
    /// <summary>Frames in the chunk.</summary>
    public int Length => End - Start;
}
