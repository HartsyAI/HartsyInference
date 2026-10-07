namespace HartsyInference.LLM.DeepSeekV41;

/// <summary>What attention layers hand down the stack instead of recomputing, for one forward pass over one sequence.</summary>
/// <remarks>Layers run in order and every source writes before its consumers read, so one slot each is enough (upstream <c>SharedAttentionRuntime</c>).
/// The arrays are the source layer's own caches, referenced and not copied.</remarks>
public sealed class DeepSeekV41SharedAttention
{
    /// <summary>The compressed latents of the latest KV-source layer.</summary>
    public float[]? CompressKv { get; set; }

    /// <summary>The index keys of the latest index-key owner.</summary>
    public float[]? IndexKeys { get; set; }

    /// <summary>Indices into the compressed latents, already offset past the window rows, <c>[tokens, TopkWidth]</c>.</summary>
    public int[]? Topk { get; set; }

    /// <summary>Entries per token in <see cref="Topk"/>.</summary>
    public int TopkWidth { get; set; }

    /// <summary>Candidate-block mask, <c>[tokens, CandidateWidth]</c>, from the candidate-source layer.</summary>
    public byte[]? Candidates { get; set; }

    /// <summary>Entries per token in <see cref="Candidates"/>.</summary>
    public int CandidateWidth { get; set; }

    /// <summary>Drops every slot, so nothing from one sequence can reach the next.</summary>
    public void Reset()
    {
        CompressKv = null;
        IndexKeys = null;
        Topk = null;
        TopkWidth = 0;
        Candidates = null;
        CandidateWidth = 0;
    }
}
