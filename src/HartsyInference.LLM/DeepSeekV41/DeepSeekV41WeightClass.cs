namespace HartsyInference.LLM.DeepSeekV41;

/// <summary>The memory classes a DeepSeek-V4.1 tensor falls into; the residency planner budgets each class separately.</summary>
public enum DeepSeekV41WeightClass
{
    /// <summary>Attention, hyper-connection, gate, shared-expert, compressor, indexer and Engram projection tensors of the backbone layers.</summary>
    Dense,

    /// <summary>Routed expert weights and their scale companions.</summary>
    Expert,

    /// <summary>The huge Engram lookup tables (read by row, never loaded whole).</summary>
    Engram,

    /// <summary>The token embedding.</summary>
    Embed,

    /// <summary>The output head and the final norm.</summary>
    Head,

    /// <summary>The image tower, aligner and image marker vectors.</summary>
    Vision,

    /// <summary>Every <c>mtp.*</c> tensor of the draft layers.</summary>
    Draft,
}
