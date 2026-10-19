namespace HartsyInference.Diffusion.Requests;

/// <summary>How a DiT attends to in-context reference-image tokens.</summary>
public enum ReferenceAttentionMode
{
    /// <summary>The references run once, attending only to each other, and every block's keys and values for them are
    /// cached and appended to the live sequence's keys and values on each step (ai-toolkit's <c>kv_cache</c>, its
    /// edit-training default).</summary>
    CachedKv,

    /// <summary>The reference tokens ride in the live sequence every step, attending to and attended by everything
    /// (ai-toolkit's <c>kv_cache: false</c>).</summary>
    Joint,
}
