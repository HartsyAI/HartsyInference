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

    /// <summary>One reference sits between the image and the text in the live sequence, under the step's own timestep
    /// modulation, on RoPE <c>r = 1</c> with its grid stretched onto the image grid (the colorize LoRA's layout). Full
    /// attention, no cache; only the image rows are read out.</summary>
    Inline,
}
