namespace HartsyInference.LLM.Looped;

/// <summary>How the recurrent block's attention layers store key/value state across loops (Dory's <c>recurrent_kv_cache_mode</c>).</summary>
public enum LoopedKvCacheMode
{
    /// <summary>Every loop keeps its own cache for each recurrent attention layer, so loop <c>k</c> attends to loop-<c>k</c> keys of every
    /// earlier token; the checkpoint default and the configuration the weights were trained under. Cache slots grow with the loop count.</summary>
    PerLoop,

    /// <summary>One cache per recurrent attention layer, overwritten by each loop for the tokens of the current forward: loop <c>k</c> sees
    /// loop-<c>k</c> keys for this chunk and last-loop keys for earlier chunks. Fixed memory whatever the loop count; output depends on how
    /// the prompt was chunked.</summary>
    LastLoop,
}
