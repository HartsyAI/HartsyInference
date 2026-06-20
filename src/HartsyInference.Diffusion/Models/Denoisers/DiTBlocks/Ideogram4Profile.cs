namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Opt-in per-forward timing accumulators for the Ideogram 4 block, gated by the
/// <c>HARTSY_DIT_PROFILE=1</c> environment variable. When enabled, the block syncs the device
/// around its attention and MLP sublayers and adds the elapsed milliseconds here; the pipeline
/// resets at each step and logs the totals. Off by default (zero overhead) since the syncs would
/// otherwise serialize the host and device.</summary>
public static class Ideogram4Profile
{
    /// <summary>True when <c>HARTSY_DIT_PROFILE=1</c>. Read once at startup.</summary>
    public static readonly bool Enabled = Environment.GetEnvironmentVariable("HARTSY_DIT_PROFILE") == "1";

    /// <summary>Accumulated milliseconds in the attention sublayer (QKV proj → SDPA → out proj).</summary>
    public static double AttentionMs;

    /// <summary>Accumulated milliseconds in the MLP/SwiGLU sublayer.</summary>
    public static double MlpMs;

    /// <summary>Resets the accumulators (called per denoise step by the pipeline).</summary>
    public static void Reset()
    {
        AttentionMs = 0;
        MlpMs = 0;
    }
}
