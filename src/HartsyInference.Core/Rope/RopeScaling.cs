namespace HartsyInference.Core.Rope;

/// <summary>Parameters for a RoPE scaling transform (HuggingFace <c>rope_scaling</c> / llama.cpp <c>{arch}.rope.scaling.*</c>).</summary>
/// <remarks>When <see cref="InvFreqFactors"/> is set (e.g. the GGUF <c>rope_freqs.weight</c> tensor that
/// llama.cpp precomputes for Llama-3), it is applied as a direct per-frequency multiplier and the formula is
/// bypassed, so the engine reproduces llama.cpp exactly.</remarks>
public sealed record RopeScaling
{
    /// <summary>Standard RoPE (no scaling).</summary>
    public static readonly RopeScaling None = new();

    public RopeScalingType Type { get; init; } = RopeScalingType.None;

    /// <summary>Extension factor (Linear / YaRN / DynamicNtk). E.g. 8.0 for Llama-3.1 128k from 16k.</summary>
    public double Factor { get; init; } = 1.0;

    /// <summary>Pre-extension training context length (Llama3 / YaRN / DynamicNtk / LongRope).</summary>
    public double OriginalContextLength { get; init; }

    /// <summary>Llama3 low-frequency factor (HF <c>low_freq_factor</c>, default 1).</summary>
    public double LowFreqFactor { get; init; } = 1.0;

    /// <summary>Llama3 high-frequency factor (HF <c>high_freq_factor</c>, default 4).</summary>
    public double HighFreqFactor { get; init; } = 4.0;

    /// <summary>YaRN fast-rotation boundary (HF <c>beta_fast</c>, default 32).</summary>
    public double BetaFast { get; init; } = 32.0;

    /// <summary>YaRN slow-rotation boundary (HF <c>beta_slow</c>, default 1).</summary>
    public double BetaSlow { get; init; } = 1.0;

    /// <summary>DeepSeek YaRN <c>mscale_all_dim</c> (GGUF <c>rope.scaling.yarn_log_multiplier</c>): the attention mscale is <c>1 + YarnLogMultiplier·ln(factor)</c> and enters the attention <i>score</i> scale squared (DeepSeek-V2/V3 MLA). 0 (default) means the standard YaRN mscale. Not used by the cos/sin layout itself.</summary>
    public double YarnLogMultiplier { get; init; }

    /// <summary>Explicit attention scaling (mscale) baked into cos/sin. <c>NaN</c> = infer per-type (YaRN: 0.1·ln(factor)+1; others: 1).</summary>
    public double AttentionFactor { get; init; } = double.NaN;

    /// <summary>LongRope short-context per-dimension factors (length D/2), used when seqLen ≤ original context.</summary>
    public IReadOnlyList<double>? ShortFactor { get; init; }

    /// <summary>LongRope long-context per-dimension factors (length D/2), used when seqLen &gt; original context.</summary>
    public IReadOnlyList<double>? LongFactor { get; init; }

    /// <summary>Optional explicit per-frequency multiplier (length D/2). When set, it is applied directly to the base inverse frequencies and the formula is skipped — this is the GGUF <c>rope_freqs.weight</c> path (llama.cpp precomputes Llama-3 scaling into this tensor).</summary>
    public IReadOnlyList<float>? InvFreqFactors { get; init; }
}
