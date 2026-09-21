namespace HartsyInference.Core.Configuration;

/// <summary>Per-model layer-dump directories, all consumed through <c>DebugDumpSink</c>.</summary>
/// <remarks>Reached as a <c>DebugDumpSink</c> constructor argument, so neither the
/// <c>GetEnvironmentVariable</c> scan nor the <c>EnvFlag</c> sweep saw any of them. Uniform shape: a directory
/// path, null when unset, and the sink treats empty as unset. Dumping never changes generated output.</remarks>
public static partial class EngineKnobs
{
    /// <summary>Directory for Ace-Step denoiser layer dumps; unset disables dumping.</summary>
    public static readonly Knob<string?> AceStepDebugDir =
        Str("diagnostics.aceStepDebugDir", null, KnobScope.Runtime, KnobDomain.Diagnostics,
            "Directory for Ace-Step denoiser layer dumps; unset disables dumping.");

    /// <summary>Directory for Anima denoiser layer dumps; unset disables dumping.</summary>
    public static readonly Knob<string?> AnimaDebugDir =
        Str("diagnostics.animaDebugDir", null, KnobScope.Runtime, KnobDomain.Diagnostics,
            "Directory for Anima denoiser layer dumps; unset disables dumping.");

    /// <summary>Directory for Anima LLM adapter layer dumps; unset disables dumping.</summary>
    public static readonly Knob<string?> AnimaLlmAdapterDebugDir =
        Str("diagnostics.animaLlmAdapterDebugDir", null, KnobScope.Runtime, KnobDomain.Diagnostics,
            "Directory for Anima LLM adapter layer dumps; unset disables dumping.");

    /// <summary>Directory for AuraFlow denoiser layer dumps; unset disables dumping.</summary>
    public static readonly Knob<string?> AuraflowDebugDir =
        Str("diagnostics.auraflowDebugDir", null, KnobScope.Runtime, KnobDomain.Diagnostics,
            "Directory for AuraFlow denoiser layer dumps; unset disables dumping.");

    /// <summary>Directory for Chroma denoiser layer dumps; unset disables dumping.</summary>
    public static readonly Knob<string?> ChromaDebugDir =
        Str("diagnostics.chromaDebugDir", null, KnobScope.Runtime, KnobDomain.Diagnostics,
            "Directory for Chroma denoiser layer dumps; unset disables dumping.");

    /// <summary>Directory for Ernie-Image denoiser layer dumps; unset disables dumping.</summary>
    public static readonly Knob<string?> ErnieImageDebugDir =
        Str("diagnostics.ernieImageDebugDir", null, KnobScope.Runtime, KnobDomain.Diagnostics,
            "Directory for Ernie-Image denoiser layer dumps; unset disables dumping.");

    /// <summary>Directory for HiDream denoiser layer dumps; unset disables dumping.</summary>
    public static readonly Knob<string?> HidreamDebugDir =
        Str("diagnostics.hidreamDebugDir", null, KnobScope.Runtime, KnobDomain.Diagnostics,
            "Directory for HiDream denoiser layer dumps; unset disables dumping.");

    /// <summary>Directory for Hunyuan-Image denoiser layer dumps; unset disables dumping.</summary>
    public static readonly Knob<string?> HunyuanImageDebugDir =
        Str("diagnostics.hunyuanImageDebugDir", null, KnobScope.Runtime, KnobDomain.Diagnostics,
            "Directory for Hunyuan-Image denoiser layer dumps; unset disables dumping.");

    /// <summary>Directory for Ideogram 4 denoiser layer dumps; unset disables dumping.</summary>
    public static readonly Knob<string?> Ideogram4DebugDir =
        Str("diagnostics.ideogram4DebugDir", null, KnobScope.Runtime, KnobDomain.Diagnostics,
            "Directory for Ideogram 4 denoiser layer dumps; unset disables dumping.");

    /// <summary>Directory for Kandinsky 5 denoiser layer dumps; unset disables dumping.</summary>
    public static readonly Knob<string?> Kandinsky5DebugDir =
        Str("diagnostics.kandinsky5DebugDir", null, KnobScope.Runtime, KnobDomain.Diagnostics,
            "Directory for Kandinsky 5 denoiser layer dumps; unset disables dumping.");

    /// <summary>Directory for Lance denoiser layer dumps; unset disables dumping.</summary>
    public static readonly Knob<string?> LanceDebugDir =
        Str("diagnostics.lanceDebugDir", null, KnobScope.Runtime, KnobDomain.Diagnostics,
            "Directory for Lance denoiser layer dumps; unset disables dumping.");

    /// <summary>Directory for Lens denoiser layer dumps; unset disables dumping.</summary>
    public static readonly Knob<string?> LensDebugDir =
        Str("diagnostics.lensDebugDir", null, KnobScope.Runtime, KnobDomain.Diagnostics,
            "Directory for Lens denoiser layer dumps; unset disables dumping.");

    /// <summary>Directory for LTX-Video denoiser layer dumps; unset disables dumping.</summary>
    public static readonly Knob<string?> LtxDebugDir =
        Str("diagnostics.ltxDebugDir", null, KnobScope.Runtime, KnobDomain.Diagnostics,
            "Directory for LTX-Video denoiser layer dumps; unset disables dumping.");

    /// <summary>Directory for Lumina 2 denoiser layer dumps; unset disables dumping.</summary>
    public static readonly Knob<string?> Lumina2DebugDir =
        Str("diagnostics.lumina2DebugDir", null, KnobScope.Runtime, KnobDomain.Diagnostics,
            "Directory for Lumina 2 denoiser layer dumps; unset disables dumping.");

    /// <summary>Directory for Qwen-Image denoiser layer dumps; unset disables dumping.</summary>
    public static readonly Knob<string?> QwenImageDebugDir =
        Str("diagnostics.qwenImageDebugDir", null, KnobScope.Runtime, KnobDomain.Diagnostics,
            "Directory for Qwen-Image denoiser layer dumps; unset disables dumping.");

    /// <summary>Directory for Qwen-Image VAE layer dumps; unset disables dumping.</summary>
    public static readonly Knob<string?> QwenVaeDebugDir =
        Str("diagnostics.qwenVaeDebugDir", null, KnobScope.Runtime, KnobDomain.Diagnostics,
            "Directory for Qwen-Image VAE layer dumps; unset disables dumping.");

    /// <summary>Directory for SD3 denoiser layer dumps; unset disables dumping.</summary>
    public static readonly Knob<string?> Sd3DebugDir =
        Str("diagnostics.sd3DebugDir", null, KnobScope.Runtime, KnobDomain.Diagnostics,
            "Directory for SD3 denoiser layer dumps; unset disables dumping.");

    /// <summary>Directory for Wan denoiser layer dumps; unset disables dumping.</summary>
    public static readonly Knob<string?> WanDebugDir =
        Str("diagnostics.wanDebugDir", null, KnobScope.Runtime, KnobDomain.Diagnostics,
            "Directory for Wan denoiser layer dumps; unset disables dumping.");

    /// <summary>Directory for Z-Image denoiser layer dumps; unset disables dumping.</summary>
    public static readonly Knob<string?> ZImageDebugDir =
        Str("diagnostics.zImageDebugDir", null, KnobScope.Runtime, KnobDomain.Diagnostics,
            "Directory for Z-Image denoiser layer dumps; unset disables dumping.");

    /// <summary>Limited-interval CFG spec "lo,hi": skips the uncond forward outside that normalized-t band.</summary>
    /// <remarks>Numerics, not Diagnostics — skipping the uncond forward changes the image. The value is a spec
    /// string parsed at the call site, which THROWS on a malformed one by design, since silently ignoring a
    /// mistyped perf knob would invalidate an A/B run.</remarks>
    public static readonly Knob<string?> CfgInterval =
        Str("numerics.cfgInterval", null, KnobScope.Runtime, KnobDomain.Numerics,
            "Limited-interval CFG spec lo,hi: skips the uncond forward outside that normalized-t band.");
}
