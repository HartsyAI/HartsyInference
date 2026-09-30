namespace HartsyInference.Core.Configuration;

/// <summary>Knobs named after the model or subsystem they belong to rather than the engine.</summary>
/// <remarks>These were missed by the first inventory of the surface, which is why
/// they arrived last. All are Runtime; the ones held in a <c>static readonly</c> field resolve once at process
/// launch, so changing them mid-process has no effect.
/// <para>Three read as debug flags but genuinely change output, so they are Numerics rather than Diagnostics:
/// <see cref="AnimaBypassLlmAdapter"/> swaps the conditioning path, <see cref="HiftDeterministic"/> disables NSF
/// additive noise, and <see cref="Mm3FlowCfgBatch"/> selects batched versus two-forward CFG.</para>
/// <para>Several are <b>presence-only</b>: the call site tests <c>is null</c>, so ANY set value enables them —
/// including <c>0</c>. They are declared as <c>string?</c> so that stays true; forcing them into a boolean
/// grammar would make <c>=0</c> start meaning "off", which is a behavior change.</para></remarks>
public static partial class EngineKnobs
{
    /// <summary>Bypasses the Anima LLM adapter, swapping which conditioning path the model takes.</summary>
    public static readonly Knob<bool> AnimaBypassLlmAdapter =
        Bool("numerics.animaBypassLlmAdapter", false, KnobScope.Runtime, KnobDomain.Numerics,
            "Bypasses the Anima LLM adapter, swapping which conditioning path the model takes.");

    /// <summary>Seeds HiFTNet's noise source deterministically, which disables the NSF additive noise.</summary>
    public static readonly Knob<bool> HiftDeterministic =
        Bool("numerics.hiftDeterministic", false, KnobScope.Runtime, KnobDomain.Numerics,
            "Seeds HiFTNet's noise source deterministically, disabling the NSF additive noise.");

    /// <summary>Batched CFG for the MiniMax-Music3 flow pipeline; off runs two separate forwards.</summary>
    public static readonly Knob<bool> Mm3FlowCfgBatch =
        Bool("numerics.mm3FlowCfgBatch", true, KnobScope.Runtime, KnobDomain.Numerics,
            "Batched CFG for the MiniMax-Music3 flow pipeline; off runs two separate forwards.");

    /// <summary>Logs per-stage Anima tensor statistics.</summary>
    public static readonly Knob<bool> AnimaDebugStats =
        Bool("diagnostics.animaDebugStats", false, KnobScope.Runtime, KnobDomain.Diagnostics,
            "Logs per-stage Anima tensor statistics.");

    /// <summary>Logs per-stage Lens tensor statistics.</summary>
    public static readonly Knob<bool> LensDebugStats =
        Bool("diagnostics.lensDebugStats", false, KnobScope.Runtime, KnobDomain.Diagnostics,
            "Logs per-stage Lens tensor statistics.");

    /// <summary>Logs Qwen3 decode diagnostics.</summary>
    public static readonly Knob<bool> Qwen3Debug =
        Bool("diagnostics.qwen3Debug", false, KnobScope.Runtime, KnobDomain.Diagnostics,
            "Logs Qwen3 decode diagnostics.");

    /// <summary>Logs HunyuanVideo VAE stage boundaries.</summary>
    public static readonly Knob<bool> HyvVaeStages =
        Bool("diagnostics.hyvVaeStages", false, KnobScope.Runtime, KnobDomain.Diagnostics,
            "Logs HunyuanVideo VAE stage boundaries.");

    // ── Presence-only: any set value enables, including "0" ──

    /// <summary>Enables Ernie-Image diagnostics. Presence-only: any value enables it, including 0.</summary>
    public static readonly Knob<string?> ErnieDiag =
        Str("diagnostics.ernieDiag", null, KnobScope.Runtime, KnobDomain.Diagnostics,
            "Enables Ernie-Image diagnostics. Presence-only: any value enables it, including 0.");

    /// <summary>Enables F5-TTS mel diagnostics. Presence-only, and the value itself is never read.</summary>
    public static readonly Knob<string?> F5DumpMel =
        Str("diagnostics.f5DumpMel", null, KnobScope.Runtime, KnobDomain.Diagnostics,
            "Enables F5-TTS mel diagnostics. Presence-only; the value is never read.");

    /// <summary>Enables Dia token diagnostics.</summary>
    public static readonly Knob<string?> DiaDebugTokens =
        Str("diagnostics.diaDebugTokens", null, KnobScope.Runtime, KnobDomain.Diagnostics,
            "Enables Dia token diagnostics.");

    // ── Dump directories ──

    /// <summary>Directory for Flux debug dumps; created if missing.</summary>
    public static readonly Knob<string?> FluxDebugDir =
        Str("diagnostics.fluxDebugDir", null, KnobScope.Runtime, KnobDomain.Diagnostics,
            "Directory for Flux debug dumps; created if missing.");

    /// <summary>Directory for OmniGen2 debug dumps. Silently inactive unless the directory already exists.</summary>
    /// <remarks>Unlike <see cref="FluxDebugDir"/> this one is never created, so a typo disables dumping quietly.</remarks>
    public static readonly Knob<string?> Omnigen2DebugDir =
        Str("diagnostics.omnigen2DebugDir", null, KnobScope.Runtime, KnobDomain.Diagnostics,
            "Directory for OmniGen2 debug dumps; inactive unless the directory already exists, and never created.");

    /// <summary>File to dump Zonos logits to.</summary>
    public static readonly Knob<string?> ZonosLogitDump =
        Str("diagnostics.zonosLogitDump", null, KnobScope.Runtime, KnobDomain.Diagnostics,
            "File to dump Zonos logits to.");

    /// <summary>Filename tag for Wan debug dumps. Defaults to empty, and only has effect alongside a dump directory.</summary>
    /// <remarks>Empty and unset are deliberately identical: the call site tests <c>{ Length: &gt; 0 }</c> and appends a trailing underscore.</remarks>
    public static readonly Knob<string?> WanDebugTag =
        Str("diagnostics.wanDebugTag", "", KnobScope.Runtime, KnobDomain.Diagnostics,
            "Filename tag for Wan debug dumps; empty and unset are identical, and it needs a dump directory to matter.");
}
