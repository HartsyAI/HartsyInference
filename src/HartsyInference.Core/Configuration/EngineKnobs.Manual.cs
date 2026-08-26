namespace HartsyInference.Core.Configuration;

/// <summary>Knobs whose declaration could not be derived mechanically from their call site.</summary>
/// <remarks>Three shapes live here, all of which a naive <c>Knob&lt;T&gt;</c> with a constant default would get wrong:
/// <list type="bullet">
/// <item>values read raw and parsed by hand, so the default sits in the parsing expression rather than in an
/// <c>EnvSwitch</c> argument;</item>
/// <item>values that <b>clamp</b> rather than reject, where <c>17</c> means 16 and not the default;</item>
/// <item>values whose default is <b>contextual</b> — computed from GPU capability or the model's own config — which
/// are declared as nullable "override" knobs where <c>null</c> means "no opinion, keep the caller's default".</item>
/// </list>
/// Baking a constant for the contextual ones would be a real bug: <c>numerics.fp8Native</c> would turn FP8 on for
/// pre-Ada cards that cannot execute it.</remarks>
public static partial class EngineKnobs
{
    // ── Contextual defaults: null means "caller keeps its own default" ──

    /// <summary>Forces native FP8 tensor-core GEMMs on or off; unset follows the card (on for Ada SM 8.9+, off below).</summary>
    public static readonly Knob<bool?> Fp8Native =
        BoolOverride("numerics.fp8Native", "HARTSY_FP8_NATIVE", KnobScope.Runtime, KnobDomain.Numerics,
            "Forces native FP8 tensor-core GEMMs on or off; unset follows the card (on for Ada SM 8.9+, off below).");

    /// <summary>Forces LTX-2 two-stage (base + refine) sampling on or off; unset follows the checkpoint's config.</summary>
    public static readonly Knob<bool?> Ltx2TwoStage =
        BoolOverride("numerics.ltx2TwoStage", "HARTSY_LTX2_TWO_STAGE", KnobScope.Construction, KnobDomain.Numerics,
            "Forces LTX-2 two-stage (base + refine) sampling on or off; unset follows the checkpoint's config.");

    /// <summary>Forces the Euler-ancestral LTX-2 sampler on or off; unset follows the pipeline config.</summary>
    public static readonly Knob<bool?> Ltx2Ancestral =
        BoolOverride("numerics.ltx2Ancestral", "HARTSY_LTX2_ANCESTRAL", KnobScope.Runtime, KnobDomain.Numerics,
            "Forces the Euler-ancestral LTX-2 sampler on or off; unset follows the pipeline config.");

    /// <summary>Token count at which the LTX-2 sigma shift stops scaling; unset uses the config value.</summary>
    public static readonly Knob<int?> Ltx2ShiftMaxTokens =
        IntOverride("numerics.ltx2ShiftMaxTokens", "HARTSY_LTX2_SHIFT_MAX_TOKENS", KnobScope.Runtime, KnobDomain.Numerics,
            "Token count at which the LTX-2 sigma shift stops scaling; unset uses the config value.");

    /// <summary>Overrides the LTX-2 sigma shift; unset uses the config value.</summary>
    public static readonly Knob<float?> Ltx2Shift =
        FloatOverride("numerics.ltx2Shift", "HARTSY_LTX2_SHIFT", KnobScope.Runtime, KnobDomain.Numerics,
            "Overrides the LTX-2 sigma shift; unset uses the config value.");

    // ── Override knobs: a positive value replaces a computed decision ──

    /// <summary>Forces MiniMax-H3 attention/MLP chunk rows; unset lets the policy size chunks from free VRAM.</summary>
    public static readonly Knob<int?> H3ChunkRows =
        IntOverride("vram.h3ChunkRows", "HARTSY_H3_CHUNK_ROWS", KnobScope.Runtime, KnobDomain.Vram,
            "Forces MiniMax-H3 attention/MLP chunk rows; unset lets the policy size chunks from free VRAM.");

    /// <summary>Forces the Hunyuan3D shape-VAE decode chunk size; unset uses the caller's chunk size.</summary>
    public static readonly Knob<int?> Hy3dVaeChunk =
        IntOverride("vram.hy3dVaeChunk", "HARTSY_HY3D_VAE_CHUNK", KnobScope.Runtime, KnobDomain.Vram,
            "Forces the Hunyuan3D shape-VAE decode chunk size; unset uses the caller's chunk size.");

    /// <summary>Forces the HeartCodec scalar-decode chunk length in frames; unset uses the model's ChunkFrames.</summary>
    public static readonly Knob<int?> HeartcodecScalarChunk =
        IntOverride("vram.heartcodecScalarChunk", "HARTSY_HEARTCODEC_SCALAR_CHUNK", KnobScope.Runtime, KnobDomain.Vram,
            "Forces the HeartCodec scalar-decode chunk length in frames; unset uses the model's ChunkFrames.");

    /// <summary>Overrides the HiFTNet streaming margin in frames; unset lets the vocaler derive it from its receptive field.</summary>
    public static readonly Knob<int?> HiftStreamMargin =
        IntOverride("numerics.hiftStreamMargin", "HARTSY_HIFT_STREAM_MARGIN", KnobScope.Runtime, KnobDomain.Numerics,
            "Overrides the HiFTNet streaming margin in frames; unset derives it from the receptive field.");

    /// <summary>Overrides the flash-attention query tile (Br); the call site still floors it at the sequence length.</summary>
    public static readonly Knob<int?> SdpaTile =
        IntOverride("numerics.sdpaTile", "HARTSY_SDPA_TILE", KnobScope.Runtime, KnobDomain.Numerics,
            "Overrides the flash-attention query tile (Br); the call site still floors it at the sequence length.");

    /// <summary>Overrides the INT8 GEMM column-chunk width; a non-positive value means no chunking at all.</summary>
    /// <remarks>Non-positive maps to <c>int.MaxValue</c> at the call site, so the mapping stays there rather than in a coercion.</remarks>
    public static readonly Knob<int?> Int8NChunk =
        IntOverride("vram.int8NChunk", "HARTSY_INT8_N_CHUNK", KnobScope.Runtime, KnobDomain.Vram,
            "Overrides the INT8 GEMM column-chunk width; a non-positive value disables chunking.");

    /// <summary>Default-ON tier of DiT step-graph capture, for architectures where it is a validated win (host-issue-bound models such as Chroma).</summary>
    /// <remarks>Shares <c>HARTSY_DIT_GRAPH</c> with <see cref="DitGraph"/> on purpose, so <c>=0</c> kills both tiers
    /// and <c>=1</c> forces both while unset leaves each at its own per-architecture default. Declared here rather
    /// than generated because the inventory keys by environment name and collapses the pair to one entry.</remarks>
    public static readonly Knob<bool> DitGraphDefaultOn =
        Bool("numerics.ditGraphDefaultOn", "HARTSY_DIT_GRAPH", true, BoolGrammar.TriState, KnobScope.Runtime, KnobDomain.Numerics,
            "Default-ON tier of DiT step-graph capture, for architectures where the per-generation graph is a validated win.");

    // ── Knobs named after their model rather than the engine, so the first inventory's HARTSY_ prefix missed them ──

    /// <summary>Solver order for the Wan FlowUniPC multistep scheduler; non-positive falls back to 2.</summary>
    public static readonly Knob<int> WanSolverOrder =
        Int("numerics.wanSolverOrder", "WAN_SOLVER_ORDER", 2, KnobScope.Runtime, KnobDomain.Numerics,
            "Solver order for the Wan FlowUniPC multistep scheduler; non-positive falls back to 2.",
            v => v > 0 ? v : 2);

    /// <summary>Trims the Wan-S2V motion-frame prefix from the decoded clip. On unless explicitly set to 0.</summary>
    public static readonly Knob<bool> WanS2vTrim =
        Bool("numerics.wanS2vTrim", "WAN_S2V_TRIM", true, BoolGrammar.Exact, KnobScope.Runtime, KnobDomain.Numerics,
            "Trims the Wan-S2V motion-frame prefix from the decoded clip.");

    /// <summary>Writes LTX-Video per-step diagnostics to this file.</summary>
    public static readonly Knob<string?> LtxDiagFile =
        Str("diagnostics.ltxDiagFile", "LTX_DIAG_FILE", null, KnobScope.Runtime, KnobDomain.Diagnostics,
            "Writes LTX-Video per-step diagnostics to this file.");

    /// <summary>Enables LTX-Video per-step diagnostics on the console.</summary>
    public static readonly Knob<bool> LtxDiag =
        Bool("diagnostics.ltxDiag", "LTX_DIAG", false, BoolGrammar.Exact, KnobScope.Runtime, KnobDomain.Diagnostics,
            "Enables LTX-Video per-step diagnostics on the console.");

    // ── Constant defaults that live in a hand-written parsing expression ──

    /// <summary>Host-RAM floor in GB below which the audio cache evicts a prior model before loading the next.</summary>
    public static readonly Knob<long> AudioEvictBelowGb =
        Long("vram.audioEvictBelowGb", "HARTSY_AUDIO_EVICT_BELOW_GB", 14L, KnobScope.Runtime, KnobDomain.Vram,
            "Host-RAM floor in GB below which the audio cache evicts a prior model before loading the next.",
            v => v > 0 ? v : 14L);

    /// <summary>VRAM headroom in MB kept free when deciding whether to auto-promote a tensor to resident.</summary>
    public static readonly Knob<long> AutopromoteHeadroomMb =
        Long("vram.autopromoteHeadroomMb", "HARTSY_AUTOPROMOTE_HEADROOM_MB", 1536L, KnobScope.Runtime, KnobDomain.Vram,
            "VRAM headroom in MB kept free when deciding whether to auto-promote a tensor to resident.");

    /// <summary>Cap in MB on the im2col banding buffer for convolutions.</summary>
    public static readonly Knob<long> Im2colBandMb =
        Long("vram.im2colBandMb", "HARTSY_IM2COL_BAND_MB", 1024L, KnobScope.Runtime, KnobDomain.Vram,
            "Cap in MB on the im2col banding buffer for convolutions.",
            v => v > 0 ? v : 1024L);

    /// <summary>Budget in MB for INT8 resident row chunking; 0 leaves the chunker to size it.</summary>
    public static readonly Knob<long> Int8RowBudgetMb =
        Long("vram.int8RowBudgetMb", "HARTSY_INT8_ROW_BUDGET_MB", 0L, KnobScope.Runtime, KnobDomain.Vram,
            "Budget in MB for INT8 resident row chunking; 0 leaves the chunker to size it.",
            v => v > 0 ? v : 0L);

    /// <summary>Split-K factor for the GEMV kernels; -1 lets the kernel choose.</summary>
    public static readonly Knob<int> GemvKsplit =
        Int("numerics.gemvKsplit", "HARTSY_GEMV_KSPLIT", -1, KnobScope.Runtime, KnobDomain.Numerics,
            "Split-K factor for the GEMV kernels; -1 lets the kernel choose.");

    /// <summary>Rows per block for the warp-per-row GEMV kernels. Clamped to 1..16 — out of range means the bound, not the default.</summary>
    public static readonly Knob<int> GemvWpb =
        Int("numerics.gemvWpb", "HARTSY_GEMV_WPB", 4, KnobScope.Runtime, KnobDomain.Numerics,
            "Rows per block for the warp-per-row GEMV kernels; clamped to 1..16.",
            v => Math.Clamp(v, 1, 16));

    /// <summary>Size in MB of the CUDA-graph capture arena. Clamped to 8..2048.</summary>
    public static readonly Knob<long> GraphArenaMb =
        Long("vram.graphArenaMb", "HARTSY_GRAPH_ARENA_MB", 32L, KnobScope.Runtime, KnobDomain.Vram,
            "Size in MB of the CUDA-graph capture arena; clamped to 8..2048.",
            v => Math.Clamp(v, 8L, 2048L));

    /// <summary>Minimum K/V sequence length before the F16 SageAttention path is preferred; measured knee is 8192.</summary>
    public static readonly Knob<int> SageF16MinSkv =
        Int("numerics.sageF16MinSkv", "HARTSY_SAGE_F16_MIN_SKV", 8192, KnobScope.Runtime, KnobDomain.Numerics,
            "Minimum K/V sequence length before the F16 SageAttention path is preferred; measured knee is 8192.",
            v => v > 0 ? v : 8192);

    /// <summary>How many host-to-device transfers the H2D trace prints before going quiet.</summary>
    public static readonly Knob<int> H2dTraceLimit =
        Int("diagnostics.h2dTraceLimit", "HARTSY_H2D_TRACE_LIMIT", 24, KnobScope.Runtime, KnobDomain.Diagnostics,
            "How many host-to-device transfers the H2D trace prints before going quiet.");

    /// <summary>Minimum CLI log level; unparsable values fall back to Warning at the call site.</summary>
    /// <remarks>Kept as a raw string so the enum parse stays where the <c>LogLevel</c> type is visible; Core cannot see it.</remarks>
    public static readonly Knob<string?> LogLevel =
        Str("diagnostics.logLevel", "HARTSY_LOG_LEVEL", null, KnobScope.Runtime, KnobDomain.Diagnostics,
            "Minimum CLI log level; unparsable values fall back to Warning.");
}
