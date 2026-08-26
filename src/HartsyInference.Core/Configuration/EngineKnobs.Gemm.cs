namespace HartsyInference.Core.Configuration;

/// <summary>GEMM and attention-kernel selection flags reached through the CUDA backend's <c>EnvFlag</c> helper.</summary>
/// <remarks>Declared as a group because they share one grammar exactly: <c>EnvFlag(name)</c> is
/// <c>GetEnvironmentVariable(name) == "1"</c>, so every one is <see cref="BoolGrammar.Exact"/> with a
/// <c>false</c> default and only a literal <c>1</c> turns it on.
/// <para>These were invisible to the inventory that generated the bulk of the registry: no
/// <c>GetEnvironmentVariable</c> call carries their names, only the helper does. That is the whole reason the
/// completeness scan now matches <c>EnvFlag(</c> as its own pattern.</para>
/// <para>Several are kill-switches for a default-ON behavior, so their <c>false</c> default means the feature
/// stays enabled — <c>numerics.noTf32</c> and <c>numerics.sdpaNoF16</c> read backwards on purpose, matching the
/// names the code already used.</para></remarks>
public static partial class EngineKnobs
{
    /// <summary>Opts into the experimental fused FlashAttention-2 kernel (TF32 tensor cores, F32 accumulate, MHA only).</summary>
    public static readonly Knob<bool> SdpaV2 =
        Bool("numerics.sdpaV2", "HARTSY_SDPA_V2", false, BoolGrammar.Exact, KnobScope.Runtime, KnobDomain.Numerics,
            "Opts into the experimental fused FlashAttention-2 kernel (TF32 tensor cores, F32 accumulate, MHA only).");

    /// <summary>Forces the FlashAttention path for every SDPA call rather than letting the backend choose.</summary>
    public static readonly Knob<bool> SdpaForceFlash =
        Bool("numerics.sdpaForceFlash", "HARTSY_SDPA_FORCE_FLASH", false, BoolGrammar.Exact, KnobScope.Runtime, KnobDomain.Numerics,
            "Forces the FlashAttention path for every SDPA call rather than letting the backend choose.");

    /// <summary>Forces F16 SDPA on for all callers; normally it is gated per call because unbounded-score architectures must not use it.</summary>
    public static readonly Knob<bool> SdpaF16 =
        Bool("numerics.sdpaF16", "HARTSY_SDPA_F16", false, BoolGrammar.Exact, KnobScope.Runtime, KnobDomain.Numerics,
            "Forces F16 SDPA on for all callers, overriding the per-call gate.");

    /// <summary>Kill-switch for F16 SDPA everywhere, including callers that opt in per call.</summary>
    public static readonly Knob<bool> SdpaNoF16 =
        Bool("numerics.sdpaNoF16", "HARTSY_SDPA_NO_F16", false, BoolGrammar.Exact, KnobScope.Runtime, KnobDomain.Numerics,
            "Kill-switch for F16 SDPA everywhere, including callers that opt in per call.");

    /// <summary>Forces split-K flash attention, and lowers the K/V length at which splitting engages to 8.</summary>
    public static readonly Knob<bool> FlashSplitForce =
        Bool("numerics.flashSplitForce", "HARTSY_FLASH_SPLIT_FORCE", false, BoolGrammar.Exact, KnobScope.Runtime, KnobDomain.Numerics,
            "Forces split-K flash attention and lowers the K/V length at which splitting engages to 8.");

    /// <summary>Kill-switch for split-K flash attention, which is otherwise a large LLM-decode win.</summary>
    public static readonly Knob<bool> FlashSplitOff =
        Bool("numerics.flashSplitOff", "HARTSY_FLASH_SPLIT_OFF", false, BoolGrammar.Exact, KnobScope.Runtime, KnobDomain.Numerics,
            "Kill-switch for split-K flash attention, which is otherwise a large LLM-decode win.");

    /// <summary>Accepts Sage attention's F32-to-F16 V-storage narrowing; only takes effect alongside an explicit Sage opt-in.</summary>
    /// <remarks>Quarantines a genuinely unsafe path: the Sage prologue materializes V as F16, so an F32 value
    /// outside the finite F16 range becomes infinity. Requires a second opt-in on purpose.</remarks>
    public static readonly Knob<bool> SageUnsafeF32VNarrow =
        Bool("numerics.sageUnsafeF32VNarrow", "HARTSY_SAGE_UNSAFE_F32_V_NARROW", false, BoolGrammar.Exact, KnobScope.Runtime, KnobDomain.Numerics,
            "Accepts Sage attention's unsafe F32-to-F16 V narrowing; only effective alongside an explicit Sage opt-in.");

    /// <summary>Enables the tensor-core GEMM path.</summary>
    public static readonly Knob<bool> TensorcoreGemm =
        Bool("numerics.tensorcoreGemm", "HARTSY_TENSORCORE_GEMM", false, BoolGrammar.Exact, KnobScope.Runtime, KnobDomain.Numerics,
            "Enables the tensor-core GEMM path.");

    /// <summary>Selects the high-precision GEMM path over the faster reduced-precision one.</summary>
    public static readonly Knob<bool> HighPrecisionGemm =
        Bool("numerics.highPrecisionGemm", "HARTSY_HIGH_PRECISION_GEMM", false, BoolGrammar.Exact, KnobScope.Runtime, KnobDomain.Numerics,
            "Selects the high-precision GEMM path over the faster reduced-precision one.");

    /// <summary>Enables the FP8 GEMM path with F16 accumulate.</summary>
    public static readonly Knob<bool> Fp8F16 =
        Bool("numerics.fp8F16", "HARTSY_FP8_F16", false, BoolGrammar.Exact, KnobScope.Runtime, KnobDomain.Numerics,
            "Enables the FP8 GEMM path with F16 accumulate.");

    /// <summary>Enables the FP8 GEMM path with F32 accumulate.</summary>
    public static readonly Knob<bool> Fp8F32 =
        Bool("numerics.fp8F32", "HARTSY_FP8_F32", false, BoolGrammar.Exact, KnobScope.Runtime, KnobDomain.Numerics,
            "Enables the FP8 GEMM path with F32 accumulate.");

    /// <summary>Opts out of TF32 tensor-core math for F32 GEMMs on Ampere+; TF32 is on by default there, matching PyTorch.</summary>
    public static readonly Knob<bool> NoTf32 =
        Bool("numerics.noTf32", "HARTSY_NO_TF32", false, BoolGrammar.Exact, KnobScope.Runtime, KnobDomain.Numerics,
            "Opts out of TF32 tensor-core math for F32 GEMMs on Ampere+, where it is on by default.");

    /// <summary>Enables the fast F16 GEMM path on Ampere+.</summary>
    public static readonly Knob<bool> GemmF16 =
        Bool("numerics.gemmF16", "HARTSY_GEMM_F16", false, BoolGrammar.Exact, KnobScope.Runtime, KnobDomain.Numerics,
            "Enables the fast F16 GEMM path on Ampere+.");
}
