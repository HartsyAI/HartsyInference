namespace HartsyInference.Core.Configuration;

/// <summary>GEMM and attention-kernel selection flags reached through the CUDA backend's <c>EnvFlag</c> helper.</summary>
/// <remarks>Declared as a group because they were all reached the same way before the migration: the CUDA
/// backend's <c>EnvFlag</c> helper, which meant a literal <c>1</c> and nothing else. Every one defaults to
/// <c>false</c>.
/// <para>These were invisible to the inventory that generated the bulk of the registry: no
/// <c>GetEnvironmentVariable</c> call carries their names, only the helper does. That is the whole reason the
/// completeness scan now matches <c>EnvFlag(</c> as its own pattern.</para>
/// <para>Several are kill-switches for a default-ON behavior, so their <c>false</c> default means the feature
/// stays enabled — <c>numerics.noTf32</c> and <c>numerics.sdpaNoF16</c> read backwards on purpose, matching the
/// names the code already used.</para></remarks>
public static partial class EngineKnobs
{
    /// <summary>Forces F16 SDPA on for all callers; normally it is gated per call because unbounded-score architectures must not use it.</summary>
    public static readonly Knob<bool> SdpaF16 =
        Bool("numerics.sdpaF16", false, KnobScope.Runtime, KnobDomain.Numerics,
            "Forces F16 SDPA on for all callers, overriding the per-call gate.");

    /// <summary>Kill-switch for F16 SDPA everywhere, including callers that opt in per call.</summary>
    public static readonly Knob<bool> SdpaNoF16 =
        Bool("numerics.sdpaNoF16", false, KnobScope.Runtime, KnobDomain.Numerics,
            "Kill-switch for F16 SDPA everywhere, including callers that opt in per call.");

    /// <summary>Forces split-K flash attention, and lowers the K/V length at which splitting engages to 8.</summary>
    public static readonly Knob<bool> FlashSplitForce =
        Bool("numerics.flashSplitForce", false, KnobScope.Runtime, KnobDomain.Numerics,
            "Forces split-K flash attention and lowers the K/V length at which splitting engages to 8.");

    /// <summary>Kill-switch for split-K flash attention, which is otherwise a large LLM-decode win.</summary>
    public static readonly Knob<bool> FlashSplitOff =
        Bool("numerics.flashSplitOff", false, KnobScope.Runtime, KnobDomain.Numerics,
            "Kill-switch for split-K flash attention, which is otherwise a large LLM-decode win.");

    /// <summary>Enables the tensor-core GEMM path.</summary>
    public static readonly Knob<bool> TensorcoreGemm =
        Bool("numerics.tensorcoreGemm", false, KnobScope.Runtime, KnobDomain.Numerics,
            "Enables the tensor-core GEMM path.");

    /// <summary>Selects the high-precision GEMM path over the faster reduced-precision one.</summary>
    public static readonly Knob<bool> HighPrecisionGemm =
        Bool("numerics.highPrecisionGemm", false, KnobScope.Runtime, KnobDomain.Numerics,
            "Selects the high-precision GEMM path over the faster reduced-precision one.");

    /// <summary>Enables the FP8 GEMM path with F16 accumulate.</summary>
    public static readonly Knob<bool> Fp8F16 =
        Bool("numerics.fp8F16", false, KnobScope.Runtime, KnobDomain.Numerics,
            "Enables the FP8 GEMM path with F16 accumulate.");

    /// <summary>Enables the FP8 GEMM path with F32 accumulate.</summary>
    public static readonly Knob<bool> Fp8F32 =
        Bool("numerics.fp8F32", false, KnobScope.Runtime, KnobDomain.Numerics,
            "Enables the FP8 GEMM path with F32 accumulate.");

    /// <summary>Opts out of TF32 tensor-core math for F32 GEMMs on Ampere+; TF32 is on by default there, matching PyTorch.</summary>
    public static readonly Knob<bool> NoTf32 =
        Bool("numerics.noTf32", false, KnobScope.Runtime, KnobDomain.Numerics,
            "Opts out of TF32 tensor-core math for F32 GEMMs on Ampere+, where it is on by default.");

    /// <summary>Runs CUDA's F32 attention GEMMs (the materialized and query-tiled SDPA paths) in plain FP32 instead of TF32;
    /// <c>numerics.noTf32</c> does not reach them.</summary>
    public static readonly Knob<bool> SdpaNoTf32 =
        Bool("numerics.sdpaNoTf32", false, KnobScope.Construction, KnobDomain.Numerics,
            "Runs CUDA's F32 attention GEMMs in plain FP32 instead of TF32: the exact attention arm.");

    /// <summary>Enables the fast F16 GEMM path on Ampere+.</summary>
    public static readonly Knob<bool> GemmF16 =
        Bool("numerics.gemmF16", false, KnobScope.Runtime, KnobDomain.Numerics,
            "Enables the fast F16 GEMM path on Ampere+.");
}
