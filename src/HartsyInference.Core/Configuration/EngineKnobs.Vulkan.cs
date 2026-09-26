namespace HartsyInference.Core.Configuration;

/// <summary>The Vulkan backend's own knobs, plus the engine's path roots and the audio-LM quantization choice.</summary>
/// <remarks>These were missed by both the first inventory of the surface and the
/// <c>EnvFlag</c> sweep matched, so they reached the registry last.
/// <para>The path roots and <c>audio.lmQuant</c> are declared as nullable overrides because their defaults are
/// <b>derived</b>, not constant: the roots resolve relative to the discovered repo root, and the audio-LM
/// quantization is Q4K on a single device but Off when the model is sharded. A constant default would break
/// sharded audio runs.</para></remarks>
public static partial class EngineKnobs
{
    /// <summary>Kill-switch for the VK_NV_cooperative_matrix2 F16 GEMM fast path, which is tried first by default.</summary>
    /// <remarks>Default ON, so only a literal <c>0</c> disables it.</remarks>
    public static readonly Knob<bool> VkCoopmat2 =
        Bool("numerics.vkCoopmat2", true, KnobScope.Construction, KnobDomain.Numerics,
            "Kill-switch for the VK_NV_cooperative_matrix2 F16 GEMM fast path, which is tried first by default.");

    /// <summary>Force-disables the cooperative-matrix matmul path so GEMMs fall back to the scalar shaders.</summary>
    public static readonly Knob<bool> VkDisableCoopmat =
        Bool("numerics.vkDisableCoopmat", false, KnobScope.Construction, KnobDomain.Numerics,
            "Force-disables the cooperative-matrix matmul path so GEMMs fall back to the scalar shaders.");

    /// <summary>Opts into the INT8 dot-product GEMM path for Linear; also requires device INT8 dot-product support.</summary>
    public static readonly Knob<bool> VkInt8 =
        Bool("numerics.vkInt8", false, KnobScope.Construction, KnobDomain.Numerics,
            "Opts into the INT8 dot-product GEMM path for Linear; also requires device INT8 dot-product support.");

    /// <summary>A GEMM with a 16-bit-float, fp8 or GGUF operand computes in F16 even when the other operand and the output are F32 — the CUDA policy, which puts such a Linear on the cooperative-matrix kernels with a transient F16 cast of the activation. Off, the output's dtype decides for every operand.</summary>
    public static readonly Knob<bool> VkF16Gemm =
        Bool("numerics.vkF16Gemm", true, KnobScope.Construction, KnobDomain.Numerics,
            "A GEMM with a 16-bit-float, fp8 or GGUF operand computes in F16 (cooperative matrix) even when the other operand and the output are F32; off, the output's dtype decides.");

    /// <summary>Uses VK_KHR_push_descriptor instead of the descriptor-pool ring when the extension is available.</summary>
    public static readonly Knob<bool> VkPushDescriptors =
        Bool("numerics.vkPushDescriptors", false, KnobScope.Construction, KnobDomain.Numerics,
            "Uses VK_KHR_push_descriptor instead of the descriptor-pool ring when the extension is available.");

    /// <summary>Submits one command buffer per op instead of batching dispatches, for isolating a faulting op.</summary>
    public static readonly Knob<bool> VkSubmitPerOp =
        Bool("numerics.vkSubmitPerOp", false, KnobScope.Construction, KnobDomain.Numerics,
            "Submits one command buffer per op instead of batching dispatches, for isolating a faulting op.");

    /// <summary>Disables the per-weight dtype-cast cache, trading re-cast cost for lower resident memory.</summary>
    public static readonly Knob<bool> VkNoWeightCastCache =
        Bool("vram.vkNoWeightCastCache", false, KnobScope.Construction, KnobDomain.Vram,
            "Disables the per-weight dtype-cast cache, trading re-cast cost for lower resident memory.");

    /// <summary>Enables VK_LAYER_KHRONOS_validation when the layer is present.</summary>
    public static readonly Knob<bool> VkValidation =
        Bool("diagnostics.vkValidation", false, KnobScope.Construction, KnobDomain.Diagnostics,
            "Enables VK_LAYER_KHRONOS_validation when the layer is present.");

    /// <summary>Enables per-op host-side Vulkan timing and buffer create/destroy accounting.</summary>
    public static readonly Knob<bool> VkProfile =
        Bool("diagnostics.vkProfile", false, KnobScope.Construction, KnobDomain.Diagnostics,
            "Enables per-op host-side Vulkan timing and buffer create/destroy accounting.");

    /// <summary>Times every Vulkan dispatch on the GPU with timestamp queries and reports GPU time per op.</summary>
    public static readonly Knob<bool> VkProfileGpu =
        Bool("diagnostics.vkProfileGpu", false, KnobScope.Construction, KnobDomain.Diagnostics,
            "Times every Vulkan dispatch on the GPU with timestamp queries and reports GPU time per op.");

    /// <summary>Logs every enumerated coopmat2 flexible-dimension config during Vulkan device setup.</summary>
    public static readonly Knob<bool> VkDumpCoopmat2 =
        Bool("diagnostics.vkDumpCoopmat2", false, KnobScope.Construction, KnobDomain.Diagnostics,
            "Logs every enumerated coopmat2 flexible-dimension config during Vulkan device setup.");

    /// <summary>File to write the Vulkan op profile to; an unopenable path falls back to stderr at the call site.</summary>
    public static readonly Knob<string?> VkProfileFile =
        Str("diagnostics.vkProfileFile", null, KnobScope.Runtime, KnobDomain.Diagnostics,
            "File to write the Vulkan op profile to; an unopenable path falls back to stderr.");

    // ── Derived defaults: null means "no opinion", the call site keeps its own resolution ──

    /// <summary>Model root; unset resolves relative to the discovered repo root.</summary>
    public static readonly Knob<string?> ModelsRoot =
        Str("paths.modelsRoot", null, KnobScope.Runtime, KnobDomain.Paths,
            "Model root; unset resolves to Models/ under the discovered repo root.");

    /// <summary>Download cache root; unset resolves under the user cache directory.</summary>
    public static readonly Knob<string?> ModelCacheRoot =
        Str("paths.modelCacheRoot", null, KnobScope.Construction, KnobDomain.Paths,
            "Download cache root; unset resolves under the user cache directory.");

    /// <summary>Generated-output root; unset resolves relative to the discovered repo root.</summary>
    public static readonly Knob<string?> OutputRoot =
        Str("paths.outputRoot", null, KnobScope.Runtime, KnobDomain.Paths,
            "Generated-output root; unset resolves to Output/ under the discovered repo root.");

    /// <summary>Repo root; unset walks up from the app base directory looking for the solution file.</summary>
    public static readonly Knob<string?> RepoRoot =
        Str("paths.repoRoot", null, KnobScope.Runtime, KnobDomain.Paths,
            "Repo root; unset walks up from the app base directory looking for the solution file.");

    /// <summary>Hunyuan3D debug dump directory.</summary>
    public static readonly Knob<string?> Hunyuan3dDebugDir =
        Str("diagnostics.hunyuan3dDebugDir", null, KnobScope.Runtime, KnobDomain.Diagnostics,
            "Directory for Hunyuan3D debug dumps.");

    /// <summary>Audio-LM weight quantization (q4k / q8_0 / off); unset is Q4K on one device but Off when sharded.</summary>
    /// <remarks>Declared as an override precisely because that default is not a constant — baking one in would
    /// quantize sharded audio runs that deliberately stay unquantized.</remarks>
    public static readonly Knob<string?> AudioLmQuant =
        Str("numerics.audioLmQuant", null, KnobScope.Construction, KnobDomain.Numerics,
            "Audio-LM weight quantization (q4k / q8_0 / off); unset is Q4K on one device, Off when sharded.");
}
