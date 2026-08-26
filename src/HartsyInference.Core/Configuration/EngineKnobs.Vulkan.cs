namespace HartsyInference.Core.Configuration;

/// <summary>The Vulkan backend's own knobs, plus the engine's path roots and the audio-LM quantization choice.</summary>
/// <remarks>These carry a third prefix (<c>HARTSYINFERENCE_</c>) that neither the <c>HARTSY_</c> inventory nor the
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
        Bool("numerics.vkCoopmat2", "HARTSYINFERENCE_VK_COOPMAT2", true, BoolGrammar.Exact, KnobScope.Construction, KnobDomain.Numerics,
            "Kill-switch for the VK_NV_cooperative_matrix2 F16 GEMM fast path, which is tried first by default.");

    /// <summary>Force-disables the cooperative-matrix matmul path so GEMMs fall back to the scalar shaders.</summary>
    public static readonly Knob<bool> VkDisableCoopmat =
        Bool("numerics.vkDisableCoopmat", "HARTSYINFERENCE_VK_DISABLE_COOPMAT", false, BoolGrammar.Exact, KnobScope.Construction, KnobDomain.Numerics,
            "Force-disables the cooperative-matrix matmul path so GEMMs fall back to the scalar shaders.");

    /// <summary>Opts into the INT8 dot-product GEMM path for Linear; also requires device INT8 dot-product support.</summary>
    public static readonly Knob<bool> VkInt8 =
        Bool("numerics.vkInt8", "HARTSYINFERENCE_VK_INT8", false, BoolGrammar.Exact, KnobScope.Construction, KnobDomain.Numerics,
            "Opts into the INT8 dot-product GEMM path for Linear; also requires device INT8 dot-product support.");

    /// <summary>Uses VK_KHR_push_descriptor instead of the descriptor-pool ring when the extension is available.</summary>
    public static readonly Knob<bool> VkPushDescriptors =
        Bool("numerics.vkPushDescriptors", "HARTSYINFERENCE_VK_PUSH_DESCRIPTORS", false, BoolGrammar.Exact, KnobScope.Construction, KnobDomain.Numerics,
            "Uses VK_KHR_push_descriptor instead of the descriptor-pool ring when the extension is available.");

    /// <summary>Submits one command buffer per op instead of batching dispatches, for isolating a faulting op.</summary>
    public static readonly Knob<bool> VkSubmitPerOp =
        Bool("numerics.vkSubmitPerOp", "HARTSYINFERENCE_VK_SUBMIT_PER_OP", false, BoolGrammar.Exact, KnobScope.Construction, KnobDomain.Numerics,
            "Submits one command buffer per op instead of batching dispatches, for isolating a faulting op.");

    /// <summary>Disables the per-weight dtype-cast cache, trading re-cast cost for lower resident memory.</summary>
    public static readonly Knob<bool> VkNoWeightCastCache =
        Bool("vram.vkNoWeightCastCache", "HARTSYINFERENCE_VK_NO_WEIGHT_CAST_CACHE", false, BoolGrammar.Exact, KnobScope.Construction, KnobDomain.Vram,
            "Disables the per-weight dtype-cast cache, trading re-cast cost for lower resident memory.");

    /// <summary>Enables VK_LAYER_KHRONOS_validation when the layer is present.</summary>
    public static readonly Knob<bool> VkValidation =
        Bool("diagnostics.vkValidation", "HARTSYINFERENCE_VK_VALIDATION", false, BoolGrammar.Exact, KnobScope.Construction, KnobDomain.Diagnostics,
            "Enables VK_LAYER_KHRONOS_validation when the layer is present.");

    /// <summary>Enables per-op host-side Vulkan timing and buffer create/destroy accounting.</summary>
    public static readonly Knob<bool> VkProfile =
        Bool("diagnostics.vkProfile", "HARTSYINFERENCE_VK_PROFILE", false, BoolGrammar.Exact, KnobScope.Construction, KnobDomain.Diagnostics,
            "Enables per-op host-side Vulkan timing and buffer create/destroy accounting.");

    /// <summary>Logs every enumerated coopmat2 flexible-dimension config during Vulkan device setup.</summary>
    public static readonly Knob<bool> VkDumpCoopmat2 =
        Bool("diagnostics.vkDumpCoopmat2", "HARTSYINFERENCE_VK_DUMP_COOPMAT2", false, BoolGrammar.Exact, KnobScope.Construction, KnobDomain.Diagnostics,
            "Logs every enumerated coopmat2 flexible-dimension config during Vulkan device setup.");

    /// <summary>File to write the Vulkan op profile to; an unopenable path falls back to stderr at the call site.</summary>
    public static readonly Knob<string?> VkProfileFile =
        Str("diagnostics.vkProfileFile", "HARTSYINFERENCE_VK_PROFILE_FILE", null, KnobScope.Runtime, KnobDomain.Diagnostics,
            "File to write the Vulkan op profile to; an unopenable path falls back to stderr.");

    // ── Derived defaults: null means "no opinion", the call site keeps its own resolution ──

    /// <summary>Model root; unset resolves relative to the discovered repo root.</summary>
    public static readonly Knob<string?> ModelsRoot =
        Str("paths.modelsRoot", "HARTSYINFERENCE_MODELS", null, KnobScope.Runtime, KnobDomain.Paths,
            "Model root; unset resolves to Models/ under the discovered repo root.");

    /// <summary>Download cache root; unset resolves under the user cache directory.</summary>
    public static readonly Knob<string?> ModelCacheRoot =
        Str("paths.modelCacheRoot", "HARTSYINFERENCE_MODEL_CACHE", null, KnobScope.Construction, KnobDomain.Paths,
            "Download cache root; unset resolves under the user cache directory.");

    /// <summary>Generated-output root; unset resolves relative to the discovered repo root.</summary>
    public static readonly Knob<string?> OutputRoot =
        Str("paths.outputRoot", "HARTSYINFERENCE_OUTPUT", null, KnobScope.Runtime, KnobDomain.Paths,
            "Generated-output root; unset resolves to Output/ under the discovered repo root.");

    /// <summary>Repo root; unset walks up from the app base directory looking for the solution file.</summary>
    public static readonly Knob<string?> RepoRoot =
        Str("paths.repoRoot", "HARTSYINFERENCE_REPO_ROOT", null, KnobScope.Runtime, KnobDomain.Paths,
            "Repo root; unset walks up from the app base directory looking for the solution file.");

    /// <summary>Hunyuan3D debug dump directory.</summary>
    public static readonly Knob<string?> Hunyuan3dDebugDir =
        Str("diagnostics.hunyuan3dDebugDir", "HARTSYINFERENCE_HUNYUAN3D_DEBUG_DIR", null, KnobScope.Runtime, KnobDomain.Diagnostics,
            "Directory for Hunyuan3D debug dumps.");

    /// <summary>Audio-LM weight quantization (q4k / q8_0 / off); unset is Q4K on one device but Off when sharded.</summary>
    /// <remarks>Declared as an override precisely because that default is not a constant — baking one in would
    /// quantize sharded audio runs that deliberately stay unquantized.</remarks>
    public static readonly Knob<string?> AudioLmQuant =
        Str("numerics.audioLmQuant", "HARTSY_AUDIO_LM_QUANT", null, KnobScope.Construction, KnobDomain.Numerics,
            "Audio-LM weight quantization (q4k / q8_0 / off); unset is Q4K on one device, Off when sharded.");
}
