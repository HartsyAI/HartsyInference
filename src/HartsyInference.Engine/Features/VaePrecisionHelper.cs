using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Engine.Features;

/// <summary>VAE precision policy: SDXL's AutoencoderKL overflows F16 (resnet activations exceed ±65504 → +Inf → NaN →
/// black output), so this mirrors ComfyUI's allow-list of <c>[bf16, fp32]</c> — BF16 on Ampere+ (same byte count as F16
/// with F32-equivalent dynamic range), F32 everywhere else. Never F16.</summary>
public static class VaePrecisionHelper
{
    /// <summary>BF16 when the backend reports BF16 support (Ampere+ on CUDA, Vulkan with VK_KHR_shader_bfloat16), else F32.</summary>
    public static DType PreferredVaeDtype(IBackend backend)
    {
        ArgumentNullException.ThrowIfNull(backend);
        return backend.Capabilities.SupportsBF16 ? DType.BF16 : DType.F32;
    }

    /// <summary>Returns a new dictionary with every weight cast to <paramref name="targetDtype"/>; tensors already at the target are referenced, not copied.</summary>
    public static Dictionary<string, Tensor> CastVaeWeights(IReadOnlyDictionary<string, Tensor> weights, DType targetDtype)
    {
        ArgumentNullException.ThrowIfNull(weights);
        Dictionary<string, Tensor> result = new Dictionary<string, Tensor>(weights.Count);
        foreach (KeyValuePair<string, Tensor> kvp in weights)
        {
            result[kvp.Key] = kvp.Value.DType == targetDtype ? kvp.Value : kvp.Value.CastTo(targetDtype);
        }
        return result;
    }
}
