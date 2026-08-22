using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Engine.Features;

/// <summary>VAE precision policy: SDXL's AutoencoderKL overflows F16 (resnet activations exceed ±65504 → +Inf → NaN → black output), so this mirrors ComfyUI's allow-list of <c>[bf16, fp32]</c> — BF16 on Ampere+ (same byte count as F16 with F32-equivalent dynamic range), F32 everywhere else. Never F16.</summary>
public static class VaePrecisionHelper
{
    /// <summary>BF16 when the backend reports BF16 support (Ampere+ on CUDA, Vulkan with VK_KHR_shader_bfloat16), else F32.</summary>
    public static DType PreferredVaeDtype(IBackend backend)
    {
        ArgumentNullException.ThrowIfNull(backend);
        // HARTSY_VAE_F32=1 forces the F32 decode path. BF16 was adopted for speed and VRAM (LTX-2.5 conv decode
        // 3.94 -> 2.89 s) and validated at SSIM 0.9983-0.9986 on SHORT clips; this switch exists so that trade can
        // be re-tested against long, motion-heavy content, where SSIM is a poor detector of periodic texture loss.
        if (Environment.GetEnvironmentVariable("HARTSY_VAE_F32") == "1") return DType.F32;
        return backend.Capabilities.SupportsBF16 ? DType.BF16 : DType.F32;
    }

    /// <summary>Returns a new dictionary with every weight cast to <paramref name="targetDtype"/>; tensors already at the target are borrowed from the input, while each cast result is caller-owned. A failed partial cast disposes every result it created before propagating the original exception.</summary>
    public static Dictionary<string, Tensor> CastVaeWeights(IReadOnlyDictionary<string, Tensor> weights, DType targetDtype)
    {
        ArgumentNullException.ThrowIfNull(weights);
        Dictionary<string, Tensor> result = new Dictionary<string, Tensor>(weights.Count);
        HashSet<Tensor> borrowed = new(ReferenceEqualityComparer.Instance);
        foreach (KeyValuePair<string, Tensor> kvp in weights)
            borrowed.Add(kvp.Value);

        try
        {
            foreach (KeyValuePair<string, Tensor> kvp in weights)
                result[kvp.Key] = kvp.Value.DType == targetDtype ? kvp.Value : kvp.Value.CastTo(targetDtype);
            return result;
        }
        catch
        {
            HashSet<Tensor> disposed = new(ReferenceEqualityComparer.Instance);
            foreach (Tensor tensor in result.Values)
            {
                if (!borrowed.Contains(tensor) && disposed.Add(tensor))
                    tensor.Dispose();
            }
            throw;
        }
    }
}
