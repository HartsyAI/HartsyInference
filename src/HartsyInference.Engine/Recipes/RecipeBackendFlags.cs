using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Cuda;
using HartsyInference.Vulkan;

namespace HartsyInference.Engine.Recipes;

/// <summary>Backend-flag propagation for recipes: any per-generation backend flag must be applied to EVERY member of
/// <see cref="RecipeContext.AllBackends"/>, not just <see cref="RecipeContext.Backend"/> — with a text encoder or VAE
/// placed on its own GPU, that device's backend needs the same flag or the protection silently doesn't apply there
/// (e.g. an fp8 encoder re-inflating its cast cache on the TE GPU). Callers keep their own trigger conditions; this
/// helper owns only the apply-to-all loop.</summary>
internal static class RecipeBackendFlags
{
    /// <summary>Sets <c>CacheWeightCasts = false</c> (weights stay checkpoint-dtype resident, transient per-GEMM
    /// dequant) on every placement backend. <paramref name="onlyWithoutNativeFp8Gemm"/> restricts the change to CUDA
    /// backends lacking native FP8 GEMM — the Flux rule, where SM 8.9+ hardware wants the cache kept on.</summary>
    public static void DisableCacheWeightCasts(RecipeContext context, string logTag, bool onlyWithoutNativeFp8Gemm = false)
    {
        foreach (IBackend backend in context.AllBackends)
        {
            if (backend is CudaBackend cudaBackend)
            {
                if (onlyWithoutNativeFp8Gemm && cudaBackend.EnableNativeFp8Gemm)
                {
                    continue;
                }
                cudaBackend.CacheWeightCasts = false;
                Logs.Info($"[{logTag}] CacheWeightCasts disabled on {backend.GetType().Name} (checkpoint-dtype resident, transient per-GEMM dequant).");
            }
            else if (backend is VulkanBackend vulkanBackend && !onlyWithoutNativeFp8Gemm)
            {
                vulkanBackend.CacheWeightCasts = false;
                Logs.Info($"[{logTag}] CacheWeightCasts disabled on {backend.GetType().Name} (checkpoint-dtype resident, transient per-GEMM dequant).");
            }
        }
    }
}
