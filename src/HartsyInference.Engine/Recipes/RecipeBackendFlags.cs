using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;

namespace HartsyInference.Engine.Recipes;

/// <summary>Backend-flag propagation for recipes: any per-generation backend flag must be applied to EVERY member of
/// <see cref="RecipeContext.AllBackends"/>, not just <see cref="RecipeContext.Backend"/> — with a text encoder or VAE
/// placed on its own GPU, that device's backend needs the same flag or the protection silently doesn't apply there
/// (e.g. an fp8 encoder re-inflating its cast cache on the TE GPU). Callers keep their own trigger conditions; this
/// helper owns only the apply-to-all loop.</summary>
internal static class RecipeBackendFlags
{
    /// <summary>Sets <c>CacheWeightCasts = false</c> (weights stay checkpoint-dtype resident, transient per-GEMM
    /// dequant) on every placement backend. <paramref name="onlyWithoutNativeFp8Gemm"/> restricts the change to
    /// backends lacking a native FP8 GEMM — the Flux rule, where SM 8.9+ hardware wants the cache kept on.</summary>
    /// <remarks>Asks each backend what it is rather than testing its type. The type test named CUDA and Vulkan
    /// explicitly, so a third backend would have been skipped silently — the flag would appear to apply and the VRAM
    /// saving would not happen. A backend that does not cache casts takes the no-op default and is unaffected.</remarks>
    public static void DisableCacheWeightCasts(RecipeContext context, string logTag, bool onlyWithoutNativeFp8Gemm = false)
    {
        foreach (IBackend backend in context.AllBackends)
        {
            if (!backend.Device.IsGpu || (onlyWithoutNativeFp8Gemm && backend.NativeFp8Gemm))
            {
                continue;
            }
            backend.CacheWeightCasts = false;
            Logs.Info($"[{logTag}] CacheWeightCasts disabled on {backend.GetType().Name} (checkpoint-dtype resident, transient per-GEMM dequant).");
        }
    }
}
