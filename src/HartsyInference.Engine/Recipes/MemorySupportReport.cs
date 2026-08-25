using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.MemoryManagement;

namespace HartsyInference.Engine.Recipes;

/// <summary>Reports, once per constructed pipeline, every memory or placement setting that is configured but cannot take effect — because the recipe does not wire it, or because the backend cannot do it at all.</summary>
/// <remarks>Driven centrally from the engine rather than from each recipe. The per-recipe call it replaces reached
/// 2 of 39 recipes, which meant the other 37 answered "configured but ignored" and "configured and working" with the
/// same silence. Making the engine ask, and the recipe declare, inverts that: a family nobody wired says so by
/// default.</remarks>
public static class MemorySupportReport
{
    /// <summary>Logs one line per configured-but-unhonoured capability for <paramref name="recipeName"/>.</summary>
    /// <param name="declared">What the recipe says it wires (<see cref="IArchitectureRecipe.MemorySupports"/>).</param>
    public static void Report(string recipeName, RecipeContext context, MemoryCapabilities declared)
    {
        ArgumentNullException.ThrowIfNull(recipeName);
        ArgumentNullException.ThrowIfNull(context);
        ReportResolvedPolicy(recipeName, context, declared);
        ReportPlacement(recipeName, context, declared);
        ReportPolicy(recipeName, context, declared);
        ReportBackend(recipeName, context, declared);
    }

    /// <summary>The one line that says what this generation is actually running under, and what this family can do about it.</summary>
    /// <remarks>Emitted even when nothing is misconfigured: the common support question is "is my setting even
    /// reaching the model", and an answer that only appears when something is wrong cannot settle it.</remarks>
    private static void ReportResolvedPolicy(string recipeName, RecipeContext context, MemoryCapabilities declared)
    {
        VramPolicy? policy = context.VramPolicy;
        (long free, long total) = context.Backend.GetVramInfo();
        string card = total > 0 ? $"{ByteFormat.Mb(free)} free of {ByteFormat.Mb(total)}" : "no VRAM report";
        Logs.Info($"[VRAM] {recipeName}: policy {policy?.Describe() ?? "inherited (env)"}, "
            + $"device {context.Backend.Device} ({card}), "
            + $"model supports {(declared == MemoryCapabilities.None ? "nothing" : declared.ToString())}.");
    }

    /// <summary>Placement devices the operator paid for (a whole second GPU) that this family will not touch.</summary>
    private static void ReportPlacement(string recipeName, RecipeContext context, MemoryCapabilities declared)
    {
        if (context.CpBackends is { Count: > 0 } && !declared.HasFlag(MemoryCapabilities.ContextParallel))
        {
            Warn(recipeName, "Context parallelism", "the generation runs single-GPU and the extra backend sits idle");
        }
        if (context.CfgParallelBackend is not null && !declared.HasFlag(MemoryCapabilities.CfgParallel))
        {
            Warn(recipeName, "CFG-parallel", "the unconditional branch runs sequentially on the primary backend");
        }
        if ((context.DitShardBackend is not null || context.DitShardBackends is { Count: > 0 })
            && !declared.HasFlag(MemoryCapabilities.DitSharding))
        {
            Warn(recipeName, "DiT sharding", "the whole denoiser runs on the primary backend, so VRAM is not pooled");
        }
        bool componentsPlaced = (context.TextEncoderBackend is not null && !ReferenceEquals(context.TextEncoderBackend, context.Backend))
            || (context.VaeBackend is not null && !ReferenceEquals(context.VaeBackend, context.Backend));
        if (componentsPlaced && !declared.HasFlag(MemoryCapabilities.ComponentPlacement))
        {
            Warn(recipeName, "Component placement", "the text encoder and VAE run on the primary backend anyway");
        }
    }

    /// <summary>VRAM levers the resolved policy asks for that this family cannot act on.</summary>
    /// <remarks>Only levers the policy EXPLICITLY pins are reported. Auto is the default everywhere, so warning about
    /// it would fire on every generation of every model and train the operator to ignore the channel.</remarks>
    private static void ReportPolicy(string recipeName, RecipeContext context, MemoryCapabilities declared)
    {
        VramPolicy? policy = context.VramPolicy;
        if (policy is null)
        {
            return;
        }
        if (policy.WeightStreaming == LeverState.On && !declared.HasFlag(MemoryCapabilities.BlockStreaming))
        {
            Warn(recipeName, "Weight streaming", "this denoiser exposes no streamable blocks, so every weight stays "
                + "resident and an oversized request will still fail");
        }
        if (policy.Caches == CachePrecision.Half && !declared.HasFlag(MemoryCapabilities.HalfPrecisionCaches))
        {
            Warn(recipeName, "Half-precision caches", "this family keeps its caches at full precision");
        }
        if (policy.QuantizedCompute == LeverState.On && !declared.HasFlag(MemoryCapabilities.QuantizedCompute))
        {
            Warn(recipeName, "Quantized compute", "this family dequantizes its weights regardless");
        }
        if (policy.ChunkScale < 1.0f && !declared.HasFlag(MemoryCapabilities.Chunking))
        {
            Warn(recipeName, "Chunk scaling", "this family does not chunk its work, so the activation peak is unchanged");
        }
        if (policy.PhaseUnload == LeverState.On && !declared.HasFlag(MemoryCapabilities.PhaseUnload))
        {
            Warn(recipeName, "Phase unload", "this family holds its components for the whole generation");
        }
    }

    /// <summary>Levers the BACKEND cannot perform whatever the recipe declares — the case a model-only check misses.</summary>
    /// <remarks>CPU and Vulkan expose no <see cref="IBackend.StreamingCache"/>, so an operator who picks a streaming
    /// tier there gets the resident layout no matter which model they load. Reporting it against the recipe alone
    /// would blame the model for a device limitation.</remarks>
    private static void ReportBackend(string recipeName, RecipeContext context, MemoryCapabilities declared)
    {
        VramPolicy? policy = context.VramPolicy;
        if (policy?.WeightStreaming != LeverState.On || context.Backend.StreamingCache is not null)
        {
            return;
        }
        // Only worth saying when the model could otherwise have honoured it; the recipe-side warning already covers
        // the other case, and two lines for one cause reads like two problems.
        if (declared.HasFlag(MemoryCapabilities.BlockStreaming))
        {
            Logs.Warning($"[{recipeName}] Weight streaming is requested and this model supports it, but the "
                + $"{context.Backend.Device} backend has no streaming weight cache — the weights stay resident. "
                + "Streaming needs the CUDA backend.");
        }
    }

    private static void Warn(string recipeName, string capability, string consequence)
        => Logs.Warning($"[{recipeName}] {capability} is configured but not wired for this model — {consequence}.");
}
