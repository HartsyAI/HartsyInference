using System.ComponentModel;
using HartsyInference.Core.Backends;
using Spectre.Console.Cli;

namespace HartsyInference.Cli.Infra;

/// <summary>Shared multi-GPU placement options for the generation commands, mirroring the SwarmUI extension's
/// settings (TextEncoderGpuId/VaeGpuId/CfgParallelGpuId/DitShardGpuId) so the same placements are drivable from
/// the CLI for testing and headless operation. All numbers are CUDA ordinals (fastest-first — NOT nvidia-smi
/// order; run <c>nvidia-smi</c> during a generation to confirm which physical card an ordinal is).</summary>
public class PlacementCliSettings : CommandSettings
{
    [CommandOption("--gpu")]
    [Description("Primary CUDA device ordinal (default 0). Fastest-first ordering, not nvidia-smi order.")]
    public int? Gpu { get; init; }

    [CommandOption("--te-gpu")]
    [Description("Run the text encoders (CLIP/T5/umT5/…) on this CUDA ordinal instead of the primary GPU.")]
    public int? TextEncoderGpu { get; init; }

    [CommandOption("--vae-gpu")]
    [Description("Run the VAE encode/decode on this CUDA ordinal instead of the primary GPU.")]
    public int? VaeGpu { get; init; }

    [CommandOption("--cfg-parallel-gpu")]
    [Description("Run CFG's negative branch on this CUDA ordinal concurrently with the positive branch (weights REPLICATED — a latency win when the denoiser fits on both cards; silently falls back to sequential otherwise — check the [[CfgParallel]] log line). Mutually exclusive with --dit-shard-gpu.")]
    public int? CfgParallelGpu { get; init; }

    [CommandOption("--dit-shard-gpu")]
    [Description("Split the denoiser's block loop across the primary GPU and this CUDA ordinal (weights POOLED — a VRAM win for DiTs that don't fit one card, not a latency win). Mutually exclusive with --cfg-parallel-gpu.")]
    public int? DitShardGpu { get; init; }

    [CommandOption("--lm-shard-gpu")]
    [Description("Split large LMs' layers across the primary GPU and this CUDA ordinal (weights POOLED). Text models layer-split as with --device \"cuda:0+cuda:1\"; big audio LMs (YuE Stage-1) then default to UN-quantized checkpoint precision instead of Q4_K (override with HARTSY_AUDIO_LM_QUANT=q4k|q8|off). Implied by --dit-shard-gpu, which feeds the same shard list.")]
    public int? LmShardGpu { get; init; }
}

/// <summary>Builds the engine placement from CLI options with the same eager validation the extension does — a
/// bad ordinal fails at startup with a clear message, never mid-generation.</summary>
public static class PlacementCli
{
    /// <summary>Resolves (primary ordinal, EngineOptions) from placement settings; both null when no
    /// placement flag was passed (byte-identical single-GPU default path).</summary>
    public static (int? Gpu, EngineOptions? Options) Build(PlacementCliSettings settings, string backendSelector)
    {
        int primary = settings.Gpu ?? 0;
        string? teSelector = SelectorFor(settings.TextEncoderGpu, primary);
        string? vaeSelector = SelectorFor(settings.VaeGpu, primary);
        string? cfgSelector = SelectorFor(settings.CfgParallelGpu, primary);
        string? ditShardSelector = SelectorFor(settings.DitShardGpu, primary);
        string? lmShardSelector = SelectorFor(settings.LmShardGpu, primary);
        if (ditShardSelector is not null && lmShardSelector is not null && ditShardSelector != lmShardSelector)
        {
            throw new ArgumentException(
                "--dit-shard-gpu and --lm-shard-gpu point at different ordinals — they share one shard device "
                + "list, so pass just --dit-shard-gpu (it implies the LM split) or make them match.");
        }
        string? shardSelector = ditShardSelector ?? lmShardSelector;

        if (teSelector is null && vaeSelector is null && cfgSelector is null && shardSelector is null)
        {
            return (settings.Gpu, null);
        }

        PlacementConfig placement = new PlacementConfig
        {
            TextEncoderDevice = teSelector,
            VaeDevice = vaeSelector,
            CfgParallelDevice = cfgSelector,
            ShardDevices = shardSelector is not null
                ? new[] { BackendFactory.WithOrdinal(backendSelector, primary), shardSelector }
                : Array.Empty<string>(),
            EnableDitSharding = ditShardSelector is not null,
        };
        // Fail fast, mirroring the extension: the engine ctor re-validates via PlacementPlanner.ValidatePlacement
        // (which also rejects --cfg-parallel-gpu + --dit-shard-gpu together with an explanatory message).
        foreach (string? selector in new[] { teSelector, vaeSelector, cfgSelector, shardSelector })
        {
            if (selector is not null)
            {
                BackendFactory.Validate(selector);
            }
        }
        return (primary, new EngineOptions { Placement = placement });
    }

    private static string? SelectorFor(int? ordinal, int primary) =>
        ordinal.HasValue && ordinal.Value != primary ? BackendFactory.WithOrdinal("cuda", ordinal.Value) : null;
}
