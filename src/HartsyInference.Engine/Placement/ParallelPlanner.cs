using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Cuda;

namespace HartsyInference.Engine.Placement;

/// <summary>Inputs for one strategy suggestion. <see cref="ModelBytes"/> is the on-disk checkpoint size (0 = unknown → the planner assumes the model fits one card and only suggests latency/throughput strategies).</summary>
public sealed record ParallelPlanRequest
{
    public required Modality Modality { get; init; }
    public long ModelBytes { get; init; }
    public required IReadOnlyList<GpuTopologyInfo> Gpus { get; init; }
    public required IReadOnlyList<GpuLinkInfo> Links { get; init; }
}

/// <summary>A suggested placement plus the one-line reason that will be logged (greppable <c>[ParallelPlan]</c>). <see cref="Placement"/> is <see cref="PlacementConfig.Single"/> when the honest answer is "parallelism does not help here".</summary>
public sealed record ParallelPlan(PlacementConfig Placement, string Reason);

/// <summary>Topology-aware strategy selection: picks fit (sharding/layer-split) vs latency (context/tensor parallel, CFG-parallel) vs single from the MEASURED verdicts in <c>benchmarks/results/</c> and <c>docs/PARALLELISM_GUIDE.md</c>, not aspiration. The rules are deliberately conservative — every branch cites the measurement that justifies it, and anything unproven resolves to the safe choice with a reason the caller logs. Advisory only: callers apply the returned <see cref="PlacementConfig"/> themselves, and <see cref="PlacementPlanner.ValidatePlacement"/> still gets the final word.</summary>
public static class ParallelPlanner
{
    /// <summary>Weight-fit safety factor: activations, casts, and KV/step transients ride on top of the checkpoint bytes (measured 1.15-1.2× across the sharding campaign; 1.3 keeps margin).</summary>
    private const double FitOverhead = 1.3;

    /// <summary>Minimum SM-count ratio (slowest/fastest) for a pair to count as "balanced" — below this, every synchronous parallel step waits on the slow card (measured: the 4090+3060 pair at ~0.36 makes context parallelism lose at every geometry).</summary>
    private const double BalancedSmRatio = 0.8;

    /// <summary>Suggests a placement for the request's modality on the probed topology and logs the decision.</summary>
    public static ParallelPlan Suggest(ParallelPlanRequest request)
    {
        ParallelPlan plan = Decide(request);
        Logs.Info($"[ParallelPlan] {Describe(plan.Placement)} — {plan.Reason}");
        return plan;
    }

    private static ParallelPlan Decide(ParallelPlanRequest request)
    {
        IReadOnlyList<GpuTopologyInfo> gpus = request.Gpus;
        if (gpus.Count < 2)
        {
            return new ParallelPlan(PlacementConfig.Single, "single GPU visible — nothing to place across.");
        }

        // Order candidate devices fastest-first by SM count (CUDA ordinal 0 is already fastest-first on
        // healthy setups, but the probe is authoritative).
        List<GpuTopologyInfo> ordered = [.. gpus.OrderByDescending(g => g.SmCount)];
        string[] devices = [.. ordered.Select(g => $"cuda:{g.Ordinal}")];
        bool fitsPrimary = request.ModelBytes <= 0
            || request.ModelBytes * FitOverhead < ordered[0].FreeMemoryBytes;
        bool fitsEverySecondary = request.ModelBytes > 0
            && ordered.Skip(1).All(g => request.ModelBytes * FitOverhead < g.FreeMemoryBytes);
        double smRatio = (double)ordered[^1].SmCount / Math.Max(1, ordered[0].SmCount);
        bool balanced = smRatio >= BalancedSmRatio
            && ordered.All(g => g.CcMajor == ordered[0].CcMajor && g.CcMinor == ordered[0].CcMinor);
        // Fast fabric = every directed pair reports an NVLink-class link.
        bool fastLinks = request.Links.Count > 0 && request.Links.All(l => l.LikelyNvLink);
        string topo = $"{gpus.Count} GPUs, smRatio={smRatio:F2}, nvlink={(fastLinks ? "yes" : "no")}, "
            + $"fits-primary={(request.ModelBytes > 0 ? fitsPrimary.ToString() : "unknown")}";

        switch (request.Modality)
        {
            case Modality.Text:
                if (!fitsPrimary)
                {
                    // Fit: layer-split pools VRAM with one boundary copy per stage — the proven fallback on
                    // any link quality (Qwen3-32B 11.8 tok/s on the no-P2P pair).
                    return new ParallelPlan(
                        new PlacementConfig { ShardDevices = devices },
                        $"layer-split for FIT ({topo}) — model exceeds the primary card; boundary copies are cheap on any link.");
                }
                if (balanced && fastLinks)
                {
                    // Latency: TP's 2 all-reduces/layer are latency-bound — only worth it on NVLink-class
                    // links with matched cards. Validation still refuses architectures TP v1 can't tile.
                    return new ParallelPlan(
                        new PlacementConfig { TensorParallelDegree = gpus.Count, ShardDevices = devices },
                        $"tensor parallel degree {gpus.Count} for LATENCY ({topo}) — balanced NVLink-class pair; "
                        + "per-layer all-reduces are link-latency-bound and lose on PCIe (measured).");
                }
                return new ParallelPlan(PlacementConfig.Single,
                    $"single for text ({topo}) — model fits and TP on this fabric loses to link latency (measured); "
                    + "use per-GPU engines for throughput (1.71x measured, DataParallelServingEngineTests).");

            case Modality.Image:
            case Modality.Video:
                if (!fitsPrimary)
                {
                    return new ParallelPlan(
                        new PlacementConfig { ShardDevices = devices, EnableDitSharding = true },
                        $"DiT block sharding for FIT ({topo}) — model exceeds the primary card; VRAM pooling, "
                        + "not latency (per-step boundary hand-off measured cheap).");
                }
                if (balanced && fastLinks)
                {
                    // Latency: context parallelism replicates weights and splits the sequence — the win case
                    // is long sequences on balanced fast links; measured to LOSE at every geometry on the
                    // unbalanced no-P2P dev pair, hence gated on both conditions.
                    return new ParallelPlan(
                        new PlacementConfig { ContextParallelDevices = devices },
                        $"context parallelism for LATENCY ({topo}) — balanced NVLink-class pair; sequence split "
                        + "amortizes, K/V exchange rides the fast link.");
                }
                if (fitsEverySecondary && request.Modality == Modality.Video)
                {
                    // CFG-parallel replicates the model and runs cond/uncond concurrently — measured ~1.8-1.9×
                    // per-step on Wan-class video when the replica genuinely fits WITH headroom, measured
                    // 2.6× SLOWER on SDXL when it fit without headroom — hence video-only + strict fit here.
                    return new ParallelPlan(
                        new PlacementConfig { CfgParallelDevice = devices[1] },
                        $"CFG-branch parallelism for LATENCY ({topo}) — replica fits the second card with margin; "
                        + "measured per-step win on video CFG (falls back observably if preload fails).");
                }
                return new ParallelPlan(PlacementConfig.Single,
                    $"single for {request.Modality} ({topo}) — model fits; on this fabric replication/split "
                    + "strategies measured slower than one fast card.");

            default:
                return new ParallelPlan(PlacementConfig.Single,
                    $"single — no multi-GPU strategy is wired for {request.Modality} (audio uses layer-split via "
                    + "ShardDevices explicitly; see docs/PARALLELISM_GUIDE.md).");
        }
    }

    private static string Describe(PlacementConfig placement)
    {
        if (placement.TensorParallelDegree > 1) return $"tensor-parallel deg={placement.TensorParallelDegree}";
        if (placement.ContextParallelDevices.Count >= 2) return $"context-parallel [{string.Join(",", placement.ContextParallelDevices)}]";
        if (placement.EnableDitSharding) return $"dit-sharding [{string.Join(",", placement.ShardDevices)}]";
        if (placement.CfgParallelDevice is not null) return $"cfg-parallel uncond={placement.CfgParallelDevice}";
        if (placement.ShardDevices.Count >= 2) return $"layer-split [{string.Join(",", placement.ShardDevices)}]";
        return "single";
    }
}
