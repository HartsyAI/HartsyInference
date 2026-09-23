using HartsyInference.Core.Backends;
using HartsyInference.Core.MemoryManagement;
using HartsyInference.Cuda;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Placement;
using HartsyInference.Engine.Planning.Memory;
using HartsyInference.Engine.Recipes;

namespace HartsyInference.Engine.Services;

/// <summary>Answers <see cref="IMemoryEstimationService"/> for one engine: estimates from cached checkpoint profiles,
/// fit judged against this engine's device, placement and effective VRAM policy.</summary>
/// <remarks><para>Capacity is TOTAL VRAM less <see cref="PlacementPlanner.PerDeviceReserveBytes"/>, never free VRAM:
/// free VRAM counts this engine's own evictable weight cache as used and moves with every allocation, so a routing
/// decision taken on it would change between two identical requests.</para>
/// <para>This class gathers the device facts; <see cref="MemoryFitJudge"/> turns them into the verdict. Only levers
/// the engine actually acts on earn capacity — block streaming when the model wires it and the
/// backend has a streaming cache, phase unload unless the policy pins it off, DiT sharding and component placement
/// when the model wires them. <see cref="MemorySupportReport"/> lists the rest as accepted-but-unconsumed; counting
/// them here would promise memory no code frees.</para></remarks>
internal sealed class MemoryEstimationService : IMemoryEstimationService
{
    /// <summary>Total VRAM per CUDA ordinal, probed once per process: shard devices are sized from this rather than
    /// by constructing their backends, which would open a device context just to ask a question.</summary>
    private static readonly Lazy<IReadOnlyDictionary<int, long>> CudaTotals = new(() =>
        CudaTopology.Probe().ToDictionary(gpu => gpu.Ordinal, gpu => gpu.TotalMemoryBytes));

    private readonly InferenceEngine _engine;

    /// <summary>The backend whose total was last read, and that total. A device's size never changes, so routing asks
    /// the driver once per backend instead of taking the backend's op scope for every queued request. A reference
    /// swapped whole, so a concurrent reader never sees one backend's total paired with another backend.</summary>
    private DeviceTotal? _deviceTotal;

    internal MemoryEstimationService(InferenceEngine engine) => _engine = engine;

    /// <inheritdoc/>
    public Task<MemoryEstimate> EstimateAsync(ModelSpec spec, MemoryEstimateRequest request,
        CancellationToken cancel = default)
    {
        cancel.ThrowIfCancellationRequested();
        return Task.FromResult(CheckpointMemoryProfile.For(spec).Estimate(request, static _ => true));
    }

    /// <inheritdoc/>
    public Task<MemoryFit> AssessAsync(ModelSpec spec, MemoryEstimateRequest request, CancellationToken cancel = default)
    {
        cancel.ThrowIfCancellationRequested();
        IBackend backend = _engine.Backend;
        VramPolicy policy = VramPolicyRegistry.Resolve(backend, request.Vram);
        long totalBytes = TotalBytes(backend);
        if (totalBytes <= 0)
        {
            return Task.FromResult(new MemoryFit
            {
                Verdict = MemoryFitVerdict.Unknown,
                EffectiveTier = policy.Tier,
                Reason = $"{backend.Capabilities.Name} does not report its memory.",
            });
        }

        CheckpointMemoryProfile profile = CheckpointMemoryProfile.For(spec);
        MemoryEstimate estimate = profile.Estimate(request, backend.SupportsResidentQuant);
        PlacementConfig placement = _engine.Placement;
        long primary = Math.Max(0, totalBytes - PlacementPlanner.PerDeviceReserveBytes);
        bool canStream = profile.Capabilities.HasFlag(MemoryCapabilities.BlockStreaming) && backend.StreamingCache is not null;
        return Task.FromResult(MemoryFitJudge.Judge(estimate, policy, canStream, primary,
            primary + PooledShardBytes(placement, profile.Capabilities), OnPrimary(placement, profile.Capabilities)));
    }

    /// <summary>Total VRAM of <paramref name="backend"/>, cached per backend instance (a SetBackend swap re-reads).</summary>
    private long TotalBytes(IBackend backend)
    {
        DeviceTotal? cached = Volatile.Read(ref _deviceTotal);
        if (cached is not null && ReferenceEquals(cached.Backend, backend))
        {
            return cached.TotalBytes;
        }
        long total = backend.GetVramInfo().TotalBytes;
        Volatile.Write(ref _deviceTotal, new DeviceTotal(backend, total));
        return total;
    }

    /// <summary>Which components count against the primary device: a text encoder or VAE placed on another device
    /// does not, provided the model actually honours component placement.</summary>
    private Func<MemoryComponent, bool> OnPrimary(PlacementConfig placement, MemoryCapabilities capabilities)
    {
        if (!capabilities.HasFlag(MemoryCapabilities.ComponentPlacement))
        {
            return static _ => true;
        }
        bool textEncoderAway = IsOtherDevice(placement.TextEncoderDevice);
        bool vaeAway = IsOtherDevice(placement.VaeDevice);
        return component => component switch
        {
            MemoryComponent.TextEncoder => !textEncoderAway,
            MemoryComponent.Vae => !vaeAway,
            _ => true,
        };
    }

    /// <summary>Usable VRAM the shard devices add to the denoiser's pool, when the model wires DiT sharding and it is
    /// enabled. Only CUDA shards are pooled, matching where sharding runs.</summary>
    private long PooledShardBytes(PlacementConfig placement, MemoryCapabilities capabilities)
    {
        if (!placement.EnableDitSharding || !capabilities.HasFlag(MemoryCapabilities.DitSharding))
        {
            return 0;
        }
        long pooled = 0;
        foreach (string device in placement.ShardDevices.Distinct(StringComparer.OrdinalIgnoreCase))
        {
            if (!IsOtherDevice(device) || BackendFactory.Kind(device) != "cuda")
            {
                continue;
            }
            if (CudaTotals.Value.TryGetValue(BackendFactory.ParseOrdinal(device), out long total))
            {
                pooled += Math.Max(0, total - PlacementPlanner.PerDeviceReserveBytes);
            }
        }
        return pooled;
    }

    private bool IsOtherDevice(string? selector) =>
        !string.IsNullOrWhiteSpace(selector)
        && !BackendFactory.CanonicalDeviceKey(selector).Equals(
            BackendFactory.CanonicalDeviceKey(_engine.BackendSelector), StringComparison.OrdinalIgnoreCase);

    /// <summary>One backend's total VRAM, immutable so it can be published with a single reference write.</summary>
    private sealed record DeviceTotal(IBackend Backend, long TotalBytes);
}
