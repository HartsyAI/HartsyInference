using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Logging;

namespace HartsyInference.Core.MemoryManagement;

/// <summary>Where a generation phase's weights will live for the duration of that phase.</summary>
public enum PhasePlacement
{
    /// <summary>Every weight is uploaded up front and stays resident — the fast path.</summary>
    Resident = 0,

    /// <summary>Weights ride a <see cref="BlockStreamingController"/> sliding window; peak VRAM is roughly the activation reserve plus a few blocks.</summary>
    Streamed = 1,
}

/// <summary>Decides, per generation phase, whether a component's weights fit resident or must be streamed — and says so
/// in the log with the numbers behind the decision.</summary>
/// <remarks>Before this existed, each pipeline hand-rolled the same query against
/// <see cref="IStreamingWeightCache.QueryAvailableWeightCacheBytes"/> with its own reserve estimate, and only two of
/// roughly twenty-five image pipelines did it at all — which is why a 12 GB card OOM'd on models ComfyUI streamed
/// happily. Centralizing it also puts the weights-vs-activations split in one place: streaming can only ever move the
/// weight term, so a phase whose peak is dominated by activations needs a smaller working set (tiling, workspace caps),
/// not a sliding window, and the log has to make that distinguishable.</remarks>
public sealed class VramPlanner
{
    private readonly IStreamingWeightCache? _cache;
    private readonly LowVramMode _mode;
    private readonly string _modelName;

    /// <summary>Creates a planner for one generation. Pass the backend's <see cref="IBackend.StreamingCache"/>; a null
    /// cache (CPU, Vulkan) forces <see cref="PhasePlacement.Resident"/> for every phase, matching today's behavior on
    /// backends that have no notion of a device weight cache.</summary>
    public VramPlanner(IStreamingWeightCache? cache, string modelName)
        : this(cache, modelName, LowVramPolicy.Resolve())
    {
    }

    /// <summary>Creates a planner with an explicit policy instead of the process-wide one.</summary>
    /// <remarks>Lets callers (and tests) pin a mode without mutating process environment state that every other
    /// concurrently-running generation would also see.</remarks>
    public VramPlanner(IStreamingWeightCache? cache, string modelName, LowVramMode mode)
    {
        ArgumentNullException.ThrowIfNull(modelName);
        _cache = cache;
        _modelName = modelName;
        _mode = mode;
    }

    /// <summary>True when this planner can produce a streamed plan at all.</summary>
    public bool CanStream => _cache is not null && _mode != LowVramMode.ForceOff;

    /// <summary>The resolved policy, for callers that log it.</summary>
    public LowVramMode Mode => _mode;

    /// <summary>Plans one phase and logs the decision with the weight/activation split behind it.</summary>
    /// <param name="phase">Phase name for the log, e.g. "text-encode", "denoise", "vae-decode".</param>
    /// <param name="weightBytes">Total bytes of the weights this phase needs resident.</param>
    /// <param name="activationReserveBytes">Bytes this phase's activations and workspace will need alongside those weights.</param>
    /// <param name="alreadyResident">True when these weights are ALREADY on the device (HARTSY_KEEP_MODELS). Must be
    /// honored: the availability query cannot see past weights that are themselves occupying the space it measures, so
    /// asking about a resident model reports "does not fit" and flips warm generations between resident and streamed on
    /// alternate runs.</param>
    /// <param name="canStream">Whether the caller's denoiser actually exposes an <see cref="IStreamingBlock"/>
    /// decomposition AND the caller is wired to drive it. Required rather than defaulted: as streaming is rolled out
    /// across denoisers, a pipeline that asks the planner before it has a streaming path would otherwise be handed
    /// <see cref="PhasePlacement.Streamed"/> and silently do the wrong thing.</param>
    public PhasePlacement PlanPhase(
        string phase,
        long weightBytes,
        long activationReserveBytes,
        bool alreadyResident,
        bool canStream)
    {
        ArgumentNullException.ThrowIfNull(phase);
        if (weightBytes < 0) throw new ArgumentOutOfRangeException(nameof(weightBytes));
        if (activationReserveBytes < 0) throw new ArgumentOutOfRangeException(nameof(activationReserveBytes));

        if (_mode == LowVramMode.ForceOff)
        {
            Logs.Info($"[VRAM] {_modelName}/{phase}: resident (HARTSY_LOWVRAM=off). {Describe(weightBytes, activationReserveBytes)}");
            return PhasePlacement.Resident;
        }
        if (alreadyResident)
        {
            Logs.Info($"[VRAM] {_modelName}/{phase}: resident (already on device, KEEP_MODELS). {Describe(weightBytes, activationReserveBytes)}");
            return PhasePlacement.Resident;
        }
        if (_cache is null || !canStream)
        {
            string why = _cache is null ? "backend has no streaming cache" : "denoiser exposes no streamable blocks";
            Logs.Info($"[VRAM] {_modelName}/{phase}: resident ({why}). {Describe(weightBytes, activationReserveBytes)}");
            return PhasePlacement.Resident;
        }

        long available = _cache.QueryAvailableWeightCacheBytes(activationReserveBytes);
        if (_mode == LowVramMode.ForceOn)
        {
            Logs.Info($"[VRAM] {_modelName}/{phase}: streamed (HARTSY_LOWVRAM=on forces it). " +
                $"{Describe(weightBytes, activationReserveBytes)}, available {Mb(available)}");
            return PhasePlacement.Streamed;
        }
        if (available >= weightBytes)
        {
            Logs.Info($"[VRAM] {_modelName}/{phase}: resident — fits. {Describe(weightBytes, activationReserveBytes)}, available {Mb(available)}");
            return PhasePlacement.Resident;
        }

        Logs.Info($"[VRAM] {_modelName}/{phase}: streamed — does not fit resident. " +
            $"{Describe(weightBytes, activationReserveBytes)}, available {Mb(available)}, short by {Mb(weightBytes - available)}");
        return PhasePlacement.Streamed;
    }

    /// <summary>True when an upcoming phase cannot get <paramref name="requiredBytes"/> of headroom without evicting
    /// what a previous phase left resident.</summary>
    /// <remarks>This is the cross-phase half of low-VRAM handling, and the half ComfyUI gets most of its 12 GB-card
    /// advantage from: peak VRAM spans text-encode → denoise → VAE-decode as <i>sequential</i> phases, so dropping the
    /// previous phase's weights is often the difference between fitting and OOM-ing. Returns false under
    /// <see cref="LowVramMode.ForceOff"/> — an operator who disabled low-VRAM handling wants the failure, not a silent
    /// eviction and re-upload.</remarks>
    public bool ShouldEvictForHeadroom(string phase, long requiredBytes)
    {
        ArgumentNullException.ThrowIfNull(phase);
        if (_cache is null || _mode == LowVramMode.ForceOff)
        {
            return false;
        }
        long available = _cache.QueryAvailableWeightCacheBytes(requiredBytes);
        if (available > 0)
        {
            return false;
        }
        Logs.Info($"[VRAM] {_modelName}/{phase}: evicting the previous phase's resident weights — needs {Mb(requiredBytes)} free.");
        return true;
    }

    /// <summary>Throws a diagnosable <see cref="OutOfVramException"/> for a phase that cannot fit even streamed.</summary>
    /// <remarks>The streamed working set is roughly the activation reserve plus the resident block window, so when
    /// <paramref name="minimumWorkingSetBytes"/> alone exceeds what the card has, no amount of weight streaming helps —
    /// the phase needs a smaller working set. Saying that explicitly is the difference between an actionable failure and
    /// the bare <c>CUDA_ERROR_OUT_OF_MEMORY</c> this replaces.</remarks>
    public void ThrowInfeasible(string phase, long minimumWorkingSetBytes)
    {
        ArgumentNullException.ThrowIfNull(phase);
        long available = _cache?.QueryAvailableWeightCacheBytes(0) ?? 0;
        throw new OutOfVramException(
            $"{_modelName}/{phase} cannot run on this device: it needs at least {Mb(minimumWorkingSetBytes)} of " +
            $"activations and workspace before any weights, but only {Mb(available)} is free. Weight streaming cannot " +
            "reduce this — lower the resolution, or use a device with more VRAM.");
    }

    /// <summary>Splits the phase's footprint into the part streaming can move (weights) and the part it cannot
    /// (activations plus workspace), so a log line distinguishes "needs a sliding window" from "needs a smaller tile".</summary>
    private static string Describe(long weightBytes, long activationReserveBytes)
        => $"weights {Mb(weightBytes)} + activations/workspace {Mb(activationReserveBytes)} = {Mb(weightBytes + activationReserveBytes)}";

    private static string Mb(long bytes) => $"{bytes / (1024 * 1024)} MB";
}
