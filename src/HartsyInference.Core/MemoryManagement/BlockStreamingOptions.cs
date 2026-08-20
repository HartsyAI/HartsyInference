using HartsyInference.Core.Backends;

namespace HartsyInference.Core.MemoryManagement;

/// <summary>How a <see cref="BlockStreamingScope"/> splits a denoiser's blocks between device-resident and streamed.</summary>
public enum BlockStreamingPolicy
{
    /// <summary>Park as many whole blocks as measured free VRAM allows and stream only the remainder. Degenerates to
    /// full residency when the whole set fits, so it subsumes <see cref="AllOrNothing"/>'s fast path while still
    /// keeping most blocks resident on a card that is merely short.</summary>
    ResidentPrefix = 0,

    /// <summary>Ask <see cref="VramPlanner.PlanPhase"/> for a resident/streamed verdict and honor it for the whole
    /// block set. The shape the Flux-family pipelines use today.</summary>
    AllOrNothing = 1,
}

/// <summary>Inputs to <see cref="BlockStreamingScope.Open"/>.</summary>
public sealed record BlockStreamingOptions
{
    /// <summary>Backend the denoise loop runs on.</summary>
    public required IBackend Backend { get; init; }

    /// <summary>Denoiser whose blocks are being placed; its <see cref="IStreamableDenoiser.BeforeBlockForward"/> is
    /// owned by the scope for its lifetime.</summary>
    public required IStreamableDenoiser Denoiser { get; init; }

    /// <summary>Model name for the log lines, e.g. <c>"LTX-2"</c>.</summary>
    public required string ModelName { get; init; }

    /// <summary>Bytes to keep free for activations, the prefetch window and pool slack. Scale it with the geometry:
    /// a flat margin lets the prefix eat the VRAM the activations need at exactly the sizes streaming exists for.</summary>
    public required long HeadroomBytes { get; init; }

    /// <summary>Placement policy; <see cref="BlockStreamingPolicy.ResidentPrefix"/> by default.</summary>
    public BlockStreamingPolicy Policy { get; init; } = BlockStreamingPolicy.ResidentPrefix;

    /// <summary>This generation's token count, compared against <see cref="ResidentPrefixPin.SizedTokens"/> to decide
    /// whether a pinned prefix still has enough headroom behind it. Ignored when <see cref="Pin"/> is null.</summary>
    public long TokenLoad { get; init; }

    /// <summary>Cross-generation prefix state; null re-sizes from scratch every generation (correct for a pipeline
    /// that frees its denoiser weights at the end of each one).</summary>
    public ResidentPrefixPin? Pin { get; init; }

    /// <summary>In-flight upload depth for the streamed suffix.</summary>
    public int PrefetchAhead { get; init; } = 2;

    /// <summary>Whether <see cref="BlockStreamingScope.EndStep"/> returns pool reservations to the driver. Turning
    /// this off is a deliberate choice a caller has to write down — the streamed path otherwise grows the pool by
    /// roughly a block per step (Ideogram 4 measured 4.1 → 11.6 GiB by step 17).</summary>
    public bool PerStepTrim { get; init; } = true;

    /// <summary>Explicit policy instead of the process/backend resolution; tests pin it without mutating the
    /// environment every concurrent generation also reads.</summary>
    public LowVramMode? Mode { get; init; }
}
