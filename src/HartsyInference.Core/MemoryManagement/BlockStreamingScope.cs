using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.MemoryManagement;

/// <summary>Owns one denoise loop's weight placement: decides resident-vs-streamed, sizes the resident prefix, preloads it, drives a <see cref="BlockStreamingController"/> over the remainder, and tears the hook down.</summary>
/// <remarks>The eight pipelines that streamed before this existed each hand-rolled the same twenty lines with a
/// different policy — a hard-coded 6 GB margin here, a planner call there, a resident prefix in a third — so a fix or
/// a measurement in one never reached the others, and a denoiser that had not been hand-wired (Wan) silently ignored
/// <c>HARTSY_LOWVRAM</c> entirely.
/// <para><see cref="VramPlanner"/> resolves the policy (<see cref="LowVramMode"/>) here rather than the split:
/// <see cref="VramPlanner.PlanPhase"/> measures <see cref="IStreamingWeightCache.QueryAvailableWeightCacheBytes"/>
/// while the prefix sizing measures <see cref="IBackend.FreeMemoryBytes"/> after a trim, and near the boundary the
/// two disagree. <see cref="BlockStreamingPolicy.AllOrNothing"/> is the branch that does defer to it.</para></remarks>
public sealed class BlockStreamingScope : IDisposable
{
    private readonly IStreamableDenoiser _denoiser;
    private readonly string _modelName;
    private readonly bool _perStepTrim;
    private readonly BlockStreamingController? _streamer;
    private bool _disposed;

    private BlockStreamingScope(BlockStreamingOptions options, int residentPrefixBlocks,
        BlockStreamingController? streamer)
    {
        _denoiser = options.Denoiser;
        _modelName = options.ModelName;
        _perStepTrim = options.PerStepTrim;
        _streamer = streamer;
        ResidentPrefixBlocks = residentPrefixBlocks;
    }

    /// <summary>Places <paramref name="options"/>'s denoiser on its backend and returns the scope that owns the placement until it is disposed.</summary>
    public static BlockStreamingScope Open(BlockStreamingOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        IBackend backend = options.Backend;
        IStreamableDenoiser denoiser = options.Denoiser;
        int blockCount = denoiser.BlockCount;
        IStreamingWeightCache? cache = backend.StreamingCache;
        VramPlanner planner = options.Mode is LowVramMode pinned
            ? new VramPlanner(cache, options.ModelName, pinned, backend)
            : new VramPlanner(cache, options.ModelName, backend);

        if (cache is null || !planner.CanStream || blockCount == 0)
        {
            backend.PreloadWeights(EnumerateAll(denoiser));
            Logs.Info($"[VRAM] {options.ModelName}/denoise: resident prefix {blockCount}, streamed 0 " +
                $"({(cache is null ? "backend has no streaming cache" : "the resolved policy forbids streaming")}).");
            return new BlockStreamingScope(options, blockCount, streamer: null);
        }

        IStreamingBlock[] blocks = new IStreamingBlock[blockCount];
        for (int b = 0; b < blockCount; b++) blocks[b] = denoiser.GetBlock(b);
        long blockBytes = blocks[0].EstimatedWeightBytes;

        // Both planners preload the shared weights themselves — the resident-prefix one has to do it before it
        // frees and re-sizes a pinned prefix, the all-or-nothing one after the planner's availability query.
        int prefix = options.Policy == BlockStreamingPolicy.AllOrNothing ? PlanAllOrNothing(options, planner, blocks)
            : PlanResidentPrefix(options, planner, blocks, blockBytes);

        if (prefix > 0)
        {
            backend.PreloadWeights(BlockRangeWeights(denoiser, 0, prefix));
        }
        if (prefix >= blockCount)
        {
            Logs.Info($"[VRAM] {options.ModelName}/denoise: {blockCount} blocks × {ByteFormat.Mb(blockBytes)}, " +
                $"resident prefix {blockCount}, streamed 0.");
            return new BlockStreamingScope(options, blockCount, streamer: null);
        }

        IStreamingBlock[] streamed = new IStreamingBlock[blockCount - prefix];
        Array.Copy(blocks, prefix, streamed, 0, streamed.Length);
        BlockStreamingController streamer = new BlockStreamingController(cache, streamed,
            prefetchAhead: options.PrefetchAhead, retainBehind: 0, backend: options.PerStepTrim ? backend : null);
        denoiser.BeforeBlockForward = i => { if (i >= prefix) streamer.BeforeBlockForward(i - prefix); };
        streamer.Prime();
        Logs.Info($"[VRAM] {options.ModelName}/denoise: {blockCount} blocks × {ByteFormat.Mb(blockBytes)}, " +
            $"resident prefix {prefix}{(options.Pin?.Resident == true ? " (persistent, no re-upload)" : "")}, " +
            $"streamed {streamed.Length} ({ByteFormat.Mb(streamed.Length * blockBytes)}/forward), " +
            $"prefetchAhead={options.PrefetchAhead}, headroom {ByteFormat.Mb(options.HeadroomBytes)}.");
        return new BlockStreamingScope(options, prefix, streamer);
    }

    /// <summary>Blocks kept device-resident for the whole loop; the streamed suffix starts here.</summary>
    public int ResidentPrefixBlocks { get; }

    /// <summary>Blocks riding the sliding window.</summary>
    public int StreamedBlocks => _denoiser.BlockCount - ResidentPrefixBlocks;

    /// <summary>True when any block is streamed.</summary>
    public bool Streaming => _streamer is not null;

    /// <summary>How many times <see cref="EndStep"/> has been called — the observable that makes a forgotten per-step trim a test failure rather than a VRAM climb nobody notices until step 17.</summary>
    public int StepsEnded { get; private set; }

    /// <summary>Shared weights plus the resident prefix's blocks — what a phase that has to displace the denoiser frees, and what it re-preloads afterwards.</summary>
    public IEnumerable<Tensor> EnumerateResidentWeights()
    {
        foreach (Tensor t in _denoiser.EnumerateSharedWeights()) yield return t;
        foreach (Tensor t in BlockRangeWeights(_denoiser, 0, ResidentPrefixBlocks)) yield return t;
    }

    /// <summary>Call at the end of every denoise step. Returns the streamed path's pool reservations to the driver unless the caller opted out via <see cref="BlockStreamingOptions.PerStepTrim"/>.</summary>
    public void EndStep()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        StepsEnded++;
        if (_perStepTrim)
        {
            _streamer?.TrimAfterStep();
        }
    }

    /// <summary>Unhooks the denoiser and evicts the streamed window. The resident prefix is left alone — whether it survives to the next generation is the pipeline's call, made through <see cref="ResidentPrefixPin"/>.</summary>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _denoiser.BeforeBlockForward = null;
        if (_streamer is null) return;
        if (_perStepTrim && StepsEnded == 0)
        {
            Logs.Warning($"[VRAM] {_modelName}/denoise streamed {StreamedBlocks} blocks but " +
                $"{nameof(EndStep)} was never called — the stream-ordered pool grows by roughly a block per step " +
                "without it. Call it at the end of every denoise step.");
        }
        _streamer.EvictAll();
        _streamer.Dispose();
    }

    /// <summary>The <see cref="BlockStreamingPolicy.ResidentPrefix"/> split: trim, measure, park what fits.</summary>
    private static int PlanResidentPrefix(BlockStreamingOptions options, VramPlanner planner,
        IStreamingBlock[] blocks, long blockBytes)
    {
        IBackend backend = options.Backend;
        IStreamableDenoiser denoiser = options.Denoiser;
        ResidentPrefixPin? pin = options.Pin;
        long headroom = options.HeadroomBytes;
        int blockCount = blocks.Length;
        bool forceStream = planner.Mode == LowVramMode.ForceOn;

        backend.PreloadWeights(denoiser.EnumerateSharedWeights());

        if (pin is not null && pin.Resident && (options.TokenLoad > pin.SizedTokens || forceStream))
        {
            // The prefix being freed may be exactly what a previous generation's captured graph baked pointers to
            // (LTX-2 replays one over this very scope), so the graph has to go before the memory does.
            VramGraphGuard.InvalidateBeforeRelease(backend);
            backend.FreeWeights(BlockRangeWeights(denoiser, 0, pin.PinnedBlocks));
            backend.TrimMemoryPool();
            pin.Resident = false;
            pin.PinnedBlocks = -1;
            Logs.Info($"[VRAM] {options.ModelName}/denoise: resident prefix released " + (forceStream
                ? "(the resolved policy forces streaming)."
                : $"for re-size (token load {options.TokenLoad} > sized {pin.SizedTokens})."));
        }
        // FreeMemoryBytes counts pool-retained blocks as USED, so an untrimmed read carries the previous phase's
        // transients and sizes the prefix against pressure that is not there.
        if (pin is null || !pin.Resident || pin.TailEvicted)
        {
            backend.TrimMemoryPool();
            if (pin is not null) pin.TailEvicted = false;
        }
        // Return before the sizing, not after it: a prefix sized here is never uploaded, so recording it on the pin
        // would leave a later Auto generation squeezing against a count that describes VRAM nobody parked in.
        if (forceStream)
        {
            return 0;
        }

        int prefix;
        if (pin is null)
        {
            prefix = ResidentPrefixSizing.Size(backend.FreeMemoryBytes(), headroom, blockBytes, blockCount);
        }
        else
        {
            if (pin.PinnedBlocks < 0 || options.TokenLoad > pin.SizedTokens)
            {
                pin.PinnedBlocks = ResidentPrefixSizing.Size(backend.FreeMemoryBytes(), headroom, blockBytes, blockCount);
                pin.SizedTokens = options.TokenLoad;
            }
            prefix = pin.PinnedBlocks;
            if (!pin.Resident && prefix > 0)
            {
                // Re-upload path: free VRAM can be tighter than the sizing generation saw. Squeeze THIS generation
                // to what fits but keep the pin, so a later one tops back up instead of re-sizing downward forever.
                int fit = ResidentPrefixSizing.Size(backend.FreeMemoryBytes(), headroom, blockBytes, blockCount);
                if (fit < prefix)
                {
                    Logs.Info($"[VRAM] {options.ModelName}/denoise: pinned prefix {prefix} squeezed to {fit} for " +
                        "this generation (free VRAM); pin kept — the next generation tops back up.");
                    prefix = fit;
                }
            }
        }
        return prefix;
    }

    /// <summary>The <see cref="BlockStreamingPolicy.AllOrNothing"/> split, straight off <see cref="VramPlanner"/>.</summary>
    private static int PlanAllOrNothing(BlockStreamingOptions options, VramPlanner planner, IStreamingBlock[] blocks)
    {
        IBackend backend = options.Backend;
        IStreamableDenoiser denoiser = options.Denoiser;
        ResidentPrefixPin? pin = options.Pin;

        // Evict BEFORE planning, never by reordering the planner: PlanPhase honours residency ahead of the force
        // because its availability query cannot see past weights occupying the space it measures. Leaving that order
        // alone and displacing the weights here is what makes a forced stream take effect on a warm generation
        // instead of silently staying resident. Mirrors PlanResidentPrefix's release-on-force branch.
        if (planner.Mode == LowVramMode.ForceOn && pin is not null && pin.Resident)
        {
            // AllOrNothing never records a partial count, so an unsized pin means the whole set is resident.
            int residentBlocks = pin.PinnedBlocks >= 0 ? pin.PinnedBlocks : blocks.Length;
            VramGraphGuard.InvalidateBeforeRelease(backend);
            backend.FreeWeights(BlockRangeWeights(denoiser, 0, residentBlocks));
            backend.TrimMemoryPool();
            pin.Resident = false;
            pin.PinnedBlocks = -1;
            Logs.Info($"[VRAM] {options.ModelName}/denoise: resident blocks released "
                + "(the resolved policy forces streaming).");
        }

        long totalBlockBytes = 0;
        foreach (IStreamingBlock block in blocks) totalBlockBytes += block.EstimatedWeightBytes;
        PhasePlacement placement = planner.PlanPhase("denoise", totalBlockBytes, options.HeadroomBytes,
            alreadyResident: pin?.Resident == true, canStream: true);
        backend.PreloadWeights(denoiser.EnumerateSharedWeights());
        return placement == PhasePlacement.Resident ? blocks.Length : 0;
    }

    private static IEnumerable<Tensor> EnumerateAll(IStreamableDenoiser denoiser)
    {
        foreach (Tensor t in denoiser.EnumerateSharedWeights()) yield return t;
        foreach (Tensor t in BlockRangeWeights(denoiser, 0, denoiser.BlockCount)) yield return t;
    }

    private static IEnumerable<Tensor> BlockRangeWeights(IStreamableDenoiser denoiser, int from, int to)
    {
        for (int b = from; b < to; b++)
        {
            foreach (Tensor t in denoiser.GetBlock(b).EnumerateWeights()) yield return t;
        }
    }

}
