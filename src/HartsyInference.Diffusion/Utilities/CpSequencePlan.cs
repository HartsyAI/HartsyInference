namespace HartsyInference.Diffusion.Utilities;

/// <summary>One rank's contiguous slice of a context-parallel token sequence. Token order is raster, so a
/// group-aligned split keeps every per-group AdaLN modulation row whole on one rank (video frames) — or is simply
/// a clean row split when modulation is per-image broadcast (Qwen-Image packed rows).</summary>
public readonly record struct CpRankRange(int TokenStart, int TokenCount, int FrameStart, int FrameCount);

/// <summary>Group-aligned split of a DiT's token sequence across context-parallel ranks, largest-remainder
/// proportional to per-rank weights (free VRAM), 1-group floor per rank. The "frame" unit is whatever contiguous
/// token group the model splits on: Wan's latent frame groups (<c>gt = T/patch_t</c>, the AdaLN group granularity,
/// so slicing per-group modulation rows by the rank's frame range stays exact) or Qwen-Image's packed image rows
/// (<c>hPacked</c> groups of <c>wPacked</c> tokens — modulation there is per-image broadcast, so any row split is
/// exact and the row unit just keeps splits spatially clean).</summary>
public sealed class CpSequencePlan
{
    /// <summary>Per-rank token/frame ranges, ascending and contiguous over the whole sequence.</summary>
    public IReadOnlyList<CpRankRange> Ranks { get; }

    /// <summary>Total sequence length (<c>Frames · TokensPerFrame</c>).</summary>
    public int TotalTokens { get; }

    /// <summary>Tokens per latent frame group (<c>gh · gw</c>).</summary>
    public int TokensPerFrame { get; }

    /// <summary>Latent frame groups (<c>gt</c>).</summary>
    public int Frames { get; }

    private CpSequencePlan(IReadOnlyList<CpRankRange> ranks, int frames, int tokensPerFrame)
    {
        Ranks = ranks;
        Frames = frames;
        TokensPerFrame = tokensPerFrame;
        TotalTokens = frames * tokensPerFrame;
    }

    /// <summary>Splits <paramref name="frames"/> frame groups across <paramref name="weights"/>.Count ranks
    /// proportionally to the weights (non-positive total → even split). Requires frames ≥ ranks.</summary>
    public static CpSequencePlan Create(int frames, int tokensPerFrame, IReadOnlyList<double> weights)
    {
        ArgumentNullException.ThrowIfNull(weights);
        int rankCount = weights.Count;
        if (rankCount < 2)
        {
            throw new ArgumentException("Context parallelism needs at least 2 ranks.", nameof(weights));
        }
        if (frames < rankCount)
        {
            throw new ArgumentException(
                $"Cannot split {frames} latent frame groups across {rankCount} ranks — fewer frames than ranks.",
                nameof(frames));
        }

        double[] w = new double[rankCount];
        double total = 0;
        for (int i = 0; i < rankCount; i++)
        {
            w[i] = Math.Max(0, weights[i]);
            total += w[i];
        }
        if (total <= 0)
        {
            for (int i = 0; i < rankCount; i++)
            {
                w[i] = 1;
            }
            total = rankCount;
        }

        // Largest-remainder over frames with a 1-frame floor (the PlacementPlanner shape, local because
        // Diffusion cannot reference Engine).
        int[] counts = new int[rankCount];
        int assigned = 0;
        for (int i = 0; i < rankCount; i++)
        {
            counts[i] = Math.Max(1, (int)Math.Floor(frames * (w[i] / total)));
            assigned += counts[i];
        }
        int[] order = [.. Enumerable.Range(0, rankCount).OrderByDescending(i => w[i])];
        int guard = 0;
        while (assigned != frames && guard++ < 4 * frames)
        {
            foreach (int i in order)
            {
                if (assigned < frames)
                {
                    counts[i]++;
                    assigned++;
                }
                else if (assigned > frames && counts[i] > 1)
                {
                    counts[i]--;
                    assigned--;
                }
                if (assigned == frames)
                {
                    break;
                }
            }
        }

        CpRankRange[] ranks = new CpRankRange[rankCount];
        int frameStart = 0;
        for (int i = 0; i < rankCount; i++)
        {
            ranks[i] = new CpRankRange(frameStart * tokensPerFrame, counts[i] * tokensPerFrame, frameStart, counts[i]);
            frameStart += counts[i];
        }
        return new CpSequencePlan(ranks, frames, tokensPerFrame);
    }
}

/// <summary>Everything one context-parallel rank's DiT forward needs: which rank it is, the sequence split, and
/// the per-block K/V rendezvous. Null in a transformer forward = the byte-identical single-backend path.</summary>
public sealed record CpForwardContext
{
    /// <summary>This forward's rank index into <see cref="CpSequencePlan.Ranks"/>.</summary>
    public required int Rank { get; init; }

    /// <summary>The frame-aligned sequence split shared by all ranks of this generation.</summary>
    public required CpSequencePlan Plan { get; init; }

    /// <summary>The shared per-block self-attention K/V exchange.</summary>
    public required CpKvExchange Exchange { get; init; }
}
