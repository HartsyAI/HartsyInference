using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>The multi-speaker conditioning of a LongCat-Video-Avatar forward (<c>ref_target_masks</c> at the token grid,
/// meituan-longcat/LongCat-Video
/// @ 58eb336): how many audio tracks the audio tokens hold, and which tokens of the first latent frame belong to each region.</summary>
/// <remarks>The regions are, in order, the first speaker, the second speaker, the background, and one region per further person who stays silent.
/// The tracks are the first speaker's, the second speaker's and, when there are silent people, one silent track. The audio tokens of a frame hold
/// the tracks one after another, <c>[frames·tracks·contextTokens, outputDim]</c>. Build the masks with
/// <see cref="Pipelines.LongCat.LongCatSpeakerMasks"/>.</remarks>
public sealed class LongCatSpeakers
{
    /// <summary>Fewest audio tracks: the two speakers.</summary>
    public const int MinTracks = 2;

    /// <summary>Fewest regions: the two speakers and the background.</summary>
    public const int MinRegions = 3;

    private readonly float[] _masks;

    /// <summary>Creates the conditioning. <paramref name="masks"/> is <c>[regions, gridHeight·gridWidth]</c> of zeros and ones; it is
    /// copied.</summary>
    /// <exception cref="ArgumentException">A count is out of range, or the masks do not match it or hold other values.</exception>
    public LongCatSpeakers(int tracks, int regions, int gridHeight, int gridWidth, ReadOnlySpan<float> masks)
    {
        if (tracks < MinTracks)
            throw new ArgumentOutOfRangeException(nameof(tracks), tracks, $"Multi-speaker audio has at least {MinTracks} tracks.");
        if (regions < MinRegions)
            throw new ArgumentOutOfRangeException(nameof(regions), regions, $"Multi-speaker masks have at least {MinRegions} regions.");
        if (gridHeight < 1 || gridWidth < 1)
            throw new ArgumentOutOfRangeException(nameof(gridHeight), "The token grid must be positive.");
        if (masks.Length != (long)regions * gridHeight * gridWidth)
            throw new ArgumentException($"The masks hold {masks.Length} values; expected {regions}×{gridHeight}×{gridWidth}.", nameof(masks));
        foreach (float value in masks)
            if (value is not (0f or 1f))
                throw new ArgumentException("The token masks hold zeros and ones only.", nameof(masks));
        Tracks = tracks;
        Regions = regions;
        GridHeight = gridHeight;
        GridWidth = gridWidth;
        _masks = masks.ToArray();
    }

    /// <summary>Audio tracks in each frame's tokens.</summary>
    public int Tracks { get; }

    /// <summary>Regions the masks name.</summary>
    public int Regions { get; }

    /// <summary>Token rows of one latent frame.</summary>
    public int GridHeight { get; }

    /// <summary>Token columns of one latent frame.</summary>
    public int GridWidth { get; }

    /// <summary>The masks, <c>[Regions, GridHeight·GridWidth]</c>.</summary>
    public ReadOnlySpan<float> Masks => _masks;

    /// <summary>The scratch a multi-speaker block holds on the device beside its own activations, for a memory budget: the reference map's query
    /// tile and region values, and the query rotation tables of <paramref name="noisyFrames"/> latent frames.</summary>
    public static long ScratchBytes(LongCatConfig config, int gridHeight, int gridWidth, int noisyFrames)
    {
        ArgumentNullException.ThrowIfNull(config);
        long referenceTokens = (long)gridHeight * gridWidth, noisyTokens = noisyFrames * referenceTokens;
        long dim = config.HiddenSize, tile = Math.Min(LongCatBlock.DefaultTileRows, noisyTokens);
        // A query tile and its head-major copy, the attention output and its merged copy; the region values; the query cosines and sines.
        return sizeof(float) * (4 * tile * dim + referenceTokens * dim + 2 * noisyTokens * config.HeadDim);
    }
}
