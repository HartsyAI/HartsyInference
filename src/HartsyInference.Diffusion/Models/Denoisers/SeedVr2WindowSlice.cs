namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Half-open [start, end) box over a patchified (T, H, W) token grid — one attention window in the
/// SeedVR2 NaDiT. Boundary windows are ragged (smaller than the nominal window size), so token count must be
/// computed per slice, never assumed uniform.</summary>
public readonly record struct SeedVr2WindowSlice(int T0, int T1, int H0, int H1, int W0, int W1)
{
    /// <summary>Frames covered on the temporal axis.</summary>
    public int Frames => T1 - T0;

    /// <summary>Rows covered on the height axis.</summary>
    public int Rows => H1 - H0;

    /// <summary>Columns covered on the width axis.</summary>
    public int Cols => W1 - W0;

    /// <summary>Video tokens inside this window (varies per window due to ragged boundaries).</summary>
    public int TokenCount => Frames * Rows * Cols;
}
