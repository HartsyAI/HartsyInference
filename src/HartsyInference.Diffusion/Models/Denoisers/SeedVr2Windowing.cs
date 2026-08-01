namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Exact port of SeedVR2's window partition (reference <c>models/dit_v2/window.py</c>). Layers
/// alternate <see cref="MakeWindows"/> (even) and <see cref="MakeShiftedWindows"/> (odd) so information
/// crosses window boundaries — 3D Swin over NaViT-flattened tokens.
/// <para>Window COUNT is fixed by config (4,3,3); window SIZE scales with the input so the per-window token
/// budget stays near the 720p training regime: the grid is notionally resized to 45×80 area before dividing.
/// Slice emission order is width-outermost, then height, then time — the reference's list-comprehension
/// order, which the weight-free gather/scatter in attention depends on. Results are deterministic per
/// (grid, counts) and callers cache them across all 32 layers.</para></summary>
public static class SeedVr2Windowing
{
    private const double ReferenceArea = 45.0 * 80.0;
    private const int TemporalClamp = 30;

    /// <summary>Regular (unshifted) partition: tiles the grid exactly — every token in exactly one window.</summary>
    /// <param name="t">Patchified temporal length ((frames−1)/4+1 after VAE, /1 after patch).</param>
    /// <param name="h">Patchified rows (pixels/8/2).</param>
    /// <param name="w">Patchified cols (pixels/8/2).</param>
    /// <param name="numT">Nominal window count on T (config: 4).</param>
    /// <param name="numH">Nominal window count on H (config: 3).</param>
    /// <param name="numW">Nominal window count on W (config: 3).</param>
    public static SeedVr2WindowSlice[] MakeWindows(int t, int h, int w, int numT, int numH, int numW)
    {
        ValidateGrid(t, h, w, numT, numH, numW);
        (int wt, int wh, int ww) = WindowSize(t, h, w, numT, numH, numW);
        int nt = CeilDiv(t, wt);
        int nh = CeilDiv(h, wh);
        int nw = CeilDiv(w, ww);

        List<SeedVr2WindowSlice> slices = new List<SeedVr2WindowSlice>(nt * nh * nw);
        for (int iw = 0; iw < nw; iw++)
        {
            int w0 = iw * ww, w1 = Math.Min((iw + 1) * ww, w);
            if (w1 <= w0)
                continue;
            for (int ih = 0; ih < nh; ih++)
            {
                int h0 = ih * wh, h1 = Math.Min((ih + 1) * wh, h);
                if (h1 <= h0)
                    continue;
                for (int it = 0; it < nt; it++)
                {
                    int t0 = it * wt, t1 = Math.Min((it + 1) * wt, t);
                    if (t1 <= t0)
                        continue;
                    slices.Add(new SeedVr2WindowSlice(t0, t1, h0, h1, w0, w1));
                }
            }
        }
        return [.. slices];
    }

    /// <summary>Shifted partition: offsets by half a window on every axis longer than one window, adding one
    /// (ragged) window per shifted axis. An axis that fits in a single window is not shifted at all.</summary>
    public static SeedVr2WindowSlice[] MakeShiftedWindows(int t, int h, int w, int numT, int numH, int numW)
    {
        ValidateGrid(t, h, w, numT, numH, numW);
        (int wt, int wh, int ww) = WindowSize(t, h, w, numT, numH, numW);

        double st = wt < t ? 0.5 : 0.0;
        double sh = wh < h ? 0.5 : 0.0;
        double sw = ww < w ? 0.5 : 0.0;

        // Reference: `nt + 1 if st > 0 else 1` — an unshifted axis gets exactly ONE window, not ceil(t/wt).
        int nt = st > 0 ? (int)Math.Ceiling((t - st) / wt) + 1 : 1;
        int nh = sh > 0 ? (int)Math.Ceiling((h - sh) / wh) + 1 : 1;
        int nw = sw > 0 ? (int)Math.Ceiling((w - sw) / ww) + 1 : 1;

        List<SeedVr2WindowSlice> slices = new List<SeedVr2WindowSlice>(nt * nh * nw);
        for (int iw = 0; iw < nw; iw++)
        {
            // (int) truncates toward zero, matching Python int(); iw==0 yields a negative product clamped to 0.
            int w0 = Math.Max((int)((iw - sw) * ww), 0), w1 = Math.Min((int)((iw - sw + 1) * ww), w);
            if (w1 <= w0)
                continue;
            for (int ih = 0; ih < nh; ih++)
            {
                int h0 = Math.Max((int)((ih - sh) * wh), 0), h1 = Math.Min((int)((ih - sh + 1) * wh), h);
                if (h1 <= h0)
                    continue;
                for (int it = 0; it < nt; it++)
                {
                    int t0 = Math.Max((int)((it - st) * wt), 0), t1 = Math.Min((int)((it - st + 1) * wt), t);
                    if (t1 <= t0)
                        continue;
                    slices.Add(new SeedVr2WindowSlice(t0, t1, h0, h1, w0, w1));
                }
            }
        }
        return [.. slices];
    }

    /// <summary>Nominal window size shared by both partitions: spatial dims are first area-normalized to the
    /// 45×80 (720p latent) token budget — Python banker's rounding — then divided by the window counts;
    /// the temporal axis is clamped to 30 before dividing.</summary>
    private static (int Wt, int Wh, int Ww) WindowSize(int t, int h, int w, int numT, int numH, int numW)
    {
        double scale = Math.Sqrt(ReferenceArea / (h * (double)w));
        int resizedH = (int)Math.Round(h * scale, MidpointRounding.ToEven);
        int resizedW = (int)Math.Round(w * scale, MidpointRounding.ToEven);
        int wh = CeilDiv(resizedH, numH);
        int ww = CeilDiv(resizedW, numW);
        int wt = CeilDiv(Math.Min(t, TemporalClamp), numT);
        return (Math.Max(wt, 1), Math.Max(wh, 1), Math.Max(ww, 1));
    }

    private static void ValidateGrid(int t, int h, int w, int numT, int numH, int numW)
    {
        if (t <= 0 || h <= 0 || w <= 0)
            throw new ArgumentException($"Token grid must be positive, got ({t},{h},{w}).");
        if (numT <= 0 || numH <= 0 || numW <= 0)
            throw new ArgumentException($"Window counts must be positive, got ({numT},{numH},{numW}).");
    }

    private static int CeilDiv(int a, int b) => (a + b - 1) / b;
}
