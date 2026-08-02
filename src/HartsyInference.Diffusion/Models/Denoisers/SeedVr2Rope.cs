namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>SeedVR2 window-local 3-axis RoPE table builders for the <c>WanRopeInterleaved</c> kernel's
/// <c>[S, headDim]</c> interleaved layout (duplicated cos/sin per adjacent pair — NeoX-style, not
/// half-split; identical rotation math in both architectures). Positions restart at zero inside every
/// window; tables are built per token in GLOBAL token order by <see cref="SeedVr2DevicePlan"/>, with
/// identity fill (cos=1/sin=0) beyond the rotated dims so one full-head rope pass leaves them unrotated.
/// <para><b>v2 (3B, mmrope3d):</b> lang freqs θ=10000, per-axis rot dim 42 → 126 of 128 dims, integer
/// positions with the temporal axis OFFSET BY THE TEXT LENGTH (<c>vid_freqs[l:l+f, :h, :w]</c>), text
/// roped with 1-D positions on all three axes.</para>
/// <para><b>v1 (7B, models/dit):</b> pixel freqs <c>linspace(1,128,10)·π</c> (matches the checkpoint's
/// dropped <c>rope.rope.freqs</c>), per-axis rot dim 20 → 60 of 128 dims, positions normalized to
/// <c>linspace(−1,1)</c> over each window axis extent, VIDEO ONLY — text is never rotated.</para></summary>
public static class SeedVr2Rope
{
    /// <summary>v2 per-axis rotation dim (head_dim 128 / 3 axes → 42, floor).</summary>
    public const int AxisDim = 42;

    /// <summary>v2 total rotated dims (3 × 42); head dims beyond this pass through unrotated.</summary>
    public const int RotDim = 126;

    /// <summary>v1 per-axis rotation dim (10 pixel freqs, interleaved pairs).</summary>
    public const int AxisDimV1 = 20;

    /// <summary>v1 total rotated dims (3 × 20); dims 60..127 pass through.</summary>
    public const int RotDimV1 = 60;

    private static readonly double[] InvFreq = BuildInvFreq();
    private static readonly double[] PixelFreq = BuildPixelFreq();

    private static double[] BuildInvFreq()
    {
        double[] inv = new double[AxisDim / 2];
        for (int i = 0; i < inv.Length; i++)
            inv[i] = 1.0 / Math.Pow(10000.0, 2.0 * i / AxisDim);
        return inv;
    }

    private static double[] BuildPixelFreq()
    {
        double[] f = new double[AxisDimV1 / 2];
        for (int i = 0; i < f.Length; i++)
            f[i] = (1.0 + 127.0 * i / (f.Length - 1)) * Math.PI;   // linspace(1, 128, 10) · π
        return f;
    }

    /// <summary>Fills one v2 row at <paramref name="rowBase"/> with the three axis angles (integer T, H, W
    /// positions; the caller passes the text-length temporal offset).</summary>
    public static void FillRow(Span<float> cos, Span<float> sin, int rowBase, int posT, int posH, int posW)
    {
        FillAxisAt(cos, sin, rowBase, 0, posT);
        FillAxisAt(cos, sin, rowBase, 1, posH);
        FillAxisAt(cos, sin, rowBase, 2, posW);
    }

    /// <summary>Fills one v1 row: axis angles are <c>linspace(−1,1,extent)[index]·freq</c> per window axis.</summary>
    public static void FillRowV1(Span<float> cos, Span<float> sin, int rowBase,
        int idxT, int extentT, int idxH, int extentH, int idxW, int extentW)
    {
        FillAxisV1(cos, sin, rowBase, 0, NormalizedPos(idxT, extentT));
        FillAxisV1(cos, sin, rowBase, 1, NormalizedPos(idxH, extentH));
        FillAxisV1(cos, sin, rowBase, 2, NormalizedPos(idxW, extentW));
    }

    // torch.linspace(-1,1,steps=N): N==1 → [-1] (NOT 0 — a (steps-1) division would NaN or silently zero).
    private static double NormalizedPos(int index, int extent)
        => extent == 1 ? -1.0 : -1.0 + 2.0 * index / (extent - 1);

    private static void FillAxisAt(Span<float> cos, Span<float> sin, int rowBase, int axis, int position)
    {
        int baseCol = rowBase + axis * AxisDim;
        for (int i = 0; i < AxisDim / 2; i++)
        {
            double angle = position * InvFreq[i];
            float c = (float)Math.Cos(angle), s = (float)Math.Sin(angle);
            cos[baseCol + 2 * i] = c;
            cos[baseCol + 2 * i + 1] = c;
            sin[baseCol + 2 * i] = s;
            sin[baseCol + 2 * i + 1] = s;
        }
    }

    private static void FillAxisV1(Span<float> cos, Span<float> sin, int rowBase, int axis, double position)
    {
        int baseCol = rowBase + axis * AxisDimV1;
        for (int i = 0; i < AxisDimV1 / 2; i++)
        {
            double angle = position * PixelFreq[i];
            float c = (float)Math.Cos(angle), s = (float)Math.Sin(angle);
            cos[baseCol + 2 * i] = c;
            cos[baseCol + 2 * i + 1] = c;
            sin[baseCol + 2 * i] = s;
            sin[baseCol + 2 * i + 1] = s;
        }
    }
}
