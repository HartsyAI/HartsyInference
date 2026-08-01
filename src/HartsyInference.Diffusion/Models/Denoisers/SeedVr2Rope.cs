namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>SeedVR2 window-local 3-axis RoPE (mmrope3d): lang-style inverse frequencies (θ=10000, per-axis
/// rot dim 42 → 21 freqs, matching the checkpoint's dropped <c>rope.rope.freqs</c> buffers), axial
/// concatenation T‖H‖W → 126 rotated dims of the 128-dim head (last 2 pass through), interleaved-PAIR
/// rotation (rotary_embedding_torch convention: freqs repeated <c>(n r)</c>, r=2 — NeoX-style pairs, not
/// half-split). Video queries/keys use axial positions with the temporal axis OFFSET BY THE TEXT LENGTH
/// (<c>vid_freqs[l:l+f, :h, :w]</c>); text tokens use 1-D positions 0..l−1 with the same freqs replicated
/// across all three axes. Positions restart at zero inside every window — freqs are computed per window
/// shape, never globally. CPU-precomputed cos/sin tables; rotation applied on host F32 (bring-up; the DiT
/// runs rope on host between backend GEMMs, matching the reference's fp32 rope-in-attn).</summary>
public static class SeedVr2Rope
{
    /// <summary>Per-axis rotation dim (head_dim 128 / 3 axes → 42, floor).</summary>
    public const int AxisDim = 42;

    /// <summary>Total rotated dims (3 × 42); head dims beyond this pass through unrotated.</summary>
    public const int RotDim = 126;

    private static readonly double[] InvFreq = BuildInvFreq();

    private static double[] BuildInvFreq()
    {
        double[] inv = new double[AxisDim / 2];
        for (int i = 0; i < inv.Length; i++)
            inv[i] = 1.0 / Math.Pow(10000.0, 2.0 * i / AxisDim);
        return inv;
    }

    /// <summary>Builds interleaved cos/sin tables for one window: rows = t·h·w video tokens followed by
    /// <paramref name="txtLen"/> text tokens; cols = <see cref="RotDim"/>. Video token (ti,hi,wi) gets axis
    /// angles from positions (txtLen+ti, hi, wi); text token p gets angle(p) on all three axes.</summary>
    public static (float[] Cos, float[] Sin) BuildTables(int t, int h, int w, int txtLen)
    {
        int vidTokens = t * h * w;
        int rows = vidTokens + txtLen;
        float[] cos = new float[rows * RotDim];
        float[] sin = new float[rows * RotDim];

        for (int ti = 0; ti < t; ti++)
        for (int hi = 0; hi < h; hi++)
        for (int wi = 0; wi < w; wi++)
        {
            int row = (ti * h + hi) * w + wi;
            FillAxis(cos, sin, row, 0, txtLen + ti);
            FillAxis(cos, sin, row, 1, hi);
            FillAxis(cos, sin, row, 2, wi);
        }
        for (int p = 0; p < txtLen; p++)
        {
            int row = vidTokens + p;
            FillAxis(cos, sin, row, 0, p);
            FillAxis(cos, sin, row, 1, p);
            FillAxis(cos, sin, row, 2, p);
        }
        return (cos, sin);
    }

    private static void FillAxis(float[] cos, float[] sin, int row, int axis, int position)
    {
        int baseCol = row * RotDim + axis * AxisDim;
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

    /// <summary>Rotates rows of <paramref name="qk"/> (layout [rows, heads, 128]) in place using tables from
    /// <see cref="BuildTables"/>. Pair rotation: (x₀,x₁) → (x₀c−x₁s, x₁c+x₀s) per adjacent pair.</summary>
    public static void Apply(Span<float> qk, int rows, int heads, int headDim, float[] cos, float[] sin)
    {
        for (int r = 0; r < rows; r++)
        {
            int tblBase = r * RotDim;
            for (int hd = 0; hd < heads; hd++)
            {
                int rowBase = (r * heads + hd) * headDim;
                for (int i = 0; i < RotDim; i += 2)
                {
                    float c = cos[tblBase + i], s = sin[tblBase + i];
                    float x0 = qk[rowBase + i], x1 = qk[rowBase + i + 1];
                    qk[rowBase + i] = x0 * c - x1 * s;
                    qk[rowBase + i + 1] = x1 * c + x0 * s;
                }
            }
        }
    }
}
