using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Device-resident constants for one SeedVR2 DiT geometry (latent grid + text length): per-parity
/// window-gather index tensors (I32, grouped by identical window shape for batched SDPA) and interleaved
/// rope cos/sin tables in GLOBAL token order (each token's angles are its position INSIDE its window; the
/// unrotated head dims beyond <see cref="SeedVr2Rope.RotDim"/> carry cos=1/sin=0 so one
/// <c>WanRopeInterleaved</c> pass covers the full 128-dim head). Preloaded to the GPU weight cache — every
/// chunk of a clip shares the geometry, so the build amortizes across the whole restore.</summary>
public sealed class SeedVr2DevicePlan : IDisposable
{
    /// <summary>One same-shape window group: gather/scatter row indices into <c>[vid ; txt]</c> (length
    /// WindowCount·(WindowTokens+TxtLen)), window token count excluding text.</summary>
    public sealed record ShapeGroup(Tensor Indices, int WindowCount, int WindowTokens);

    /// <summary>One partition parity (regular or shifted): its shape groups, total window count, and the
    /// global-token-order rope tables <c>[Lv, headDim]</c>.</summary>
    public sealed record Partition(ShapeGroup[] Groups, int WindowCount, Tensor VidCos, Tensor VidSin);

    /// <summary>Regular (even blocks) and shifted (odd blocks) partitions.</summary>
    public Partition[] Partitions { get; }

    /// <summary>Text rope tables <c>[Lt, headDim]</c> (positions 0..Lt−1 on all three axes — window-shape
    /// independent, shared by both parities). Null on the v1 architecture, which never rotates text.</summary>
    public Tensor? TxtCos { get; }

    /// <summary>Sin table companion to <see cref="TxtCos"/>.</summary>
    public Tensor? TxtSin { get; }

    /// <summary>Geometry key: (gridT, gridH, gridW, txtLen).</summary>
    public (int T, int H, int W, int Lt) Key { get; }

    private readonly List<Tensor> _owned = [];
    private bool _disposed;

    /// <summary>Builds the plan for one geometry and preloads all tensors to the backend weight cache.</summary>
    public unsafe SeedVr2DevicePlan(IBackend backend, SeedVr2Config config, int gridT, int gridH, int gridW, int txtLen)
    {
        Key = (gridT, gridH, gridW, txtLen);
        int lv = gridT * gridH * gridW;
        int headDim = config.HeadDim;
        (int nT, int nH, int nW) = config.WindowCounts;

        Partitions = new Partition[2];
        for (int parity = 0; parity < 2; parity++)
        {
            SeedVr2WindowSlice[] windows = parity == 1
                ? SeedVr2Windowing.MakeShiftedWindows(gridT, gridH, gridW, nT, nH, nW)
                : SeedVr2Windowing.MakeWindows(gridT, gridH, gridW, nT, nH, nW);

            Tensor vidCos = new Tensor(new TensorShape(lv, headDim), DType.F32);
            Tensor vidSin = new Tensor(new TensorShape(lv, headDim), DType.F32);
            if (config.PixelRope)
                FillVidTablesV1(vidCos, vidSin, windows, gridH, gridW, headDim);
            else
                FillVidTables(vidCos, vidSin, windows, gridH, gridW, txtLen, headDim);

            List<ShapeGroup> groups = [];
            foreach (IGrouping<(int F, int R, int C), SeedVr2WindowSlice> group in
                windows.GroupBy(s => (s.Frames, s.Rows, s.Cols)))
            {
                SeedVr2WindowSlice[] grp = [.. group];
                int winTokens = group.Key.F * group.Key.R * group.Key.C;
                int s = winTokens + txtLen;
                Tensor idx = new Tensor(new TensorShape((long)grp.Length * s), DType.I32);
                int* ip = (int*)idx.DataPointer;
                int pos = 0;
                foreach (SeedVr2WindowSlice slice in grp)
                {
                    for (int t = slice.T0; t < slice.T1; t++)
                    for (int h = slice.H0; h < slice.H1; h++)
                    for (int w = slice.W0; w < slice.W1; w++)
                        ip[pos++] = (t * gridH + h) * gridW + w;
                    for (int p = 0; p < txtLen; p++)
                        ip[pos++] = lv + p;
                }
                groups.Add(new ShapeGroup(idx, grp.Length, winTokens));
                _owned.Add(idx);
            }
            Partitions[parity] = new Partition([.. groups], windows.Length, vidCos, vidSin);
            _owned.Add(vidCos);
            _owned.Add(vidSin);
        }

        if (!config.PixelRope)
        {
            TxtCos = new Tensor(new TensorShape(txtLen, headDim), DType.F32);
            TxtSin = new Tensor(new TensorShape(txtLen, headDim), DType.F32);
            FillTxtTables(TxtCos, TxtSin, txtLen, headDim);
            _owned.Add(TxtCos);
            _owned.Add(TxtSin);
        }

        backend.PreloadWeights(_owned);
    }

    private static void FillVidTables(Tensor cosT, Tensor sinT, SeedVr2WindowSlice[] windows,
        int gridH, int gridW, int txtLen, int headDim)
    {
        Span<float> cos = cosT.AsSpan<float>();
        Span<float> sin = sinT.AsSpan<float>();
        InitIdentity(cos, sin, headDim);
        foreach (SeedVr2WindowSlice slice in windows)
        {
            for (int t = slice.T0; t < slice.T1; t++)
            for (int h = slice.H0; h < slice.H1; h++)
            for (int w = slice.W0; w < slice.W1; w++)
            {
                int token = (t * gridH + h) * gridW + w;
                // Positions restart per window; temporal axis is offset by the text length (§2.5).
                SeedVr2Rope.FillRow(cos, sin, token * headDim, txtLen + (t - slice.T0), h - slice.H0, w - slice.W0);
            }
        }
    }

    private static void FillVidTablesV1(Tensor cosT, Tensor sinT, SeedVr2WindowSlice[] windows,
        int gridH, int gridW, int headDim)
    {
        Span<float> cos = cosT.AsSpan<float>();
        Span<float> sin = sinT.AsSpan<float>();
        InitIdentity(cos, sin, headDim);
        foreach (SeedVr2WindowSlice slice in windows)
        {
            int fT = slice.T1 - slice.T0, fH = slice.H1 - slice.H0, fW = slice.W1 - slice.W0;
            for (int t = slice.T0; t < slice.T1; t++)
            for (int h = slice.H0; h < slice.H1; h++)
            for (int w = slice.W0; w < slice.W1; w++)
            {
                int token = (t * gridH + h) * gridW + w;
                SeedVr2Rope.FillRowV1(cos, sin, token * headDim,
                    t - slice.T0, fT, h - slice.H0, fH, w - slice.W0, fW);
            }
        }
    }

    private static void FillTxtTables(Tensor cosT, Tensor sinT, int txtLen, int headDim)
    {
        Span<float> cos = cosT.AsSpan<float>();
        Span<float> sin = sinT.AsSpan<float>();
        InitIdentity(cos, sin, headDim);
        for (int p = 0; p < txtLen; p++)
            SeedVr2Rope.FillRow(cos, sin, p * headDim, p, p, p);
    }

    /// <summary>cos=1/sin=0 everywhere so the pass-through dims beyond RotDim rotate by zero.</summary>
    private static void InitIdentity(Span<float> cos, Span<float> sin, int headDim)
    {
        cos.Fill(1f);
        sin.Clear();
    }

    /// <summary>Frees the weight-cache entries and the host tensors.</summary>
    public void Release(IBackend backend)
    {
        if (_disposed)
            return;
        backend.FreeWeights(_owned);
        Dispose();
    }

    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;
        foreach (Tensor t in _owned)
            t.Dispose();
        _owned.Clear();
    }
}
