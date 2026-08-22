using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.Trellis;

/// <summary>Sparse-voxel ops for TRELLIS stage 2, implemented without a sparse library: submanifold conv via scatter→dense <see cref="IBackend.Conv3d"/>→gather, and downsample via average-pool over <c>coord/factor</c> groups.</summary>
public static unsafe class SparseOps
{
    /// <summary>Reorders an spconv conv weight <c>[Cout, kD, kH, kW, Cin]</c> → <c>[Cout, Cin, kD, kH, kW]</c> (the <see cref="IBackend.Conv3d"/> layout), one-time at load.</summary>
    public static Tensor PermuteConvWeight(Tensor w)
    {
        int cout = (int)w.Shape[0], k = (int)w.Shape[1], cin = (int)w.Shape[4];
        Tensor o = new(new TensorShape(new long[] { cout, cin, k, k, k }), DType.F32);
        Tensor wf = w.DType != DType.F32 ? w.CastTo(DType.F32) : w;
        float* s = (float*)wf.DataPointer, d = (float*)o.DataPointer;
        for (int co = 0; co < cout; co++)
            for (int kd = 0; kd < k; kd++)
                for (int kh = 0; kh < k; kh++)
                    for (int kw = 0; kw < k; kw++)
                        for (int ci = 0; ci < cin; ci++)
                            d[(((long)co * cin + ci) * k + kd) * k * k + kh * k + kw] =
                                s[((((long)co * k + kd) * k + kh) * k + kw) * cin + ci];
        return o;
    }

    /// <summary>Submanifold 3D conv (stride 1, k3, pad 1) whose output voxel set equals the input's; <paramref name="weightP"/> is the permuted weight from <see cref="PermuteConvWeight"/>.</summary>
    public static SparseTensor SubmanifoldConv3d(IBackend backend, SparseTensor x, Tensor weightP, Tensor? bias)
    {
        int r = x.Resolution, n = x.Count, cin = x.Channels, cout = (int)weightP.Shape[0];
        long r3 = (long)r * r * r;

        // coords → device I32 tensor (uploaded once for the scatter + gather).
        Tensor coordsT = new(new TensorShape(n, 4), DType.I32);
        int* ct = (int*)coordsT.DataPointer;
        fixed (int* cp = x.Coords) for (int i = 0; i < n * 4; i++) ct[i] = cp[i];

        // Grid built + consumed on-device (no host scatter loop, no multi-GB grid H2D).
        Tensor grid = new(new TensorShape(new long[] { 1, cin, r, r, r }), DType.F32);
        backend.Fill(grid, 0f);
        backend.SparseScatterToGrid(grid, x.Feats, coordsT, cin, r);
        Tensor outGrid = new(new TensorShape(new long[] { 1, cout, r, r, r }), DType.F32);
        backend.Conv3d(outGrid, grid, weightP, bias, 1, 1, 1, 1, 1, 1);
        Tensor outFeats = new(new TensorShape(n, cout), DType.F32);
        backend.SparseGatherFromGrid(outFeats, outGrid, coordsT, cout, r);
        grid.Dispose(); outGrid.Dispose(); coordsT.Dispose();
        return x.Replace(outFeats);
    }

    /// <summary>Extracts the 27 <c>[Cout,Cin]</c> per-kernel-offset weight slices from an spconv conv weight — the GEMM weights for the sparse-conv rulebook; <paramref name="f16"/> keeps them native F16 to halve slice VRAM.</summary>
    public static Tensor[] ConvWeightSlices(Tensor w, bool f16 = false)
    {
        int cout = (int)w.Shape[0], k = (int)w.Shape[1], cin = (int)w.Shape[4];
        Tensor wf = w.DType != DType.F32 ? w.CastTo(DType.F32) : w;
        float* s = (float*)wf.DataPointer;
        Tensor[] slices = new Tensor[k * k * k];
        for (int kd = 0; kd < k; kd++)
            for (int kh = 0; kh < k; kh++)
                for (int kw = 0; kw < k; kw++)
                {
                    Tensor o = new(new TensorShape(cout, cin), DType.F32);
                    float* d = (float*)o.DataPointer;
                    for (int co = 0; co < cout; co++)
                        for (int ci = 0; ci < cin; ci++)
                            d[(long)co * cin + ci] = s[((((long)co * k + kd) * k + kh) * k + kw) * cin + ci];
                    slices[(kd * k + kh) * k + kw] = f16 ? o.CastTo(DType.F16) : o;
                }
        GC.KeepAlive(wf);   // s points into wf's buffer; the per-slice allocs can GC-collect the otherwise-dead wf mid-loop
        return slices;
    }

    /// <summary>Submanifold 3D conv via the spconv rulebook (no dense grid): per kernel offset, gather → cuBLAS GEMM (<paramref name="wSlices"/>) → scatter-add, ~20-90× less compute than the dense-grid path at high channel counts.</summary>
    public static SparseTensor SubmanifoldConv3dSparse(IBackend backend, SparseTensor x, Tensor[] wSlices, Tensor? bias)
    {
        int n = x.Count, cin = x.Channels, cout = (int)wSlices[0].Shape[0];
        (int[] inIdx, int[] outIdx)[] rb = GetRulebook(x.Coords, n);   // spatial rulebook: cached per voxel set (see GetRulebook)

        Tensor outFeats = new(new TensorShape(1, n, cout), DType.F32); backend.Fill(outFeats, 0f);
        List<Tensor> temps = new();   // disposed after one Sync — freeing device buffers mid-async-kernel races (AV)
        for (int off = 0; off < 27; off++)
        {
            int m = rb[off].inIdx.Length;
            if (m == 0) continue;
            Tensor inT = IntTensor(rb[off].inIdx, m), outT = IntTensor(rb[off].outIdx, m);
            Tensor gathered = new(new TensorShape(1, m, cin), DType.F32); backend.RowGather(gathered, x.Feats, inT, m, cin);
            Tensor gemm = new(new TensorShape(1, m, cout), DType.F32); backend.Linear(gemm, gathered, wSlices[off], null);
            backend.RowScatterAdd(outFeats, gemm, outT, m, cout);
            temps.Add(inT); temps.Add(outT); temps.Add(gathered); temps.Add(gemm);
        }
        backend.Sync();
        foreach (Tensor t in temps) t.Dispose();
        if (bias is not null)
        {
            Tensor ones = new(new TensorShape(1, cout), DType.F32); backend.Fill(ones, 1f);
            Tensor biased = new(new TensorShape(1, n, cout), DType.F32); backend.AffineBroadcastLastDim(biased, outFeats, ones, bias);
            outFeats.Dispose(); ones.Dispose(); outFeats = biased;
        }
        return x.Replace(outFeats);
    }

    /// <summary>Presents sparse feats as a <c>[1,N,C]</c> rank-3 tensor without a cache-missing reshape when it is already that shape — a fresh <c>Reshape</c> would round-trip the whole feature buffer through host memory every forward.</summary>
    public static Tensor As3D(Tensor feats)
    {
        if (feats.Shape.Rank == 3 && feats.Shape[0] == 1) return feats;
        int c = (int)feats.Shape[feats.Shape.Rank - 1];
        return feats.Reshape(new TensorShape(1, (int)(feats.ElementCount / c), c));
    }

    private static long Pack(int x, int y, int z) => ((long)x << 20) | ((long)y << 10) | (long)z;
    private static Tensor IntTensor(int[] src, int m)
    {
        Tensor t = new(new TensorShape(m), DType.I32); int* p = (int*)t.DataPointer;
        for (int i = 0; i < m; i++) p[i] = src[i];
        return t;
    }

    // The spconv rulebook (per kernel offset: the (in,out) voxel-index pairs of active neighbours) is a pure function
    // of the voxel COORDS — identical across all 44 sampler steps (feats change, coords don't). Building it is a host
    // coord-hash + 27×N neighbour scan; caching it (keyed by the coords-array reference, which the sampler threads
    // stably and Downsample now returns stably) runs that scan ONCE per voxel set instead of per forward. HOST-only
    // (no device tensors cached) → no cross-backend device-pointer hazard; the cheap per-forward index upload stays.
    private static readonly System.Runtime.CompilerServices.ConditionalWeakTable<int[], (int[] inIdx, int[] outIdx)[]> _rulebookCache = new();

    private static (int[] inIdx, int[] outIdx)[] GetRulebook(int[] coords, int n)
    {
        if (_rulebookCache.TryGetValue(coords, out (int[] inIdx, int[] outIdx)[]? cached)) return cached;
        Dictionary<long, int> hash = new(n);
        for (int i = 0; i < n; i++) hash[Pack(coords[i * 4 + 1], coords[i * 4 + 2], coords[i * 4 + 3])] = i;
        (int[] inIdx, int[] outIdx)[] rb = new (int[], int[])[27];
        int[] inBuf = new int[n], outBuf = new int[n];
        for (int kd = 0; kd < 3; kd++)
            for (int kh = 0; kh < 3; kh++)
                for (int kw = 0; kw < 3; kw++)
                {
                    int dx = kd - 1, dy = kh - 1, dz = kw - 1, m = 0;
                    for (int o = 0; o < n; o++)
                    {
                        int nx = coords[o * 4 + 1] + dx, ny = coords[o * 4 + 2] + dy, nz = coords[o * 4 + 3] + dz;
                        if (nx < 0 || ny < 0 || nz < 0) continue;
                        if (hash.TryGetValue(Pack(nx, ny, nz), out int ni)) { inBuf[m] = ni; outBuf[m] = o; m++; }
                    }
                    rb[(kd * 3 + kh) * 3 + kw] = (inBuf[..m], outBuf[..m]);
                }
        _rulebookCache.Add(coords, rb);
        return rb;
    }

    /// <summary>Average-pool downsample by <paramref name="factor"/>, returning the downsampled tensor plus the per-input-voxel group index so a paired <see cref="Upsample"/> can gather back to the input coords.</summary>
    private sealed class DownGeom { public int[] Idx = null!, Coords = null!, Counts = null!; public int M; }
    private static readonly System.Runtime.CompilerServices.ConditionalWeakTable<int[], DownGeom> _downCache = new();

    public static (SparseTensor down, int[] idx) Downsample(IBackend backend, SparseTensor x, int factor)
    {
        int n = x.Count, c = x.Channels;
        // The downsample MAPPING (group index per voxel, deduped coords, group sizes) is geometry — constant across
        // sampler steps — so cache it keyed by the input coords reference and recompute only the feature averaging.
        // Returning the SAME cached coords array keeps the res-32 voxel set's identity stable so its rulebook caches too.
        if (!_downCache.TryGetValue(x.Coords, out DownGeom? g))
        {
            int[] idx = new int[n];
            Dictionary<(int, int, int, int), int> map = new(n);
            List<int> newCoords = new();
            fixed (int* cp = x.Coords)
            {
                for (int i = 0; i < n; i++)
                {
                    (int, int, int, int) key = (cp[i * 4], cp[i * 4 + 1] / factor, cp[i * 4 + 2] / factor, cp[i * 4 + 3] / factor);
                    if (!map.TryGetValue(key, out int ni))
                    {
                        ni = map.Count; map[key] = ni;
                        newCoords.Add(key.Item1); newCoords.Add(key.Item2); newCoords.Add(key.Item3); newCoords.Add(key.Item4);
                    }
                    idx[i] = ni;
                }
            }
            int mm = map.Count;
            int[] counts = new int[mm];
            for (int i = 0; i < n; i++) counts[idx[i]]++;
            g = new DownGeom { Idx = idx, Coords = newCoords.ToArray(), Counts = counts, M = mm };
            _downCache.Add(x.Coords, g);
        }
        int m = g.M;
        Tensor outFeats = new(new TensorShape(m, c), DType.F32);
        float* of = (float*)outFeats.DataPointer; new Span<float>(of, m * c).Clear();
        float* f = (float*)x.Feats.DataPointer;
        for (int i = 0; i < n; i++)
        {
            int ni = g.Idx[i];
            for (int ch = 0; ch < c; ch++) of[(long)ni * c + ch] += f[(long)i * c + ch];
        }
        // TRELLIS uses torch.scatter_reduce(reduce='mean') with the DEFAULT include_self=True — the zero-initialized
        // "self" is included in the mean, so the divisor is count+1 (not count).
        for (int ni = 0; ni < m; ni++) { float inv = 1f / (g.Counts[ni] + 1); for (int ch = 0; ch < c; ch++) of[(long)ni * c + ch] *= inv; }
        return (new SparseTensor(outFeats, g.Coords, x.Resolution / factor), g.Idx);
    }

    /// <summary>Nearest-neighbour upsample: restores the pre-downsample voxel set by gathering each input voxel's downsampled-group feature on-device via <paramref name="idx"/>, avoiding a full feature-buffer D2H/H2D round trip.</summary>
    public static SparseTensor Upsample(IBackend backend, SparseTensor x, int[] idx, int[] upCoords, int upResolution)
    {
        int n = idx.Length, c = x.Channels;
        Tensor idxT = IntTensor(idx, n);
        Tensor outFeats = new(new TensorShape(1, n, c), DType.F32);
        backend.RowGather(outFeats, As3D(x.Feats), idxT, n, c);
        idxT.Dispose();
        return new SparseTensor(outFeats, upCoords, upResolution);
    }
}
