using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.Trellis;

/// <summary>Sparse-voxel ops for TRELLIS stage 2. The two coord-changing novelties are implemented without a sparse
/// library: <b>submanifold conv</b> = scatter features onto a dense grid → dense <see cref="IBackend.Conv3d"/> →
/// gather at the active voxels (inactive neighbours are zero, so a dense conv equals the submanifold conv exactly),
/// and <b>downsample</b> = average-pool over <c>coord/factor</c> groups. Correctness-first (host scatter/gather);
/// the grid is one-shot per conv. See <c>docs/Research/TRELLIS_ARCHITECTURE.md</c>.</summary>
public static unsafe class SparseOps
{
    /// <summary>Reorders an spconv conv weight <c>[Cout, kD, kH, kW, Cin]</c> → <c>[Cout, Cin, kD, kH, kW]</c>
    /// (the <see cref="IBackend.Conv3d"/> layout). One-time at load.</summary>
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

    /// <summary>Submanifold 3D conv (stride 1, k3, pad 1): output voxel set = input's. <paramref name="weightP"/> is
    /// the permuted <c>[Cout,Cin,kD,kH,kW]</c> weight (see <see cref="PermuteConvWeight"/>).</summary>
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

    /// <summary>Extracts the 27 <c>[Cout,Cin]</c> per-kernel-offset weight slices from an spconv conv weight
    /// <c>[Cout, kD, kH, kW, Cin]</c> — the GEMM weights for the sparse-conv rulebook. Offset index = (kd·3+kh)·3+kw.</summary>
    public static Tensor[] ConvWeightSlices(Tensor w)
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
                    slices[(kd * k + kh) * k + kw] = o;
                }
        return slices;
    }

    /// <summary>Submanifold 3D conv via the spconv rulebook (no dense grid): a coord hash finds active neighbours per
    /// kernel offset, then per offset gather → cuBLAS GEMM (<paramref name="wSlices"/>) → scatter-add. ~20-90× less
    /// compute than the dense-grid path at high channel counts (and no multi-GB grid). Output = input voxel set.</summary>
    public static SparseTensor SubmanifoldConv3dSparse(IBackend backend, SparseTensor x, Tensor[] wSlices, Tensor? bias)
    {
        int n = x.Count, cin = x.Channels, cout = (int)wSlices[0].Shape[0];
        Dictionary<long, int> hash = new(n);
        for (int i = 0; i < n; i++) hash[Pack(x.Coords[i * 4 + 1], x.Coords[i * 4 + 2], x.Coords[i * 4 + 3])] = i;

        Tensor outFeats = new(new TensorShape(1, n, cout), DType.F32); backend.Fill(outFeats, 0f);
        int[] inBuf = new int[n], outBuf = new int[n];
        for (int kd = 0; kd < 3; kd++)
            for (int kh = 0; kh < 3; kh++)
                for (int kw = 0; kw < 3; kw++)
                {
                    int dx = kd - 1, dy = kh - 1, dz = kw - 1, m = 0;
                    for (int o = 0; o < n; o++)
                    {
                        int nx = x.Coords[o * 4 + 1] + dx, ny = x.Coords[o * 4 + 2] + dy, nz = x.Coords[o * 4 + 3] + dz;
                        if (nx < 0 || ny < 0 || nz < 0) continue;
                        if (hash.TryGetValue(Pack(nx, ny, nz), out int ni)) { inBuf[m] = ni; outBuf[m] = o; m++; }
                    }
                    if (m == 0) continue;
                    Tensor inT = IntTensor(inBuf, m), outT = IntTensor(outBuf, m);
                    Tensor gathered = new(new TensorShape(1, m, cin), DType.F32); backend.RowGather(gathered, x.Feats, inT, m, cin);
                    Tensor gemm = new(new TensorShape(1, m, cout), DType.F32); backend.Linear(gemm, gathered, wSlices[(kd * 3 + kh) * 3 + kw], null);
                    backend.RowScatterAdd(outFeats, gemm, outT, m, cout);
                    gathered.Dispose(); gemm.Dispose(); inT.Dispose(); outT.Dispose();
                }
        if (bias is not null)
        {
            Tensor ones = new(new TensorShape(1, cout), DType.F32); backend.Fill(ones, 1f);
            Tensor biased = new(new TensorShape(1, n, cout), DType.F32); backend.AffineBroadcastLastDim(biased, outFeats, ones, bias);
            outFeats.Dispose(); ones.Dispose(); outFeats = biased;
        }
        return x.Replace(outFeats);
    }

    private static long Pack(int x, int y, int z) => ((long)x << 20) | ((long)y << 10) | (long)z;
    private static Tensor IntTensor(int[] src, int m)
    {
        Tensor t = new(new TensorShape(m), DType.I32); int* p = (int*)t.DataPointer;
        for (int i = 0; i < m; i++) p[i] = src[i];
        return t;
    }

    /// <summary>Average-pool downsample by <paramref name="factor"/> (coord/factor + dedup, mean over each group).
    /// Returns the downsampled tensor + the per-input-voxel group index (<c>idx[i]</c>) so a paired upsample can
    /// gather back (<c>up_feats[i] = down_feats[idx[i]]</c>, restoring the input coords).</summary>
    public static (SparseTensor down, int[] idx) Downsample(IBackend backend, SparseTensor x, int factor)
    {
        int n = x.Count, c = x.Channels;
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
        int m = map.Count;
        Tensor outFeats = new(new TensorShape(m, c), DType.F32);
        float* of = (float*)outFeats.DataPointer; new Span<float>(of, m * c).Clear();
        float* f = (float*)x.Feats.DataPointer;
        int[] counts = new int[m];
        for (int i = 0; i < n; i++)
        {
            int ni = idx[i]; counts[ni]++;
            for (int ch = 0; ch < c; ch++) of[(long)ni * c + ch] += f[(long)i * c + ch];
        }
        // TRELLIS uses torch.scatter_reduce(reduce='mean') with the DEFAULT include_self=True — the zero-initialized
        // "self" is included in the mean, so the divisor is count+1 (not count).
        for (int ni = 0; ni < m; ni++) { float inv = 1f / (counts[ni] + 1); for (int ch = 0; ch < c; ch++) of[(long)ni * c + ch] *= inv; }
        return (new SparseTensor(outFeats, newCoords.ToArray(), x.Resolution / factor), idx);
    }

    /// <summary>Nearest-neighbour upsample: restores the pre-downsample voxel set (<paramref name="upCoords"/> /
    /// <paramref name="upResolution"/>) by gathering each input voxel's downsampled-group feature via
    /// <paramref name="idx"/> (from the paired <see cref="Downsample"/>).</summary>
    public static SparseTensor Upsample(SparseTensor x, int[] idx, int[] upCoords, int upResolution)
    {
        int n = idx.Length, c = x.Channels;
        Tensor outFeats = new(new TensorShape(n, c), DType.F32);
        float* of = (float*)outFeats.DataPointer, f = (float*)x.Feats.DataPointer;
        for (int i = 0; i < n; i++)
            for (int ch = 0; ch < c; ch++) of[(long)i * c + ch] = f[(long)idx[i] * c + ch];
        return new SparseTensor(outFeats, upCoords, upResolution);
    }
}
