using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Sparse;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Core.Tests.Sparse;

public sealed unsafe class SparseOpsTests
{
    [Fact]
    public void Downsample_MeanIncludeSelf_DividesGroupSumByCountPlusOne()
    {
        CpuBackend backend = new();
        SparseTensor x = Make(new[] { 0, 0, 0, 0, 0, 1, 0, 0 }, resolution: 4, feats: new[] { 3f, 6f }, channels: 1);

        (SparseTensor withSelf, _) = SparseOps.Downsample(backend, x, 2, meanIncludeSelf: true);
        (SparseTensor withoutSelf, _) = SparseOps.Downsample(backend, x, 2, meanIncludeSelf: false);

        Assert.Equal(3f, Read(withSelf.Feats)[0], 5);   // (3 + 6) / (2 + 1)
        Assert.Equal(4.5f, Read(withoutSelf.Feats)[0], 5); // (3 + 6) / 2
    }

    [Fact]
    public void SubmanifoldConv_RulebookGemmMatchesDenseScatterConv()
    {
        CpuBackend backend = new();
        Random rng = new(7);
        int n = 12, res = 6, cin = 2, cout = 3;
        int[] coords = RandomCoords(rng, n, res);
        float[] feats = Enumerable.Range(0, n * cin).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray();
        SparseTensor x = Make(coords, res, feats, cin);
        // spconv layout [Cout, k, k, k, Cin]
        Tensor w = Filled(new TensorShape(new long[] { cout, 3, 3, 3, cin }), rng);

        SparseTensor sparse = SparseOps.SubmanifoldConv3dSparse(backend, x, SparseOps.ConvWeightSlices(w), null);
        SparseTensor dense = SparseOps.SubmanifoldConv3d(backend, x, SparseOps.PermuteConvWeight(w), null);

        float[] a = Read(sparse.Feats), b = Read(dense.Feats);
        Assert.Equal(a.Length, b.Length);
        for (int i = 0; i < a.Length; i++) Assert.Equal(b[i], a[i], 4);
    }

    [Fact]
    public void SubmanifoldConv_ChunkedRulebookIsBitIdenticalToUnchunked()
    {
        CpuBackend backend = new();
        Random rng = new(13);
        int n = 40, res = 6, cin = 3, cout = 5;
        SparseTensor x = Make(RandomCoords(rng, n, res), res, Enumerable.Range(0, n * cin).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray(), cin);
        Tensor[] slices = SparseOps.ConvWeightSlices(Filled(new TensorShape(new long[] { cout, 3, 3, 3, cin }), rng));
        Tensor bias = Filled(new TensorShape(cout), rng);

        // 140 bytes is 7 rows of max(cin, cout) = 5 floats, so most offsets split into several chunks with a short tail.
        float[] whole = Read(SparseOps.SubmanifoldConv3dSparse(backend, x, slices, bias, 1L << 30).Feats);
        float[] chunked = Read(SparseOps.SubmanifoldConv3dSparse(backend, x, slices, bias, 140).Feats);

        Assert.Equal(whole.Length, chunked.Length);
        for (int i = 0; i < whole.Length; i++)
            Assert.Equal(BitConverter.SingleToInt32Bits(whole[i]), BitConverter.SingleToInt32Bits(chunked[i]));
    }

    [Fact]
    public void SubmanifoldConv_NeighboursDoNotMatchAcrossBatchItems()
    {
        CpuBackend backend = new();
        SparseTensor x = Make(new[] { 0, 1, 1, 1, 1, 1, 1, 1 }, resolution: 4, feats: new[] { 1f, 10f }, channels: 1);
        Tensor w = new(new TensorShape(new long[] { 1, 3, 3, 3, 1 }), DType.F32);
        backend.Fill(w, 1f);

        SparseTensor y = SparseOps.SubmanifoldConv3dSparse(backend, x, SparseOps.ConvWeightSlices(w), null);

        Assert.Equal(new[] { 1f, 10f }, Read(y.Feats));
    }

    [Fact]
    public void Upsample_GathersEachVoxelFromItsDownsampledGroup()
    {
        CpuBackend backend = new();
        int[] coords = { 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0 };
        SparseTensor x = Make(coords, 4, new[] { 1f, 2f, 5f }, 1);

        (SparseTensor down, int[] idx) = SparseOps.Downsample(backend, x, 2, meanIncludeSelf: false);
        SparseTensor up = SparseOps.Upsample(backend, down, idx, coords, 4);

        float[] f = Read(up.Feats);
        Assert.Equal(new[] { 1.5f, 1.5f, 5f }, f.Take(3).ToArray());
    }

    private static SparseTensor Make(int[] coords, int resolution, float[] feats, int channels)
    {
        Tensor t = new(new TensorShape(coords.Length / 4, channels), DType.F32);
        float* p = (float*)t.DataPointer;
        for (int i = 0; i < feats.Length; i++) p[i] = feats[i];
        return new SparseTensor(t, coords, resolution);
    }

    private static Tensor Filled(TensorShape shape, Random rng)
    {
        Tensor t = new(shape, DType.F32);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < shape.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }

    private static float[] Read(Tensor t)
    {
        float* p = (float*)t.DataPointer;
        return Enumerable.Range(0, (int)t.ElementCount).Select(i => p[i]).ToArray();
    }

    private static int[] RandomCoords(Random rng, int n, int res)
    {
        HashSet<(int, int, int)> seen = new();
        List<int> coords = new();
        while (seen.Count < n)
        {
            (int, int, int) c = (rng.Next(res), rng.Next(res), rng.Next(res));
            if (!seen.Add(c)) continue;
            coords.Add(0); coords.Add(c.Item1); coords.Add(c.Item2); coords.Add(c.Item3);
        }
        return coords.ToArray();
    }
}
