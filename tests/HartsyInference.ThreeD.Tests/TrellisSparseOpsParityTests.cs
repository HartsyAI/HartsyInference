using HartsyInference.Cuda;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.ThreeD.Models.Trellis;
using HartsyInference.Core.Tensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.ThreeD.Tests;

/// <summary>TRELLIS sparse-op parity vs a first-principles numpy reference (submanifold conv = dense-conv-on-scatter,
/// downsample = avg-pool — both provably equal to the real spconv/spatial ops, so no spconv install needed). Gated
/// on the reference dump at <c>/tmp/trellis_ref/sparse_ops_io.safetensors</c>.</summary>
[Trait("Category", "GpuIntegration")]
public sealed unsafe class TrellisSparseOpsParityTests
{
    private readonly ITestOutputHelper _out;
    public TrellisSparseOpsParityTests(ITestOutputHelper o) => _out = o;

    [Fact]
    public void SubmanifoldConv3d_MatchesReference()
    {
        string refIo = "/tmp/trellis_ref/sparse_ops_io.safetensors";
        string ptx = Path.Combine(Path.GetDirectoryName(typeof(TrellisSparseOpsParityTests).Assembly.Location)!, "Ptx");
        if (!File.Exists(refIo) || !Directory.Exists(ptx)) { _out.WriteLine("SKIPPED."); return; }
        using CudaBackend backend = new(deviceOrdinal: 0, ptxDir: ptx);
        using SafeTensorsLoader rl = new(); rl.Load(refIo);

        Tensor feats = rl.GetTensor("feats");     // [N, Cin]
        int[] coords = ReadCoords(rl.GetTensor("coords"));
        Tensor weight = rl.GetTensor("weight");   // [Cout, K,K,K, Cin]
        Tensor bias = rl.GetTensor("bias");
        Tensor refConv = rl.GetTensor("conv_out");   // [N, Cout]

        SparseTensor x = new(feats, coords, 16);
        Tensor wP = SparseOps.PermuteConvWeight(weight);
        SparseTensor y = SparseOps.SubmanifoldConv3d(backend, x, wP, bias);

        (double mx, double corr) = Cmp(y.Feats, refConv);
        _out.WriteLine($"Submanifold conv [{x.Count},{x.Channels}]→[{y.Count},{y.Channels}]: maxAbs={mx:E3} corr={corr:F8}");
        Assert.True(corr > 0.99999 && mx < 1e-3, $"Submanifold conv ≠ ref corr={corr} maxAbs={mx}");
    }

    [Fact]
    public void SubmanifoldConv3dSparse_MatchesReference()
    {
        string refIo = "/tmp/trellis_ref/sparse_ops_io.safetensors";
        string ptx = Path.Combine(Path.GetDirectoryName(typeof(TrellisSparseOpsParityTests).Assembly.Location)!, "Ptx");
        if (!File.Exists(refIo) || !Directory.Exists(ptx)) { _out.WriteLine("SKIPPED."); return; }
        using CudaBackend backend = new(deviceOrdinal: 0, ptxDir: ptx);
        using SafeTensorsLoader rl = new(); rl.Load(refIo);
        Tensor feats = rl.GetTensor("feats"); int[] coords = ReadCoords(rl.GetTensor("coords"));
        Tensor weight = rl.GetTensor("weight"); Tensor bias = rl.GetTensor("bias"); Tensor refConv = rl.GetTensor("conv_out");
        SparseTensor x = new(feats, coords, 16);
        Tensor[] slices = SparseOps.ConvWeightSlices(weight);
        SparseTensor y = SparseOps.SubmanifoldConv3dSparse(backend, x, slices, bias);
        (double mx, double corr) = Cmp(y.Feats, refConv);
        _out.WriteLine($"Submanifold conv (rulebook) [{x.Count},{x.Channels}]→[{y.Count},{y.Channels}]: maxAbs={mx:E3} corr={corr:F8}");
        Assert.True(corr > 0.99999 && mx < 5e-2, $"rulebook conv ≠ ref corr={corr} maxAbs={mx}");
    }

    [Fact]
    public void Downsample_MatchesReference()
    {
        string refIo = "/tmp/trellis_ref/sparse_ops_io.safetensors";
        string ptx = Path.Combine(Path.GetDirectoryName(typeof(TrellisSparseOpsParityTests).Assembly.Location)!, "Ptx");
        if (!File.Exists(refIo) || !Directory.Exists(ptx)) { _out.WriteLine("SKIPPED."); return; }
        using CudaBackend backend = new(deviceOrdinal: 0, ptxDir: ptx);
        using SafeTensorsLoader rl = new(); rl.Load(refIo);

        Tensor feats = rl.GetTensor("feats");
        int[] coords = ReadCoords(rl.GetTensor("coords"));
        int[] refCoords = ReadCoords(rl.GetTensor("down_coords"));
        Tensor refFeats = rl.GetTensor("down_feats");   // [M, C]
        int c = (int)refFeats.Shape[1];

        SparseTensor x = new(feats, coords, 16);
        (SparseTensor down, int[] _) = SparseOps.Downsample(backend, x, 2);

        // Coord-keyed compare (dedup order differs between numpy-sort and C#-first-seen).
        Dictionary<(int, int, int, int), int> refMap = new();
        for (int i = 0; i < refCoords.Length / 4; i++) refMap[(refCoords[i * 4], refCoords[i * 4 + 1], refCoords[i * 4 + 2], refCoords[i * 4 + 3])] = i;
        float* df = (float*)down.Feats.DataPointer; float* rf = (float*)refFeats.DataPointer;
        double mx = 0; int matched = 0;
        for (int i = 0; i < down.Count; i++)
        {
            (int, int, int, int) key = (down.Coords[i * 4], down.Coords[i * 4 + 1], down.Coords[i * 4 + 2], down.Coords[i * 4 + 3]);
            Assert.True(refMap.TryGetValue(key, out int ri), $"C# down voxel {key} not in reference");
            matched++;
            for (int ch = 0; ch < c; ch++) mx = Math.Max(mx, Math.Abs(df[(long)i * c + ch] - rf[(long)ri * c + ch]));
        }
        _out.WriteLine($"Downsample [{x.Count}]→[{down.Count}] (ref {refCoords.Length / 4}): matched={matched} maxAbs={mx:E3}");
        Assert.True(down.Count == refCoords.Length / 4 && mx < 1e-5, $"Downsample ≠ ref count={down.Count} maxAbs={mx}");
    }

    private static int[] ReadCoords(Tensor t)
    {
        int n = (int)t.Shape.ElementCount;
        int[] o = new int[n]; int* p = (int*)t.DataPointer;
        for (int i = 0; i < n; i++) o[i] = p[i];
        return o;
    }

    private static (double, double) Cmp(Tensor a, Tensor b)
    {
        long n = a.Shape.ElementCount; float* pa = (float*)a.DataPointer; float* pb = (float*)b.DataPointer;
        double mx = 0, dot = 0, na = 0, nb = 0;
        for (long i = 0; i < n; i++) { double x = pa[i], y = pb[i]; mx = Math.Max(mx, Math.Abs(x - y)); dot += x * y; na += x * x; nb += y * y; }
        return (mx, dot / (Math.Sqrt(na * nb) + 1e-12));
    }
}
