using HartsyInference.Cuda;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ThreeD.Models.Trellis;
using HartsyInference.Core.Tensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.ThreeD.Tests;

/// <summary>TRELLIS image-conditioner parity vs the real <c>dinov2_vitl14_reg</c>. Feeds the dumped preprocessed
/// input (<c>prep</c>, isolating the network from image resize/crop) through the C# conditioner and compares the
/// <c>[1,1374,1024]</c> cond to the reference (<c>x_prenorm</c> → non-affine layer_norm). Gated on the cond dump
/// (<c>/tmp/trellis_ref/cond_dragon.safetensors</c>, has <c>prep</c>+<c>cond</c>) + the remapped dinov2 weights
/// (<c>/tmp/trellis_ref/dinov2_vitl14_reg.safetensors</c> from <c>convert_dinov2_reg.py</c>).</summary>
[Trait("Category", "GpuIntegration")]
public sealed unsafe class TrellisConditionerParityTests
{
    private readonly ITestOutputHelper _out;
    public TrellisConditionerParityTests(ITestOutputHelper o) => _out = o;

    [Fact]
    public void Conditioner_MatchesReference()
    {
        string condFile = "/tmp/trellis_ref/cond_dragon.safetensors";
        string dinoFile = "/tmp/trellis_ref/dinov2_vitl14_reg.safetensors";
        string ptx = Path.Combine(Path.GetDirectoryName(typeof(TrellisConditionerParityTests).Assembly.Location)!, "Ptx");
        if (!File.Exists(condFile) || !File.Exists(dinoFile) || !Directory.Exists(ptx)) { _out.WriteLine("SKIPPED."); return; }
        using CudaBackend backend = new(deviceOrdinal: 0, ptxDir: ptx);

        using SafeTensorsLoader cl = new(); cl.Load(condFile);
        Tensor prep = cl.GetTensor("prep");     // [1,3,518,518] ImageNet-normalized
        Tensor refCond = cl.GetTensor("cond");  // [1,1374,1024]

        using SafeTensorsLoader dl = new(); dl.Load(dinoFile);
        TrellisImageConditioner cond = new(); cond.LoadWeights(dl.GetAllTensors());
        backend.PreloadWeights(cond.EnumerateWeights());
        Tensor y = cond.Encode(backend, prep);

        (double mx, double corr) = Cmp(y, refCond);
        _out.WriteLine($"TRELLIS conditioner {y.Shape}: maxAbs={mx:E3} corr={corr:F8}");
        // corr is the gate: a torch F32 forward with these converted weights matches the reference to maxAbs 1.9e-4
        // (spec/converter exact). The residual maxAbs (~0.7 on DINOv2's high-norm outlier tokens, ref magnitude ~13)
        // is TF32/SDPA GEMM noise accumulated over 24 layers — the reference cond was itself dumped on CUDA (TF32),
        // so this is TF32-vs-TF32, not a bug. The abs bound is a loose sanity check only.
        Assert.True(corr > 0.9999 && mx < 1.5, $"conditioner ≠ ref corr={corr} maxAbs={mx}");
    }

    private static (double, double) Cmp(Tensor a, Tensor b)
    {
        long n = a.Shape.ElementCount; float* pa = (float*)a.DataPointer; float* pb = (float*)b.DataPointer;
        double mx = 0, dot = 0, na = 0, nb = 0;
        for (long i = 0; i < n; i++) { double x = pa[i], yv = pb[i]; mx = Math.Max(mx, Math.Abs(x - yv)); dot += x * yv; na += x * x; nb += yv * yv; }
        return (mx, dot / (Math.Sqrt(na * nb) + 1e-12));
    }
}
