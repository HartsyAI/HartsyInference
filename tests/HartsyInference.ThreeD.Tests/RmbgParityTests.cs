using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.Vision.Rmbg;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.ThreeD.Tests;

/// <summary>Real-weight parity for BriaRMBG-1.4 (ISNet background removal) vs the upstream model. Gated on
/// <c>RMBG_DIR</c> containing <c>rmbg_weights.safetensors</c> + <c>rmbg_ref_io.safetensors</c> (<c>input</c>
/// [1,3,1024,1024] preprocessed + <c>mask</c> [1,1,1024,1024] = sigmoid(d1)) from
/// <c>tests/python-reference/rmbg_reference/dump_rmbg.py</c>. Runs on the GPU (deep conv net at 1024²).</summary>
public sealed unsafe class RmbgParityTests
{
    private readonly ITestOutputHelper _out;
    public RmbgParityTests(ITestOutputHelper o) => _out = o;

    [Fact]
    public void Rmbg_Mask_MatchesReference()
    {
        string? dir = Environment.GetEnvironmentVariable("RMBG_DIR");
        if (dir is null) { _out.WriteLine("SKIPPED: RMBG_DIR unset."); return; }
        string wPath = Path.Combine(dir, "rmbg_weights.safetensors"), rPath = Path.Combine(dir, "rmbg_ref_io.safetensors");
        if (!File.Exists(wPath) || !File.Exists(rPath)) { _out.WriteLine("SKIPPED: ref files missing."); return; }
        string ptx = Path.Combine(Path.GetDirectoryName(typeof(RmbgParityTests).Assembly.Location)!, "Ptx");
        if (!Directory.Exists(ptx)) { _out.WriteLine("SKIPPED: no PTX."); return; }

        using CudaBackend backend = new(deviceOrdinal: 0, ptxDir: ptx);
        using SafeTensorsLoader wl = new(); wl.Load(wPath);
        using SafeTensorsLoader rl = new(); rl.Load(rPath);

        BriaRmbg net = new();
        net.LoadWeights(wl.GetAllTensors());
        backend.PreloadWeights(net.EnumerateWeights());

        Tensor input = rl.GetTensor("input");     // [1,3,1024,1024]
        Tensor mask = net.Forward(backend, input);
        Tensor refMask = rl.GetTensor("mask");

        Assert.Equal(refMask.ElementCount, mask.ElementCount);
        Tensor maskF = mask.CastTo(DType.F32);
        (double maxAbs, double corr) = Cmp((float*)maskF.DataPointer, (float*)refMask.DataPointer, refMask.ElementCount);
        _out.WriteLine($"RMBG mask: maxAbs={maxAbs:E3} corr={corr:F8}");
        maskF.Dispose(); mask.Dispose();
        Assert.True(corr > 0.9999, $"corr {corr}");
        Assert.True(maxAbs < 5e-2, $"maxAbs {maxAbs}");
    }

    private static (double, double) Cmp(float* a, float* b, long n)
    {
        double mx = 0, dot = 0, na = 0, nb = 0;
        for (long i = 0; i < n; i++) { double x = a[i], y = b[i]; mx = Math.Max(mx, Math.Abs(x - y)); dot += x * y; na += x * x; nb += y * y; }
        return (mx, dot / (Math.Sqrt(na * nb) + 1e-12));
    }
}
