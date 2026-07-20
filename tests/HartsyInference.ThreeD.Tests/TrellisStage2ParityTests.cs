using HartsyInference.Cuda;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ThreeD.Models.Trellis;
using HartsyInference.Core.Tensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.ThreeD.Tests;

/// <summary>TRELLIS stage-2 SLAT flow parity vs the real <c>SLatFlowModel</c> (dumped with the proven dense-conv +
/// dense-attention monkeypatch — see <c>/tmp/trellis_ref/dump_slat_flow.py</c>). Gated on the ckpt + reference dump.</summary>
[Trait("Category", "GpuIntegration")]
public sealed unsafe class TrellisStage2ParityTests
{
    private readonly ITestOutputHelper _out;
    public TrellisStage2ParityTests(ITestOutputHelper o) => _out = o;

    [Fact]
    public void SlatFlow_Velocity_MatchesReference()
    {
        string weights = "/tmp/TRELLIS-weights/ckpts/slat_flow_img_dit_L_64l8p2_fp16.safetensors";
        string refIo = "/tmp/trellis_ref/slat_flow_io.safetensors";
        string ptx = Path.Combine(Path.GetDirectoryName(typeof(TrellisStage2ParityTests).Assembly.Location)!, "Ptx");
        if (!File.Exists(weights) || !File.Exists(refIo) || !Directory.Exists(ptx)) { _out.WriteLine("SKIPPED."); return; }

        using CudaBackend backend = new(deviceOrdinal: 0, ptxDir: ptx);
        backend.CacheWeightCasts = false;
        using SafeTensorsLoader wl = new(); wl.Load(weights);
        using SafeTensorsLoader rl = new(); rl.Load(refIo);

        SlatFlowModel flow = new();
        flow.LoadWeights(wl.GetAllTensors());
        backend.PreloadWeights(flow.EnumerateWeights());

        Tensor noise = rl.GetTensor("noise");       // [N, 8]
        Tensor coordsT = rl.GetTensor("coords");     // [N, 4] int
        Tensor cond = rl.GetTensor("cond");           // [1, 1374, 1024]
        Tensor refV = rl.GetTensor("velocity");       // [N, 8]
        float tModel = ((float*)rl.GetTensor("t").DataPointer)[0];

        int nv = (int)(coordsT.Shape.ElementCount / 4);
        int[] coords = new int[nv * 4]; int* cp = (int*)coordsT.DataPointer;
        for (int i = 0; i < coords.Length; i++) coords[i] = cp[i];

        SparseTensor x = new(noise.Reshape(new TensorShape(1, nv, 8)), coords, 64);
        SparseTensor y = flow.Forward(backend, x, tModel, cond);

        long n = refV.Shape.ElementCount; float* a = (float*)y.Feats.DataPointer; float* b = (float*)refV.DataPointer;
        double mx = 0, dot = 0, na = 0, nb = 0;
        for (long i = 0; i < n; i++) { double p = a[i], q = b[i]; mx = Math.Max(mx, Math.Abs(p - q)); dot += p * q; na += p * p; nb += q * q; }
        double corr = dot / (Math.Sqrt(na * nb) + 1e-12);
        _out.WriteLine($"SLAT flow velocity [{nv},8] t={tModel}: maxAbs={mx:E3} corr={corr:F8}");
        Assert.True(corr > 0.999, $"SLAT flow CUDA≠torch corr={corr} maxAbs={mx}");
    }
}
