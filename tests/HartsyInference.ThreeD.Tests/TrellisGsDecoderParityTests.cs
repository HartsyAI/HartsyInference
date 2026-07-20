using HartsyInference.Cuda;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ThreeD.Models.Trellis;
using HartsyInference.Core.Tensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.ThreeD.Tests;

/// <summary>TRELLIS SLAT→Gaussian decoder network parity vs the real <c>SLatGaussianDecoder</c> (dumped with the
/// windowed-attention monkeypatch — <c>/tmp/trellis_ref/dump_gs_dec.py</c>). Compares the 448-dim per-voxel gaussian
/// params before <c>to_representation</c>.</summary>
[Trait("Category", "GpuIntegration")]
public sealed unsafe class TrellisGsDecoderParityTests
{
    private readonly ITestOutputHelper _out;
    public TrellisGsDecoderParityTests(ITestOutputHelper o) => _out = o;

    [Fact]
    public void GsDecoder_NetOutput_MatchesReference()
    {
        string weights = "/tmp/TRELLIS-weights/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16.safetensors";
        string refIo = "/tmp/trellis_ref/gs_dec_io.safetensors";
        string ptx = Path.Combine(Path.GetDirectoryName(typeof(TrellisGsDecoderParityTests).Assembly.Location)!, "Ptx");
        if (!File.Exists(weights) || !File.Exists(refIo) || !Directory.Exists(ptx)) { _out.WriteLine("SKIPPED."); return; }

        using CudaBackend backend = new(deviceOrdinal: 0, ptxDir: ptx);
        backend.CacheWeightCasts = false;
        using SafeTensorsLoader wl = new(); wl.Load(weights);
        using SafeTensorsLoader rl = new(); rl.Load(refIo);

        SlatGaussianDecoder dec = new();
        dec.LoadWeights(wl.GetAllTensors());
        backend.PreloadWeights(dec.EnumerateWeights());

        Tensor slat = rl.GetTensor("slat");           // [N, 8]
        Tensor refOut = rl.GetTensor("netout");       // [N, 448]
        int nv = (int)(rl.GetTensor("coords").Shape.ElementCount / 4);
        int[] coords = new int[nv * 4]; int* cp = (int*)rl.GetTensor("coords").DataPointer; for (int i = 0; i < coords.Length; i++) coords[i] = cp[i];

        SparseTensor x = new(slat.Reshape(new TensorShape(1, nv, 8)), coords, 64);
        SparseTensor y = dec.Forward(backend, x);

        long n = refOut.Shape.ElementCount; float* a = (float*)y.Feats.DataPointer; float* b = (float*)refOut.DataPointer;
        double mx = 0, dot = 0, na = 0, nb = 0;
        for (long i = 0; i < n; i++) { double p = a[i], q = b[i]; mx = Math.Max(mx, Math.Abs(p - q)); dot += p * q; na += p * p; nb += q * q; }
        double corr = dot / (Math.Sqrt(na * nb) + 1e-12);
        _out.WriteLine($"GS decoder netout [{nv},448]: maxAbs={mx:E3} corr={corr:F8}");
        Assert.True(corr > 0.999, $"GS decoder CUDA≠torch corr={corr} maxAbs={mx}");
    }
}
