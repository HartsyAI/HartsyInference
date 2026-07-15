using HartsyInference.Cuda;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.ThreeD.Models.Trellis;
using HartsyInference.Core.Tensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.ThreeD.Tests;

/// <summary>TRELLIS stage-1 (dense) parity vs a standalone torch reference. Gated on the ckpt + reference-dump
/// files (env <c>TRELLIS_SS_DEC</c> / <c>TRELLIS_SS_DEC_REF</c>, or the default <c>/tmp</c> paths from the build).
/// The reference feeds a fixed-seed random latent (so occupancy is all-negative — parity compares the logit
/// <b>values</b>, not the active-voxel set; the active-set gate is the full-pipeline e2e test).</summary>
[Trait("Category", "GpuIntegration")]
public sealed unsafe class TrellisStage1ParityTests
{
    private readonly ITestOutputHelper _out;
    public TrellisStage1ParityTests(ITestOutputHelper o) => _out = o;

    [Fact]
    public void SsDecoder_Occupancy_MatchesReference()
    {
        string weights = Environment.GetEnvironmentVariable("TRELLIS_SS_DEC") ?? "/tmp/TRELLIS-weights/ckpts/ss_dec_conv3d_16l8_fp16.safetensors";
        string refIo = Environment.GetEnvironmentVariable("TRELLIS_SS_DEC_REF") ?? "/tmp/trellis_ref/ss_dec_io.safetensors";
        string ptx = Path.Combine(Path.GetDirectoryName(typeof(TrellisStage1ParityTests).Assembly.Location)!, "Ptx");
        if (!File.Exists(weights) || !File.Exists(refIo) || !Directory.Exists(ptx)) { _out.WriteLine("SKIPPED: weights/ref/PTX missing."); return; }

        using CudaBackend backend = new(deviceOrdinal: 0, ptxDir: ptx);
        backend.CacheWeightCasts = false;
        using SafeTensorsLoader wl = new(); wl.Load(weights);
        using SafeTensorsLoader rl = new(); rl.Load(refIo);

        SparseStructureDecoder dec = new();
        dec.LoadWeights(wl.GetAllTensors());
        backend.PreloadWeights(dec.EnumerateWeights());

        Tensor latent = rl.GetTensor("latent");   // [1,8,16,16,16]
        Tensor refOcc = rl.GetTensor("occ");       // [1,1,64,64,64]
        Tensor occ = dec.Decode(backend, latent);

        long n = occ.Shape.ElementCount;
        float* a = (float*)occ.DataPointer; float* b = (float*)refOcc.DataPointer;
        double mx = 0, dot = 0, na = 0, nb = 0;
        for (long i = 0; i < n; i++) { double x = a[i], y = b[i]; mx = Math.Max(mx, Math.Abs(x - y)); dot += x * y; na += x * x; nb += y * y; }
        double corr = dot / (Math.Sqrt(na * nb) + 1e-12);
        _out.WriteLine($"SS decoder occupancy [1,1,64³]: maxAbs={mx:E3} corr={corr:F8}");
        occ.Dispose();
        Assert.True(corr > 0.9999, $"SS decoder CUDA≠torch corr={corr} maxAbs={mx}");
    }

    [Fact]
    public void SsFlow_Velocity_MatchesReference()
    {
        string weights = Environment.GetEnvironmentVariable("TRELLIS_SS_FLOW") ?? "/tmp/TRELLIS-weights/ckpts/ss_flow_img_dit_L_16l8_fp16.safetensors";
        string refIo = Environment.GetEnvironmentVariable("TRELLIS_SS_FLOW_REF") ?? "/tmp/trellis_ref/ss_flow_io.safetensors";
        string ptx = Path.Combine(Path.GetDirectoryName(typeof(TrellisStage1ParityTests).Assembly.Location)!, "Ptx");
        if (!File.Exists(weights) || !File.Exists(refIo) || !Directory.Exists(ptx)) { _out.WriteLine("SKIPPED: weights/ref/PTX missing."); return; }

        using CudaBackend backend = new(deviceOrdinal: 0, ptxDir: ptx);
        backend.CacheWeightCasts = false;
        using SafeTensorsLoader wl = new(); wl.Load(weights);
        using SafeTensorsLoader rl = new(); rl.Load(refIo);

        SparseStructureFlow flow = new();
        flow.LoadWeights(wl.GetAllTensors());
        backend.PreloadWeights(flow.EnumerateWeights());

        Tensor latent = rl.GetTensor("latent");   // [1,8,16,16,16]
        Tensor cond = rl.GetTensor("cond");         // [1,1374,1024]
        Tensor refV = rl.GetTensor("velocity");     // [1,8,16,16,16]
        float tModel = ((float*)rl.GetTensor("t").DataPointer)[0];   // 500.0
        Tensor v = flow.Forward(backend, latent, tModel, cond);

        long n = v.Shape.ElementCount;
        float* a = (float*)v.DataPointer; float* b = (float*)refV.DataPointer;
        double mx = 0, dot = 0, na = 0, nb = 0;
        for (long i = 0; i < n; i++) { double x = a[i], y = b[i]; mx = Math.Max(mx, Math.Abs(x - y)); dot += x * y; na += x * x; nb += y * y; }
        double corr = dot / (Math.Sqrt(na * nb) + 1e-12);
        _out.WriteLine($"SS flow velocity [1,8,16³] t={tModel}: maxAbs={mx:E3} corr={corr:F8}");
        v.Dispose();
        Assert.True(corr > 0.9999, $"SS flow CUDA≠torch corr={corr} maxAbs={mx}");
    }

    [Fact]
    public void SsPipeline_Occupancy_MatchesReference()   // full stage 1: noise → sampler → decoder → occupancy
    {
        string fw = "/tmp/TRELLIS-weights/ckpts/ss_flow_img_dit_L_16l8_fp16.safetensors";
        string dw = "/tmp/TRELLIS-weights/ckpts/ss_dec_conv3d_16l8_fp16.safetensors";
        string refIo = "/tmp/trellis_ref/ss_pipeline_io.safetensors";
        string ptx = Path.Combine(Path.GetDirectoryName(typeof(TrellisStage1ParityTests).Assembly.Location)!, "Ptx");
        if (!File.Exists(fw) || !File.Exists(dw) || !File.Exists(refIo) || !Directory.Exists(ptx)) { _out.WriteLine("SKIPPED."); return; }

        using CudaBackend backend = new(deviceOrdinal: 0, ptxDir: ptx);
        backend.CacheWeightCasts = false;
        using SafeTensorsLoader fl = new(); fl.Load(fw);
        using SafeTensorsLoader dl = new(); dl.Load(dw);
        using SafeTensorsLoader rl = new(); rl.Load(refIo);

        SparseStructureFlow flow = new(); flow.LoadWeights(fl.GetAllTensors());
        SparseStructureDecoder dec = new(); dec.LoadWeights(dl.GetAllTensors());
        backend.PreloadWeights(flow.EnumerateWeights());
        backend.PreloadWeights(dec.EnumerateWeights());

        Tensor noise = rl.GetTensor("noise"); Tensor cond = rl.GetTensor("cond"); Tensor refOcc = rl.GetTensor("occ");
        Tensor neg = new(cond.Shape, DType.F32); backend.Fill(neg, 0f);
        TrellisSparseStructureSampler sampler = new();
        Tensor zS = sampler.Sample(backend, flow, noise, cond, neg);
        Tensor occ = dec.Decode(backend, zS); zS.Dispose();

        long n = occ.Shape.ElementCount; float* a = (float*)occ.DataPointer; float* b = (float*)refOcc.DataPointer;
        double mx = 0, dot = 0, na = 0, nb = 0; int activeC = 0, activeR = 0, agree = 0;
        for (long i = 0; i < n; i++)
        {
            double x = a[i], y = b[i]; mx = Math.Max(mx, Math.Abs(x - y)); dot += x * y; na += x * x; nb += y * y;
            bool ac = a[i] > 0, ar = b[i] > 0; if (ac) activeC++; if (ar) activeR++; if (ac == ar) agree++;
        }
        double corr = dot / (Math.Sqrt(na * nb) + 1e-12);
        _out.WriteLine($"SS pipeline occ [1,1,64³]: corr={corr:F8} maxAbs={mx:E3} activeCUDA={activeC} activeRef={activeR} voxelAgree={100.0 * agree / n:F3}%");
        occ.Dispose();
        Assert.True(corr > 0.999, $"SS pipeline CUDA≠torch corr={corr}");
        Assert.True(agree > 0.999 * n, $"active-voxel agreement {100.0 * agree / n:F2}% too low");
    }
}
