using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ThreeD.Models.Hunyuan3D;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.ThreeD.Tests;

/// <summary>Real-weight parity for the Hunyuan3D-2 shape DiT (Flux double/single-stream, no RoPE) vs the upstream
/// hy3dgen <c>Hunyuan3DDiT</c>. Gated on <c>HY3D_DIT_DIR</c> containing <c>dit_weights.safetensors</c> (the
/// <c>model.*</c> weights, F32) + <c>dit_ref_io.safetensors</c> (<c>latent</c> [1,3072,64], <c>cond</c> [1,1370,1536],
/// <c>timestep</c>, <c>velocity</c> [1,3072,64]), from <c>tests/python-reference/dump_hunyuan3d_dit.py</c>. GPU.</summary>
public sealed unsafe class Hunyuan3DDitParityTests
{
    private readonly ITestOutputHelper _out;
    public Hunyuan3DDitParityTests(ITestOutputHelper o) => _out = o;

    [Fact]
    public void Dit_Velocity_MatchesReference()
    {
        string? dir = Environment.GetEnvironmentVariable("HY3D_DIT_DIR");
        if (dir is null) { _out.WriteLine("SKIPPED: HY3D_DIT_DIR unset."); return; }
        string wPath = Path.Combine(dir, "dit_weights.safetensors"), rPath = Path.Combine(dir, "dit_ref_io.safetensors");
        if (!File.Exists(wPath) || !File.Exists(rPath)) { _out.WriteLine("SKIPPED: ref files missing."); return; }
        string ptx = Path.Combine(Path.GetDirectoryName(typeof(Hunyuan3DDitParityTests).Assembly.Location)!, "Ptx");
        if (!Directory.Exists(ptx)) { _out.WriteLine("SKIPPED: no PTX."); return; }

        using CudaBackend backend = new(deviceOrdinal: 0, ptxDir: ptx);
        backend.CacheWeightCasts = false;
        using SafeTensorsLoader wl = new(); wl.Load(wPath);
        using SafeTensorsLoader rl = new(); rl.Load(rPath);

        Hunyuan3DDit dit = new(Hunyuan3DConfig.Hunyuan3D2);
        dit.LoadWeights(wl.GetAllTensors());
        backend.PreloadWeights(dit.EnumerateWeights());

        Tensor latent = rl.GetTensor("latent");       // [1,3072,64]
        Tensor cond = rl.GetTensor("cond");           // [1,1370,1536]
        Tensor tsr = rl.GetTensor("timestep");        // [1]
        float t = ((float*)tsr.DataPointer)[0];

        Tensor vel = dit.Forward(backend, latent, cond, t);   // [1,3072,64]
        Tensor refVel = rl.GetTensor("velocity");

        Assert.Equal(refVel.ElementCount, vel.ElementCount);
        Tensor velF = vel.CastTo(DType.F32);
        (double maxAbs, double corr, double stdA, double stdB) = Cmp((float*)velF.DataPointer, (float*)refVel.DataPointer, refVel.ElementCount);
        _out.WriteLine($"DiT velocity: maxAbs={maxAbs:E3} corr={corr:F8} std(ours)={stdA:F4} std(ref)={stdB:F4}");
        velF.Dispose(); vel.Dispose();
        Assert.True(corr > 0.9999, $"corr {corr}");
        // maxAbs is TF32 tensor-core accumulation over 48 blocks × ~4400 tokens (corr ~1.0 is the real signal).
        Assert.True(maxAbs < 2e-1, $"maxAbs {maxAbs}");
    }

    private static (double, double, double, double) Cmp(float* a, float* b, long n)
    {
        double mx = 0, dot = 0, na = 0, nb = 0, sa = 0, sb = 0;
        for (long i = 0; i < n; i++)
        {
            double x = a[i], y = b[i];
            mx = Math.Max(mx, Math.Abs(x - y)); dot += x * y; na += x * x; nb += y * y; sa += x; sb += y;
        }
        double ma = sa / n, mb = sb / n;
        return (mx, dot / (Math.Sqrt(na * nb) + 1e-12), Math.Sqrt(na / n - ma * ma), Math.Sqrt(nb / n - mb * mb));
    }
}
