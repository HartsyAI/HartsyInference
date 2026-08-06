using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ThreeD.Models.Hunyuan3D;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.ThreeD.Tests;

/// <summary>Real-weight parity for the Hunyuan3D-2 ShapeVAE decoder (post_kl + transformer resblocks + geo_decoder
/// cross-attn) vs the upstream hy3dgen decode path. Gated on <c>HY3D_VAE_DIR</c> containing
/// <c>vae_weights.safetensors</c> + <c>vae_ref_io.safetensors</c> (<c>latent</c> [1,3072,64], <c>queries</c>
/// [1,N,3], <c>occupancy</c> [1,N,1]) from <c>tests/python-reference/dump_hunyuan3d_vae.py</c>. GPU.</summary>
public sealed unsafe class Hunyuan3DVaeParityTests
{
    private readonly ITestOutputHelper _out;
    public Hunyuan3DVaeParityTests(ITestOutputHelper o) => _out = o;

    [Fact]
    public void Vae_Occupancy_MatchesReference()
    {
        string? dir = Environment.GetEnvironmentVariable("HY3D_VAE_DIR");
        if (dir is null) { _out.WriteLine("SKIPPED: HY3D_VAE_DIR unset."); return; }
        string wPath = Path.Combine(dir, "vae_weights.safetensors"), rPath = Path.Combine(dir, "vae_ref_io.safetensors");
        if (!File.Exists(wPath) || !File.Exists(rPath)) { _out.WriteLine("SKIPPED: ref files missing."); return; }
        string ptx = Path.Combine(Path.GetDirectoryName(typeof(Hunyuan3DVaeParityTests).Assembly.Location)!, "Ptx");
        if (!Directory.Exists(ptx)) { _out.WriteLine("SKIPPED: no PTX."); return; }

        using CudaBackend backend = new(deviceOrdinal: 0, ptxDir: ptx);
        using SafeTensorsLoader wl = new(); wl.Load(wPath);
        using SafeTensorsLoader rl = new(); rl.Load(rPath);

        Hunyuan3DShapeVae vae = new(Hunyuan3DConfig.Hunyuan3D2);
        vae.LoadWeights(wl.GetAllTensors());
        backend.PreloadWeights(vae.EnumerateWeights());

        Tensor latent = rl.GetTensor("latent");        // [1,3072,64]
        Tensor queries = rl.GetTensor("queries");      // [1,N,3]
        Tensor refOcc = rl.GetTensor("occupancy");     // [1,N,1]
        int n = (int)queries.Shape[1];

        Tensor processed = vae.ProcessLatents(backend, latent);
        (Tensor kNorm, Tensor v) kv = vae.PrepareKv(backend, processed);
        Tensor occ = vae.DecodePoints(backend, kv, queries, n);   // queries [1,N,3] = flat [N,3] for FourierEmbed
        processed.Dispose(); kv.kNorm.Dispose(); kv.v.Dispose();

        Tensor occF = occ.DType == DType.F32 ? occ : occ.CastTo(DType.F32);
        (double maxAbs, double corr) = Cmp((float*)occF.DataPointer, (float*)refOcc.DataPointer, n);
        _out.WriteLine($"VAE occupancy: maxAbs={maxAbs:E3} corr={corr:F8}");
        if (!ReferenceEquals(occF, occ)) occF.Dispose();
        occ.Dispose();
        Assert.True(corr > 0.9999, $"corr {corr}");
        // maxAbs is TF32 tensor-core accumulation over 16 resblocks + geo cross-attn (corr ~1.0 is the real signal).
        Assert.True(maxAbs < 2e-1, $"maxAbs {maxAbs}");
    }

    private static (double, double) Cmp(float* a, float* b, long n)
    {
        double mx = 0, dot = 0, na = 0, nb = 0;
        for (long i = 0; i < n; i++) { double x = a[i], y = b[i]; mx = Math.Max(mx, Math.Abs(x - y)); dot += x * y; na += x * x; nb += y * y; }
        return (mx, dot / (Math.Sqrt(na * nb) + 1e-12));
    }
}
