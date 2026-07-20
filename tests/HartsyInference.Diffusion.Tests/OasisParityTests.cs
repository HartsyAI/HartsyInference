using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Real-weight numeric parity for Oasis-500m vs the upstream <c>etched-ai/open-oasis</c> reference.
/// Gated on <c>OASIS_DIT</c> (oasis500m.safetensors), <c>OASIS_VAE</c> (vit-l-20.safetensors), <c>OASIS_REF</c>
/// (oasis_ref_io.safetensors from <c>dump_oasis_reference.py</c>). <c>PARITY_BACKEND=cuda</c> runs on the GPU.
/// Skips cleanly when unset.</summary>
public sealed unsafe class OasisParityTests
{
    private readonly ITestOutputHelper _out;
    public OasisParityTests(ITestOutputHelper o) => _out = o;

    private static bool IsCuda => string.Equals(Environment.GetEnvironmentVariable("PARITY_BACKEND"), "cuda", StringComparison.OrdinalIgnoreCase);
    private static double AbsTol => IsCuda ? 0.2 : 5e-3;

    private static IBackend MakeBackend()
    {
        if (!IsCuda) return new CpuBackend();
        return new HartsyInference.Cuda.CudaBackend(0, PtxDir());
    }

    private static string PtxDir()
    {
        string? d = AppContext.BaseDirectory;
        for (int i = 0; i < 8 && d is not null; i++, d = Path.GetDirectoryName(d))
        {
            string cand = Path.Combine(d, "src", "HartsyInference.Cuda", "Ptx");
            if (Directory.Exists(cand)) return cand;
        }
        string local = Path.Combine(AppContext.BaseDirectory, "Ptx");
        return Directory.Exists(local) ? local : throw new DirectoryNotFoundException("PTX dir not found");
    }

    private static (SafeTensorsLoader dit, SafeTensorsLoader vae, SafeTensorsLoader refIo)? Load()
    {
        string? d = Environment.GetEnvironmentVariable("OASIS_DIT");
        string? v = Environment.GetEnvironmentVariable("OASIS_VAE");
        string? r = Environment.GetEnvironmentVariable("OASIS_REF");
        if (d is null || v is null || r is null || !File.Exists(d) || !File.Exists(v) || !File.Exists(r)) return null;
        SafeTensorsLoader dl = new(); dl.Load(d);
        SafeTensorsLoader vl = new(); vl.Load(v);
        SafeTensorsLoader rl = new(); rl.Load(r);
        return (dl, vl, rl);
    }

    [Fact]
    public void VaeEncode_MatchReference()
    {
        (SafeTensorsLoader dit, SafeTensorsLoader vae, SafeTensorsLoader refIo)? loaded = Load();
        if (loaded is null) return; // gated
        (SafeTensorsLoader dit, SafeTensorsLoader vae, SafeTensorsLoader refIo) = loaded.Value;
        using (dit) using (vae) using (refIo)
        {
            using IBackend backend = MakeBackend();
            OasisVitVae model = new(); model.LoadWeights(vae.GetAllTensors());

            Tensor rgb = refIo.GetTensor("vae_rgb_in");           // [1,3,360,640]
            Tensor latent = model.Encode(backend, rgb);           // [1,16,18,32]
            Tensor refMean = refIo.GetTensor("vae_encode_mean");  // [1,576,16]

            // C# latent[c*576 + i]  vs  ref[i*16 + c]
            float* cs = (float*)latent.DataPointer; float* rf = (float*)refMean.DataPointer;
            int s = 18 * 32, lat = 16;
            (double maxAbs, double corr) = CompareIndexed(cs, rf, s, lat, transpose: true);
            _out.WriteLine($"VAE encode mean: maxAbs={maxAbs:E3} corr={corr:F8}");
            latent.Dispose();
            Assert.True(maxAbs < AbsTol, $"maxAbs {maxAbs}");
            Assert.True(corr > 0.9999, $"corr {corr}");
        }
    }

    [Fact]
    public void VaeDecode_MatchReference()
    {
        (SafeTensorsLoader dit, SafeTensorsLoader vae, SafeTensorsLoader refIo)? loaded = Load();
        if (loaded is null) return; // gated
        (SafeTensorsLoader dit, SafeTensorsLoader vae, SafeTensorsLoader refIo) = loaded.Value;
        using (dit) using (vae) using (refIo)
        {
            using IBackend backend = MakeBackend();
            OasisVitVae model = new(); model.LoadWeights(vae.GetAllTensors());

            Tensor refLat = refIo.GetTensor("vae_dec_latent");    // [1,576,16]
            int s = 18 * 32, lat = 16;
            Tensor latent = new(new TensorShape(1, lat, 18, 32), DType.F32);
            float* lp = (float*)latent.DataPointer; float* rp = (float*)refLat.DataPointer;
            for (int i = 0; i < s; i++) for (int c = 0; c < lat; c++) lp[(long)c * s + i] = rp[(long)i * lat + c];

            Tensor rgb = model.Decode(backend, latent); latent.Dispose(); // [1,3,360,640]
            Tensor refRgb = refIo.GetTensor("vae_decode_rgb");
            (double maxAbs, double corr) = Compare(rgb, refRgb);
            _out.WriteLine($"VAE decode rgb: maxAbs={maxAbs:E3} corr={corr:F8}");
            rgb.Dispose();
            Assert.True(maxAbs < AbsTol, $"maxAbs {maxAbs}");
            Assert.True(corr > 0.9999, $"corr {corr}");
        }
    }

    [Fact]
    public void DitForward_MatchReference()
    {
        (SafeTensorsLoader dit, SafeTensorsLoader vae, SafeTensorsLoader refIo)? loaded = Load();
        if (loaded is null) return; // gated
        (SafeTensorsLoader dit, SafeTensorsLoader vae, SafeTensorsLoader refIo) = loaded.Value;
        using (dit) using (vae) using (refIo)
        {
            using IBackend backend = MakeBackend();
            OasisDit model = new(OasisDitConfig.S2); model.LoadWeights(dit.GetAllTensors());

            Tensor xB = refIo.GetTensor("dit_x");        // [1,T,16,18,32]
            Tensor tB = refIo.GetTensor("dit_t");        // [1,T]
            Tensor condB = refIo.GetTensor("dit_cond");  // [1,T,25]
            int tFrames = (int)xB.Shape[1];

            Tensor latents = new(new TensorShape(tFrames, 16, 18, 32), DType.F32);
            long per = (long)16 * 18 * 32;
            new ReadOnlySpan<float>((float*)xB.DataPointer, (int)(tFrames * per)).CopyTo(new Span<float>((float*)latents.DataPointer, (int)(tFrames * per)));

            int[] noiseIdx = new int[tFrames];
            float* tp = (float*)tB.DataPointer;
            for (int f = 0; f < tFrames; f++) noiseIdx[f] = (int)MathF.Round(tp[f]);

            Tensor actions = new(new TensorShape(tFrames, 25), DType.F32);
            new ReadOnlySpan<float>((float*)condB.DataPointer, tFrames * 25).CopyTo(new Span<float>((float*)actions.DataPointer, tFrames * 25));

            Tensor v = model.Forward(backend, latents, noiseIdx, actions);  // [T,16,18,32]
            Tensor refV = refIo.GetTensor("dit_v");  // [1,T,16,18,32]
            (double maxAbs, double corr) = Compare(v, refV);
            _out.WriteLine($"DiT v-pred: maxAbs={maxAbs:E3} corr={corr:F8}");
            v.Dispose(); latents.Dispose(); actions.Dispose();
            Assert.True(maxAbs < AbsTol, $"maxAbs {maxAbs}");
            Assert.True(corr > 0.9999, $"corr {corr}");
        }
    }

    [Fact]
    public void DitBisect_Intermediates()
    {
        (SafeTensorsLoader dit, SafeTensorsLoader vae, SafeTensorsLoader refIo)? loaded = Load();
        if (loaded is null) return; // gated
        (SafeTensorsLoader dit, SafeTensorsLoader vae, SafeTensorsLoader refIo) = loaded.Value;
        using (dit) using (vae) using (refIo)
        {
            using IBackend backend = MakeBackend();
            OasisDit model = new(OasisDitConfig.S2); model.LoadWeights(dit.GetAllTensors());

            Tensor xB = refIo.GetTensor("dit_x");
            Tensor tB = refIo.GetTensor("dit_t");
            Tensor condB = refIo.GetTensor("dit_cond");
            int tFrames = (int)xB.Shape[1];
            Tensor latents = new(new TensorShape(tFrames, 16, 18, 32), DType.F32);
            long per = (long)16 * 18 * 32;
            new ReadOnlySpan<float>((float*)xB.DataPointer, (int)(tFrames * per)).CopyTo(new Span<float>((float*)latents.DataPointer, (int)(tFrames * per)));
            int[] noiseIdx = new int[tFrames];
            float* tp = (float*)tB.DataPointer;
            for (int f = 0; f < tFrames; f++) noiseIdx[f] = (int)MathF.Round(tp[f]);
            Tensor actions = new(new TensorShape(tFrames, 25), DType.F32);
            new ReadOnlySpan<float>((float*)condB.DataPointer, tFrames * 25).CopyTo(new Span<float>((float*)actions.DataPointer, tFrames * 25));

            Dictionary<string, Tensor> taps = new();
            Tensor v = model.Forward(backend, latents, noiseIdx, actions, taps);
            v.Dispose();

            foreach ((string tap, string refName) in new[] { ("c", "dit_c"), ("xembed", "dit_xembed"), ("block0", "dit_block0"), ("blockLast", "dit_blockLast") })
            {
                (double maxAbs, double corr) = Compare(taps[tap], refIo.GetTensor(refName));
                _out.WriteLine($"DiT tap {tap,-8}: maxAbs={maxAbs:E3} corr={corr:F8}");
            }
        }
    }

    private static (double, double) Compare(Tensor a, Tensor b)
    {
        long n = a.ElementCount;
        float* pa = (float*)a.DataPointer; float* pb = (float*)b.DataPointer;
        double maxAbs = 0, dot = 0, na = 0, nb = 0;
        for (long i = 0; i < n; i++)
        {
            double x = pa[i], y = pb[i];
            maxAbs = Math.Max(maxAbs, Math.Abs(x - y)); dot += x * y; na += x * x; nb += y * y;
        }
        return (maxAbs, dot / (Math.Sqrt(na * nb) + 1e-12));
    }

    /// <summary>Compares C# channel-major <c>[c*s + i]</c> against ref token-major <c>[i*lat + c]</c>.</summary>
    private static (double, double) CompareIndexed(float* cs, float* rf, int s, int lat, bool transpose)
    {
        double maxAbs = 0, dot = 0, na = 0, nb = 0;
        for (int i = 0; i < s; i++)
            for (int c = 0; c < lat; c++)
            {
                double x = cs[(long)c * s + i], y = rf[(long)i * lat + c];
                maxAbs = Math.Max(maxAbs, Math.Abs(x - y)); dot += x * y; na += x * x; nb += y * y;
            }
        return (maxAbs, dot / (Math.Sqrt(na * nb) + 1e-12));
    }
}
