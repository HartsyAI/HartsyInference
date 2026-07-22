using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers.Diamond;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Real-weight numeric parity for the DIAMOND (eloialonso/diamond) Atari denoiser U-Net + EDM
/// preconditioning vs the upstream reference. Gated on <c>DIAMOND_WEIGHTS</c> (breakout_inner.safetensors,
/// the <c>denoiser.inner_model.*</c> keys) + <c>DIAMOND_REF</c> (diamond_ref_io.safetensors from
/// <c>dump_diamond_reference.py</c>). <c>PARITY_BACKEND=cuda</c> runs on the GPU. Skips cleanly when unset.</summary>
public sealed unsafe class DiamondParityTests
{
    private readonly ITestOutputHelper _out;
    public DiamondParityTests(ITestOutputHelper o) => _out = o;

    private static bool IsCuda => string.Equals(Environment.GetEnvironmentVariable("PARITY_BACKEND"), "cuda", StringComparison.OrdinalIgnoreCase);
    private static double AbsTol => IsCuda ? 5e-2 : 5e-3;

    private static IBackend MakeBackend()
    {
        if (!IsCuda) return new CpuBackend();
        string? d = AppContext.BaseDirectory;
        for (int i = 0; i < 8 && d is not null; i++, d = Path.GetDirectoryName(d))
        {
            string cand = Path.Combine(d, "src", "HartsyInference.Cuda", "Ptx");
            if (Directory.Exists(cand)) return new HartsyInference.Cuda.CudaBackend(0, cand);
        }
        return new HartsyInference.Cuda.CudaBackend(0, Path.Combine(AppContext.BaseDirectory, "Ptx"));
    }

    private static (SafeTensorsLoader w, SafeTensorsLoader r)? Load()
    {
        string? w = Environment.GetEnvironmentVariable("DIAMOND_WEIGHTS");
        string? r = Environment.GetEnvironmentVariable("DIAMOND_REF");
        if (w is null || r is null || !File.Exists(w) || !File.Exists(r)) return null;
        SafeTensorsLoader wl = new(); wl.Load(w);
        SafeTensorsLoader rl = new(); rl.Load(r);
        return (wl, rl);
    }

    [Fact]
    public void InnerModel_Output_MatchReference()
    {
        (SafeTensorsLoader w, SafeTensorsLoader r)? e = Load();
        if (e is null) return; // gated
        using (e.Value.w) using (e.Value.r)
        {
            using IBackend backend = MakeBackend();
            DiamondDenoiser den = new(DiamondConfig.Atari(4));
            den.LoadWeights(e.Value.w.GetAllTensors(), "");

            Tensor obs = e.Value.r.GetTensor("obs");      // [1,12,64,64]
            Tensor noisy = e.Value.r.GetTensor("noisy");  // [1,3,64,64]
            float sigma = ((float*)e.Value.r.GetTensor("sigma").DataPointer)[0];
            int[] act = ReadAct(e.Value.r.GetTensor("act"));

            (float cIn, _, _, float cNoise) = den.ComputeConditioners(sigma);
            Tensor rn = Scale(noisy, cIn);
            Tensor ro = Scale(obs, 1f / den.Config.SigmaData);
            Tensor innerOut = den.Inner.Forward(backend, rn, cNoise, ro, act);
            rn.Dispose(); ro.Dispose();

            (double maxAbs, double corr) = Compare(innerOut, e.Value.r.GetTensor("innerout"));
            _out.WriteLine($"DIAMOND inner U-Net: maxAbs={maxAbs:E3} corr={corr:F8}");
            innerOut.Dispose();
            Assert.True(corr > 0.9999, $"corr {corr}");
            Assert.True(maxAbs < AbsTol, $"maxAbs {maxAbs}");
        }
    }

    [Fact]
    public void Denoise_Quantized_MatchReference()
    {
        (SafeTensorsLoader w, SafeTensorsLoader r)? e = Load();
        if (e is null) return; // gated
        using (e.Value.w) using (e.Value.r)
        {
            using IBackend backend = MakeBackend();
            DiamondDenoiser den = new(DiamondConfig.Atari(4));
            den.LoadWeights(e.Value.w.GetAllTensors(), "");

            Tensor obs = e.Value.r.GetTensor("obs");
            Tensor noisy = e.Value.r.GetTensor("noisy");
            float sigma = ((float*)e.Value.r.GetTensor("sigma").DataPointer)[0];
            int[] act = ReadAct(e.Value.r.GetTensor("act"));

            Tensor d = den.Denoise(backend, noisy, sigma, obs, act, quantize: true);
            (double maxAbs, double corr) = Compare(d, e.Value.r.GetTensor("denoised"));
            _out.WriteLine($"DIAMOND denoise (quantized): maxAbs={maxAbs:E3} corr={corr:F8}");
            d.Dispose();
            Assert.True(corr > 0.9999, $"corr {corr}");
            Assert.True(maxAbs < (IsCuda ? 0.1 : 1e-2), $"maxAbs {maxAbs}");
        }
    }

    [Fact]
    public void Sampler_FixedNoise_MatchReference()
    {
        (SafeTensorsLoader w, SafeTensorsLoader r)? e = Load();
        if (e is null) return; // gated
        using (e.Value.w) using (e.Value.r)
        {
            using IBackend backend = MakeBackend();
            DiamondDenoiser den = new(DiamondConfig.Atari(4));
            den.LoadWeights(e.Value.w.GetAllTensors(), "");
            DiamondSampler sampler = new(den);

            Tensor obs = e.Value.r.GetTensor("obs");
            int[] act = ReadAct(e.Value.r.GetTensor("act"));
            Tensor xInit = e.Value.r.GetTensor("sampler_xinit");
            Tensor outp = sampler.Sample(backend, xInit, obs, act);
            (double maxAbs, double corr) = Compare(outp, e.Value.r.GetTensor("sampler_out"));
            _out.WriteLine($"DIAMOND sampler (3-step Euler): maxAbs={maxAbs:E3} corr={corr:F8}");
            outp.Dispose();
            Assert.True(corr > 0.9999, $"corr {corr}");
            Assert.True(maxAbs < (IsCuda ? 0.1 : 1e-2), $"maxAbs {maxAbs}");
        }
    }

    private static int[] ReadAct(Tensor act)
    {
        int n = (int)act.ElementCount;
        int[] a = new int[n];
        float* p = (float*)act.DataPointer;
        for (int i = 0; i < n; i++) a[i] = (int)MathF.Round(p[i]);
        return a;
    }

    private static Tensor Scale(Tensor x, float s)
    {
        Tensor o = new(x.Shape, DType.F32);
        float* xp = (float*)x.DataPointer, op = (float*)o.DataPointer;
        for (long i = 0; i < x.ElementCount; i++) op[i] = xp[i] * s;
        return o;
    }

    private static (double, double) Compare(Tensor a, Tensor b)
    {
        long n = Math.Min(a.ElementCount, b.ElementCount);
        float* pa = (float*)a.DataPointer, pb = (float*)b.DataPointer;
        double maxAbs = 0, dot = 0, na = 0, nb = 0;
        for (long i = 0; i < n; i++)
        {
            double x = pa[i], y = pb[i];
            maxAbs = Math.Max(maxAbs, Math.Abs(x - y)); dot += x * y; na += x * x; nb += y * y;
        }
        return (maxAbs, dot / (Math.Sqrt(na * nb) + 1e-12));
    }
}
