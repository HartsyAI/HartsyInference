using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Sampling;
using HartsyInference.Diffusion.Schedulers;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Correctness gates for the sampler core, all synthetic — no checkpoint, no GPU.
///
/// <para>Two properties carry this suite. The first is <b>bit-identity of the default path</b>: a generation naming no
/// sampler must execute the same device op it did before <see cref="ISampler"/> existed, or every pipeline conversion
/// silently becomes a quality change nobody signed off on. The second is the <b>constant-denoiser trajectory</b>, which
/// is an exact analytic solution every consistent sampler must reproduce — it catches sign flips and wrong
/// coefficients, the errors that otherwise only show up as a subtly worse image.</para></summary>
public sealed class SamplerCoreTests
{
    private static readonly TensorShape Shape = new TensorShape(1, 4, 8, 8);

    /// <summary>A denoiser whose x0 estimate is a fixed constant everywhere, whatever the noise level.
    ///
    /// <para>This makes the probability-flow ODE exactly linear: from any <c>x = c + σ·k</c> the true solution at the
    /// next noise level is <c>c + σ_next·k</c>, and at the terminal <c>σ = 0</c> it is exactly <c>c</c>. A correct
    /// deterministic sampler of ANY order lands on <c>c</c> to floating-point precision; one with a sign error or a
    /// mis-derived coefficient does not.</para></summary>
    private sealed class ConstantDenoiser : IDenoisePredictor
    {
        private readonly IBackend _backend;
        private readonly float _constant;

        public ConstantDenoiser(IBackend backend, float constant)
        {
            _backend = backend;
            _constant = constant;
        }

        public int Evaluations { get; private set; }

        public PredictionType Prediction => PredictionType.Epsilon;

        public float GuidanceScale => 1.0f;

        /// <summary>Returns the epsilon that makes <c>x0 = x − σ·eps</c> equal the constant: <c>eps = (x − c)/σ</c>.</summary>
        public DenoisePrediction Predict(Tensor x, float sigma, int stepIndex)
        {
            Evaluations++;
            Tensor eps = new Tensor(x.Shape, DType.F32);
            unsafe
            {
                float* src = (float*)x.DataPointer;
                float* dst = (float*)eps.DataPointer;
                for (long i = 0; i < x.ElementCount; i++)
                {
                    dst[i] = (src[i] - _constant) / sigma;
                }
            }
            return new DenoisePrediction(eps, eps);
        }
    }

    /// <summary>The flow-matching counterpart of <see cref="ConstantDenoiser"/>. Flow families return
    /// <c>v = noise − x0</c> on a rectified path with sigma in (0,1], and <see cref="SamplerMath.ToDenoised"/> claims
    /// the SAME <c>x0 = x − σ·pred</c> formula serves both domains. This exercises that claim: if the shared formula
    /// were wrong for flow, every DiT family — Flux, Qwen-Image, Krea2, Z-Image, the other 20+ — would sample subtly
    /// wrong the moment it moved past plain Euler, with nothing to point at.</summary>
    private sealed class ConstantFlowDenoiser : IDenoisePredictor
    {
        private readonly float _constant;

        public ConstantFlowDenoiser(float constant) => _constant = constant;

        public PredictionType Prediction => PredictionType.FlowVelocity;

        public float GuidanceScale => 1.0f;

        public DenoisePrediction Predict(Tensor x, float sigma, int stepIndex)
        {
            Tensor velocity = new Tensor(x.Shape, DType.F32);
            unsafe
            {
                float* src = (float*)x.DataPointer;
                float* dst = (float*)velocity.DataPointer;
                for (long i = 0; i < x.ElementCount; i++)
                {
                    dst[i] = (src[i] - _constant) / sigma;
                }
            }
            return new DenoisePrediction(velocity, velocity);
        }
    }

    private static float[] Sigmas(int steps, float max = 14.6f, float min = 0.03f)
        => SigmaSchedule.Karras(min, max, steps);

    /// <summary>A flow-match sigma grid: linear from 1 down to the terminal zero, the shape
    /// <see cref="FlowMatchEulerDiscreteScheduler"/> produces.</summary>
    private static float[] FlowSigmas(int steps)
    {
        float[] sigmas = new float[steps + 1];
        for (int i = 0; i < steps; i++)
        {
            sigmas[i] = 1.0f - ((float)i / steps);
        }
        sigmas[steps] = 0f;
        return sigmas;
    }

    private static unsafe Tensor Filled(float value)
    {
        Tensor t = new Tensor(Shape, DType.F32);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++)
        {
            p[i] = value;
        }
        return t;
    }

    private static unsafe float[] Read(Tensor t)
    {
        float[] data = new float[t.ElementCount];
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++)
        {
            data[i] = p[i];
        }
        return data;
    }

    /// <summary>THE rollout gate. Driving <see cref="EulerSampler"/> must produce byte-identical output to calling
    /// <see cref="IBackend.CfgEulerStep"/> directly in a hand-written loop — the exact code every pipeline had before
    /// the sampler seam. Exact equality, not a tolerance: the sampler is supposed to be the same instructions, not
    /// merely equivalent arithmetic, and a tolerance here would hide a real reordering.</summary>
    [Fact]
    public void EulerSampler_IsBitIdenticalToDirectCfgEulerStep()
    {
        IBackend backend = new CpuBackend();
        float[] sigmas = Sigmas(12);
        const float Constant = 0.25f;

        using Tensor viaSampler = Filled(3.0f);
        ConstantDenoiser samplerDenoiser = new ConstantDenoiser(backend, Constant);
        EulerSampler sampler = new EulerSampler(sigmas);
        sampler.Reset(Shape);
        for (int i = 0; i < sigmas.Length - 1; i++)
        {
            sampler.Step(backend, viaSampler, samplerDenoiser, i);
        }

        using Tensor viaDirect = Filled(3.0f);
        ConstantDenoiser directDenoiser = new ConstantDenoiser(backend, Constant);
        for (int i = 0; i < sigmas.Length - 1; i++)
        {
            using DenoisePrediction prediction = directDenoiser.Predict(viaDirect, sigmas[i], i);
            backend.CfgEulerStep(viaDirect, prediction.Cond, prediction.Uncond, 1.0f, sigmas[i + 1] - sigmas[i]);
        }

        Assert.Equal(Read(viaDirect), Read(viaSampler));
        Assert.Equal(directDenoiser.Evaluations, samplerDenoiser.Evaluations);
    }

    /// <summary>Every deterministic sampler must solve the constant-denoiser ODE exactly, landing on the constant at
    /// the terminal sigma. The tolerance is loose only because the multistep methods accumulate F32 rounding over 12
    /// steps; a coefficient or sign error misses by orders of magnitude, not by 1e-3.</summary>
    [Theory]
    [InlineData("euler")]
    [InlineData("heun")]
    [InlineData("dpm_2")]
    [InlineData("lms")]
    [InlineData("dpmpp_2m")]
    public void DeterministicSampler_SolvesTheConstantDenoiserExactly(string name)
    {
        IBackend backend = new CpuBackend();
        float[] sigmas = Sigmas(12);
        const float Constant = 0.25f;

        using Tensor z = Filled(3.0f);
        ConstantDenoiser denoiser = new ConstantDenoiser(backend, Constant);
        ISampler sampler = SamplerRegistry.Create(name, sigmas, seed: 7);
        sampler.Reset(Shape);
        for (int i = 0; i < sigmas.Length - 1; i++)
        {
            sampler.Step(backend, z, denoiser, i);
        }

        foreach (float value in Read(z))
        {
            Assert.True(MathF.Abs(value - Constant) < 1e-3f,
                $"{name} landed on {value}, expected the denoised constant {Constant}.");
        }
    }

    /// <summary>Second-order samplers must actually evaluate the model twice per step. If one silently degraded to a
    /// single evaluation it would still pass the trajectory test above (the constant denoiser is exact at any order)
    /// while being a different, cheaper, lower-accuracy sampler on a real model.</summary>
    [Theory]
    [InlineData("heun", 2)]
    [InlineData("dpm_2", 2)]
    [InlineData("dpmpp_2s_ancestral", 2)]
    [InlineData("euler", 1)]
    [InlineData("dpmpp_2m", 1)]
    public void SamplerEvaluatesTheModelTheExpectedNumberOfTimesPerStep(string name, int perStep)
    {
        IBackend backend = new CpuBackend();
        float[] sigmas = Sigmas(6);
        using Tensor z = Filled(3.0f);
        ConstantDenoiser denoiser = new ConstantDenoiser(backend, 0.25f);
        ISampler sampler = SamplerRegistry.Create(name, sigmas, seed: 7);
        sampler.Reset(Shape);

        // Only the non-terminal steps: every second-order method deliberately degrades to one evaluation at sigma 0.
        int steps = sigmas.Length - 2;
        for (int i = 0; i < steps; i++)
        {
            sampler.Step(backend, z, denoiser, i);
        }
        Assert.Equal(steps * perStep, denoiser.Evaluations);
    }

    /// <summary>Stochastic samplers cannot land exactly (they inject noise by design), but they must stay finite and
    /// bounded — an ancestral method that adds noise on the terminal step, or gets the sigma_up/sigma_down split
    /// backwards, blows up or returns visible noise instead of an image.</summary>
    [Theory]
    [InlineData("euler_ancestral")]
    [InlineData("dpm_2_ancestral")]
    [InlineData("dpmpp_2s_ancestral")]
    [InlineData("dpmpp_2m_sde")]
    public void StochasticSampler_StaysFiniteAndConvergesNear(string name)
    {
        IBackend backend = new CpuBackend();
        float[] sigmas = Sigmas(20);
        const float Constant = 0.25f;

        using Tensor z = Filled(3.0f);
        ConstantDenoiser denoiser = new ConstantDenoiser(backend, Constant);
        ISampler sampler = SamplerRegistry.Create(name, sigmas, seed: 1234);
        sampler.Reset(Shape);
        for (int i = 0; i < sigmas.Length - 1; i++)
        {
            sampler.Step(backend, z, denoiser, i);
        }

        float[] result = Read(z);
        Assert.All(result, v => Assert.True(float.IsFinite(v), $"{name} produced a non-finite value {v}."));
        // The terminal step has sigma_up == 0 for every ancestral method, so the run must still END on the constant.
        foreach (float value in result)
        {
            Assert.True(MathF.Abs(value - Constant) < 1e-2f,
                $"{name} ended at {value}, too far from the denoised constant {Constant} — the terminal step is "
                + "supposed to add no noise.");
        }
    }

    /// <summary>The same samplers must solve the FLOW-matching form exactly too. This is the property that lets one
    /// sampler set serve all 26 image families instead of one set per prediction domain — and the one that fails
    /// silently if <see cref="SamplerMath.ToDenoised"/>'s shared epsilon/flow formula is wrong.</summary>
    [Theory]
    [InlineData("euler")]
    [InlineData("heun")]
    [InlineData("dpm_2")]
    [InlineData("dpmpp_2m")]
    public void DeterministicSampler_SolvesTheFlowMatchingFormExactly(string name)
    {
        IBackend backend = new CpuBackend();
        float[] sigmas = FlowSigmas(16);
        const float Constant = -0.75f;

        using Tensor z = Filled(2.0f);
        ConstantFlowDenoiser denoiser = new ConstantFlowDenoiser(Constant);
        ISampler sampler = SamplerRegistry.Create(name, sigmas, seed: 3);
        sampler.Reset(Shape);
        for (int i = 0; i < sigmas.Length - 1; i++)
        {
            sampler.Step(backend, z, denoiser, i);
        }

        foreach (float value in Read(z))
        {
            Assert.True(MathF.Abs(value - Constant) < 1e-3f,
                $"{name} on flow-matching sigmas landed on {value}, expected {Constant}.");
        }
    }

    /// <summary><see cref="EulerSampler"/> refuses a v-prediction model rather than mis-integrating it. Its fused
    /// update is the epsilon/flow expression; silently applying it to v-prediction would produce mud with no error.</summary>
    [Fact]
    public void EulerSampler_RefusesVPrediction()
    {
        IBackend backend = new CpuBackend();
        using Tensor z = Filled(1.0f);
        EulerSampler sampler = new EulerSampler(Sigmas(4));
        sampler.Reset(Shape);
        Assert.Throws<NotSupportedException>(() => sampler.Step(backend, z, new VPredictionDenoiser(), 0));
    }

    private sealed class VPredictionDenoiser : IDenoisePredictor
    {
        public PredictionType Prediction => PredictionType.VPrediction;

        public float GuidanceScale => 1.0f;

        public DenoisePrediction Predict(Tensor x, float sigma, int stepIndex)
        {
            Tensor v = new Tensor(x.Shape, DType.F32);
            return new DenoisePrediction(v, v);
        }
    }

    /// <summary>Same seed, same trajectory. A stochastic sampler that drew from a shared or time-based source would
    /// make generations unreproducible, which is worse than a slightly different image.</summary>
    [Fact]
    public void StochasticSampler_IsReproducibleForAGivenSeed()
    {
        IBackend backend = new CpuBackend();
        float[] sigmas = Sigmas(8);

        float[] RunOnce(int seed)
        {
            using Tensor z = Filled(3.0f);
            ConstantDenoiser denoiser = new ConstantDenoiser(backend, 0.25f);
            ISampler sampler = SamplerRegistry.Create("euler_ancestral", sigmas, seed);
            sampler.Reset(Shape);
            for (int i = 0; i < sigmas.Length - 1; i++)
            {
                sampler.Step(backend, z, denoiser, i);
            }
            return Read(z);
        }

        Assert.Equal(RunOnce(99), RunOnce(99));
    }

    /// <summary>A sampler must not carry history across generations. <see cref="DpmPlusPlus2MSampler"/> extrapolates
    /// from the previous step's denoised estimate; if <see cref="ISampler.Reset"/> failed to clear it, the first step
    /// of the SECOND image would extrapolate from the FIRST image's latent — a cross-generation contamination that no
    /// single-run test can see.</summary>
    [Fact]
    public void MultistepSampler_ResetClearsHistoryBetweenRuns()
    {
        IBackend backend = new CpuBackend();
        float[] sigmas = Sigmas(10);
        ISampler sampler = SamplerRegistry.Create("dpmpp_2m", sigmas, seed: 0);

        float[] Run()
        {
            using Tensor z = Filled(3.0f);
            ConstantDenoiser denoiser = new ConstantDenoiser(backend, 0.25f);
            sampler.Reset(Shape);
            for (int i = 0; i < sigmas.Length - 1; i++)
            {
                sampler.Step(backend, z, denoiser, i);
            }
            return Read(z);
        }

        Assert.Equal(Run(), Run());
    }
}
