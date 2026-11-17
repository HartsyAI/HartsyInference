using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Engine.Features;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ThreeD.Models.Idol;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.ThreeD.Tests.Parity;

/// <summary>Idol-turbo on the CPU in float32 against the official <c>turbo.py</c> at idol-sdk <c>956439f</c> (the dumps of
/// <c>tests/python-reference/idol_reference/dump_idol_denoiser.py</c> and <c>dump_idol_sampler.py</c>): one forward per phase expert at the dense layout, then the whole five-step
/// guidance-free sampler from the reference's own noise. The engine takes the Comfy-format slow and fast files (<c>idol-turbo/loras/</c> beside <c>idol/</c> in <c>HARTSY_IDOL_MODELS</c>),
/// which the dump script checked hold the same deltas as the embedded <c>idol_turbo_lora.safetensors</c> the official code reads. A missing input skips; with
/// <c>HARTSY_REQUIRE_REAL_WEIGHTS=1</c> it fails.</summary>
[Trait("Category", "Integration")]
public sealed class IdolTurboParityTests
{
    private const string ModelsEnv = "HARTSY_IDOL_MODELS";
    private const string ReferenceEnv = "HARTSY_IDOL_REF";
    private const string RequireEnv = "HARTSY_REQUIRE_REAL_WEIGHTS";

    // Peak-relative, as for the base denoiser; the five-step loop compounds five forwards, so it gets a looser limit.
    private const double ForwardTolerance = 3e-4;
    private const double SamplerTolerance = 2e-3;

    private readonly ITestOutputHelper _output;

    public IdolTurboParityTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void PhaseExperts_MatchTheOfficialTurboOnTheCpu()
    {
        if (!TryOpen(out Inputs? inputs))
            return;
        using (inputs)
        {
            using IdolTurboAdapters turbo = IdolTurboAdapters.Attach(new CpuBackend(), inputs.Weights, inputs.SlowPath, inputs.FastPath);
            using IdolDenoiser model = new(turbo.Weights);
            CpuBackend backend = new();
            using Tensor context = model.EmbedContext(backend, inputs.Cond["feature1"], inputs.Cond["feature2"]);
            Dictionary<string, Tensor> want = inputs.Want("den", "idol-turbo.safetensors");
            List<string> failures = [];
            foreach ((string phase, float t) in new[] { ("slow", 800f), ("fast", 200f) })
            {
                using IDisposable scope = turbo.Phase(t);
                (Tensor velocity, Tensor camera) = model.Forward(backend, inputs.Denoiser["latent"], inputs.Denoiser["camera"], t, context);
                using (velocity)
                using (camera)
                {
                    Report($"{phase} v_latent (t={t})", velocity, want[$"{phase}_v_latent"], ForwardTolerance, failures);
                    Report($"{phase} v_camera (t={t})", camera, want[$"{phase}_v_camera"], ForwardTolerance, failures);
                }
            }

            Assert.True(failures.Count == 0, string.Join("; ", failures));
        }
    }

    [Fact]
    public void FiveStepSampler_MatchesTheOfficialTurboLoopOnTheCpu()
    {
        if (!TryOpen(out Inputs? inputs))
            return;
        using (inputs)
        {
            CpuBackend backend = new();
            using IdolTurboAdapters turbo = IdolTurboAdapters.Attach(backend, inputs.Weights, inputs.SlowPath, inputs.FastPath);
            using IdolDenoiser model = new(turbo.Weights);
            using Tensor context = model.EmbedContext(backend, inputs.Cond["feature1"], inputs.Cond["feature2"]);
            Dictionary<string, Tensor> want = inputs.Want("samp", "sampler-idol-turbo.safetensors");
            (Tensor latent, Tensor camera) = IdolSampler.Sample(backend, model, context, context, inputs.Denoiser["latent"], inputs.Denoiser["camera"],
                IdolTurboAdapters.DefaultSteps, guidance: 0f, IdolSampler.DefaultRescaleT, passScope: turbo.Phase);
            List<string> failures = [];
            using (latent)
            using (camera)
            {
                Report("turbo 5-step latent", latent, want["latent"], SamplerTolerance, failures);
                Report("turbo 5-step camera", camera, want["camera"], SamplerTolerance, failures);
            }

            Assert.True(failures.Count == 0, string.Join("; ", failures));
        }
    }

    private sealed class Inputs : IDisposable
    {
        private readonly List<SafeTensorsLoader> _loaders = [];
        private readonly string _refDir;

        public Inputs(string refDir, IdolWeights weights, string slow, string fast, Dictionary<string, Tensor> cond, Dictionary<string, Tensor> denoiser, params SafeTensorsLoader[] loaders)
        {
            _refDir = refDir;
            Weights = weights;
            SlowPath = slow;
            FastPath = fast;
            Cond = cond;
            Denoiser = denoiser;
            _loaders.AddRange(loaders);
        }

        public IdolWeights Weights { get; }
        public string SlowPath { get; }
        public string FastPath { get; }
        public Dictionary<string, Tensor> Cond { get; }
        public Dictionary<string, Tensor> Denoiser { get; }

        public Dictionary<string, Tensor> Want(string folder, string file)
        {
            SafeTensorsLoader loader = new();
            loader.Load(Path.Combine(_refDir, folder, file));
            _loaders.Add(loader);
            return loader.GetAllTensors();
        }

        public void Dispose()
        {
            Weights.Dispose();
            foreach (SafeTensorsLoader loader in _loaders)
                loader.Dispose();
        }
    }

    private bool TryOpen([NotNullWhen(true)] out Inputs? inputs)
    {
        inputs = null;
        string? models = Environment.GetEnvironmentVariable(ModelsEnv);
        string? refDir = Environment.GetEnvironmentVariable(ReferenceEnv);
        string checkpoint = Path.Combine(models ?? "", "idol", "diffusion_models", "idol_fp16.safetensors");
        string slow = Path.Combine(models ?? "", "idol-turbo", "loras", "idol_turbo_slow_comfy.safetensors");
        string fast = Path.Combine(models ?? "", "idol-turbo", "loras", "idol_turbo_fast_comfy.safetensors");
        string conditioning = Path.Combine(refDir ?? "", "cond", "conditioning.safetensors");
        string denoiserInputs = Path.Combine(refDir ?? "", "den", "denoiser_inputs.safetensors");
        string[] required =
        [
            checkpoint, slow, fast, conditioning, denoiserInputs,
            Path.Combine(refDir ?? "", "den", "idol-turbo.safetensors"), Path.Combine(refDir ?? "", "samp", "sampler-idol-turbo.safetensors"),
        ];
        string[] missing = required.Where(p => !File.Exists(p)).ToArray();
        if (missing.Length > 0)
        {
            string message = $"required real-weight asset(s) missing: {string.Join(", ", missing)}";
            if (Environment.GetEnvironmentVariable(RequireEnv) == "1")
                throw new InvalidOperationException($"{RequireEnv}=1 but {message}.");
            _output.WriteLine($"SKIPPED: {message}");
            return false;
        }

        SafeTensorsLoader condLoader = new(), inputLoader = new();
        condLoader.Load(conditioning);
        inputLoader.Load(denoiserInputs);
        inputs = new Inputs(refDir!, IdolWeights.Load(checkpoint), slow, fast, condLoader.GetAllTensors(), inputLoader.GetAllTensors(), condLoader, inputLoader);
        return true;
    }

    private void Report(string stage, Tensor got, Tensor want, double tolerance, List<string> failures)
    {
        ReadOnlySpan<float> g = got.AsReadOnlySpan<float>(), w = want.AsReadOnlySpan<float>();
        Assert.Equal(w.Length, g.Length);
        (double rel, double maxAbs) = ParityCompare.RelativeL2(g, w);
        double peak = 0;
        foreach (float v in w) peak = Math.Max(peak, Math.Abs(v));
        double peakRelative = maxAbs / peak;
        bool pass = peakRelative <= tolerance;
        _output.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"{stage}: max abs {maxAbs:E3}, relRMS {rel:E2}, reference peak {peak:E3}, peak-relative {peakRelative:E2} vs {tolerance:E0}, {(pass ? "pass" : "FAIL")}"));
        if (!pass) failures.Add($"{stage} peak-relative error {peakRelative:E2} exceeds {tolerance:E0}");
    }
}
