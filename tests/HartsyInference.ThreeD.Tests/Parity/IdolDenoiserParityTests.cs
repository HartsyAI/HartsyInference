using System.Globalization;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ThreeD.Models.Idol;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.ThreeD.Tests.Parity;

/// <summary>Real-weight parity of Idol's denoiser on the CPU in float32 against idol-sdk at <c>956439f</c> (the dumps of
/// <c>tests/python-reference/idol_reference/dump_idol_denoiser.py</c>): the image stream, then one forward at the dense layout with the
/// stage taps of the dump. Needs the Comfy-Org/Idol repack and the pinned token positions (<c>HARTSY_IDOL_MODELS</c>, the folder that
/// holds <c>idol/</c>) and the dump folder (<c>HARTSY_IDOL_REF</c>). A missing input skips; with <c>HARTSY_REQUIRE_REAL_WEIGHTS=1</c> it fails.</summary>
[Trait("Category", "Integration")]
public sealed class IdolDenoiserParityTests
{
    private const string ModelsEnv = "HARTSY_IDOL_MODELS";
    private const string ReferenceEnv = "HARTSY_IDOL_REF";
    private const string RequireEnv = "HARTSY_REQUIRE_REAL_WEIGHTS";

    // Peak-relative: max abs error over the stage's peak |reference|. FP32 CPU against FP32 CPU; the starting points are the add-model
    // values (GEMM 1e-5, attention 1e-4, full denoiser 1e-3).
    private const double EarlyTolerance = 1e-4;
    private const double BlockTolerance = 3e-4;

    private readonly ITestOutputHelper _output;

    public IdolDenoiserParityTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void DenseForward_MatchesIdolSdkOnTheCpu()
    {
        string? models = Environment.GetEnvironmentVariable(ModelsEnv);
        string? refDir = Environment.GetEnvironmentVariable(ReferenceEnv);
        string checkpoint = Path.Combine(models ?? "", "idol", "diffusion_models", "idol_fp16.safetensors");
        string posPe = Path.Combine(models ?? "", "idol", IdolWeights.PosPeFileName);
        string conditioning = Path.Combine(refDir ?? "", "cond", "conditioning.safetensors");
        string inputs = Path.Combine(refDir ?? "", "den", "denoiser_inputs.safetensors");
        string reference = Path.Combine(refDir ?? "", "den", "idol.safetensors");
        string[] missing = new[] { checkpoint, posPe, conditioning, inputs, reference }.Where(p => !File.Exists(p)).ToArray();
        if (missing.Length > 0)
        {
            string message = $"required real-weight asset(s) missing: {string.Join(", ", missing)}";
            if (Environment.GetEnvironmentVariable(RequireEnv) == "1")
                throw new InvalidOperationException($"{RequireEnv}=1 but {message}.");
            _output.WriteLine($"SKIPPED: {message}");
            return;
        }

        using IdolWeights weights = IdolWeights.Load(checkpoint, posPe);
        using IdolDenoiser model = new(weights);
        using SafeTensorsLoader condLoader = new(), inputLoader = new(), refLoader = new();
        condLoader.Load(conditioning);
        inputLoader.Load(inputs);
        refLoader.Load(reference);
        Dictionary<string, Tensor> cond = condLoader.GetAllTensors(), input = inputLoader.GetAllTensors(), want = refLoader.GetAllTensors();

        CpuBackend backend = new();
        List<string> failures = [];
        Dictionary<string, Tensor> taps = new(StringComparer.Ordinal);
        void Tap(string name, Tensor t)
        {
            Tensor copy = new(t.Shape, DType.F32);
            t.AsReadOnlySpan<float>().CopyTo(copy.AsSpan<float>());
            taps[name] = copy;
        }

        try
        {
            using Tensor context = model.EmbedContext(backend, cond["feature1"], cond["feature2"], Tap);
            Report("after_context_refiner", taps["after_context_refiner"], want["after_context_refiner"], EarlyTolerance, failures);

            (Tensor velocity, Tensor camera) = model.Forward(backend, input["latent"], input["camera"], 731f, context, Tap);
            using (velocity)
            using (camera)
            {
                Report("embedded_tokens", taps["embedded_tokens"], want["embedded_tokens"], EarlyTolerance, failures);
                Report("after_noise_refiner", taps["after_noise_refiner"], want["after_noise_refiner"], EarlyTolerance, failures);
                ReportRotation(taps["repo0_cos"], taps["repo0_sin"], want["repo0_cos_sin"], failures);
                Report("after_block0", taps["after_block0"], want["after_block0"], BlockTolerance, failures);
                Report("after_block23", taps["after_block23"], want["after_block23"], BlockTolerance, failures);
                Report("v_latent", velocity, want["v_latent"], BlockTolerance, failures);
                Report("v_camera", camera, want["v_camera"], BlockTolerance, failures);
            }
        }
        finally
        {
            foreach (Tensor t in taps.Values)
                t.Dispose();
        }

        Assert.True(failures.Count == 0, string.Join("; ", failures));
    }

    /// <summary>The reference taps (cos, sin) as <c>[1, L, H, 32, 2]</c>; the engine holds them as two <c>[L, H, 32]</c> tensors.</summary>
    private void ReportRotation(Tensor cos, Tensor sin, Tensor want, List<string> failures)
    {
        ReadOnlySpan<float> w = want.AsReadOnlySpan<float>();
        float[] c = new float[w.Length / 2], s = new float[w.Length / 2];
        for (int i = 0; i < c.Length; i++)
        {
            c[i] = w[2 * i];
            s[i] = w[2 * i + 1];
        }

        Report("repo0 cos", cos, c, EarlyTolerance, failures);
        Report("repo0 sin", sin, s, EarlyTolerance, failures);
    }

    private void Report(string stage, Tensor got, Tensor want, double tolerance, List<string> failures)
        => Report(stage, got, want.AsReadOnlySpan<float>().ToArray(), tolerance, failures);

    private void Report(string stage, Tensor got, float[] want, double tolerance, List<string> failures)
    {
        ReadOnlySpan<float> g = got.AsReadOnlySpan<float>();
        Assert.Equal(want.Length, g.Length);
        (double rel, double maxAbs) = ParityCompare.RelativeL2(g, want);
        double peak = 0;
        foreach (float v in want) peak = Math.Max(peak, Math.Abs(v));
        double peakRelative = maxAbs / peak;
        bool pass = peakRelative <= tolerance;
        _output.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"{stage}: max abs {maxAbs:E3}, relRMS {rel:E2}, reference peak {peak:E3}, peak-relative {peakRelative:E2} vs {tolerance:E0}, {(pass ? "pass" : "FAIL")}"));
        if (!pass) failures.Add($"{stage} peak-relative error {peakRelative:E2} exceeds {tolerance:E0}");
    }
}
