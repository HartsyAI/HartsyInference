using System.Globalization;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ThreeD.Models.Idol;
using HartsyInference.ThreeD.Models.Idol.Budget;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.ThreeD.Tests.Parity;

/// <summary>Budget3D's sparse denoiser against the reference (<c>dump_budget3d.py</c> on BudgetAny's code at <c>95bbeab</c>, float32 on the CPU): one velocity at the time 731 on the layout of each budget, built from the reference's
/// own layout indices, from the saved latent, camera and image tokens. The CPU test runs the smallest budget (all tokens are 8³ cubes, so it covers the mixers, the position encoder and both refine stages); the CUDA test runs
/// every budget, including the s = 2 and s = 1 tokens of the larger ones. Needs the Budget3D checkpoint (<c>HARTSY_IDOL_MODELS</c>, the folder that holds <c>budget3d-idol/</c>) and the dump folder (<c>HARTSY_IDOL_REF</c>).</summary>
[Trait("Category", "Integration")]
public sealed class BudgetDenoiserParityTests
{
    private const string ModelsEnv = "HARTSY_IDOL_MODELS";
    private const string ReferenceEnv = "HARTSY_IDOL_REF";
    private const string RequireEnv = "HARTSY_REQUIRE_REAL_WEIGHTS";

    private readonly ITestOutputHelper _output;

    public BudgetDenoiserParityTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void SmallestBudget_MatchesTheReferenceOnTheCpu() => Run(new CpuBackend(), cuda: false, [0.0625], peakLimit: 3e-4, rmsLimit: 1e-4);

    [Fact]
    [Trait("Category", "GpuIntegration")]
    public void EveryBudget_MatchesTheReferenceOnCuda()
    {
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }

        using CudaBackend backend = new(0, PtxDir());
        Run(backend, cuda: true, [0.0625, 0.1, 0.25, 0.5, 1.0], peakLimit: 5e-3, rmsLimit: 2e-3);
    }

    private void Run(IBackend backend, bool cuda, double[] budgets, double peakLimit, double rmsLimit)
    {
        string? models = Environment.GetEnvironmentVariable(ModelsEnv);
        string? refDir = Environment.GetEnvironmentVariable(ReferenceEnv);
        string checkpoint = Path.Combine(models ?? "", "budget3d-idol", "budget3d_idol.safetensors");
        string conditioning = Path.Combine(refDir ?? "", "cond", "conditioning.safetensors");
        string inputs = Path.Combine(refDir ?? "", "den", "denoiser_inputs.safetensors");
        string layouts = Path.Combine(refDir ?? "", "b3d", "layouts.safetensors");
        string reference = Path.Combine(refDir ?? "", "b3d", "budget3d-idol.safetensors");
        string[] missing = new[] { checkpoint, conditioning, inputs, layouts, reference }.Where(p => !File.Exists(p)).ToArray();
        if (missing.Length > 0)
        {
            string message = $"required real-weight asset(s) missing: {string.Join(", ", missing)}";
            if (Environment.GetEnvironmentVariable(RequireEnv) == "1")
                throw new InvalidOperationException($"{RequireEnv}=1 but {message}.");
            _output.WriteLine($"SKIPPED: {message}");
            return;
        }

        using IdolWeights weights = IdolWeights.Load(checkpoint);
        using BudgetAdapterWeights adapters = BudgetAdapterWeights.Load(checkpoint);
        using IdolDenoiser trunk = new(weights);
        using BudgetDenoiser model = new(trunk, adapters);
        using SafeTensorsLoader condLoader = new(), inputLoader = new(), layoutLoader = new(), refLoader = new();
        condLoader.Load(conditioning);
        inputLoader.Load(inputs);
        layoutLoader.Load(layouts);
        refLoader.Load(reference);
        Dictionary<string, Tensor> cond = condLoader.GetAllTensors(), input = inputLoader.GetAllTensors(), layout = layoutLoader.GetAllTensors(), want = refLoader.GetAllTensors();

        if (cuda)
        {
            backend.PreloadWeights(trunk.EnumerateWeights());
            backend.PreloadWeights(model.EnumerateWeights());
        }

        List<string> failures = [];
        try
        {
            using Tensor context = trunk.EmbedContext(backend, cond["feature1"], cond["feature2"]);
            foreach (double budget in budgets)
            {
                string key = ("b" + budget.ToString("0.####", CultureInfo.InvariantCulture)).Replace(".", "p");
                long[] k = layout[$"{key}_k4k2"].AsReadOnlySpan<long>().ToArray();
                BudgetLayout shape = BudgetLayout.FromIndices(Ints(layout[$"{key}_idx4"]), Ints(layout[$"{key}_idx2"]), Ints(layout[$"{key}_idx1"]), (int)k[0], (int)k[1]);
                (Tensor velocity, Tensor camera) = model.Forward(backend, input["latent"], input["camera"], 731f, context, shape);
                using (velocity)
                using (camera)
                {
                    backend.Sync();
                    Report($"budget {budget} ({shape.Tokens} tokens) v_latent", velocity, want[$"{key}_v_latent"], peakLimit, rmsLimit, failures);
                    Report($"budget {budget} ({shape.Tokens} tokens) v_camera", camera, want[$"{key}_v_camera"], peakLimit, rmsLimit, failures);
                }
            }
        }
        finally
        {
            backend.Sync();
            if (cuda)
            {
                backend.FreeWeights(trunk.EnumerateWeights());
                backend.FreeWeights(model.EnumerateWeights());
            }
        }

        Assert.True(failures.Count == 0, string.Join("; ", failures));
    }

    private void Report(string stage, Tensor got, Tensor want, double peakLimit, double rmsLimit, List<string> failures)
    {
        ReadOnlySpan<float> g = got.AsReadOnlySpan<float>(), w = want.AsReadOnlySpan<float>();
        Assert.Equal(w.Length, g.Length);
        (double rel, double maxAbs) = ParityCompare.RelativeL2(g, w);
        double peak = 0;
        foreach (float v in w) peak = Math.Max(peak, Math.Abs(v));
        double peakRelative = maxAbs / peak;
        bool pass = peakRelative <= peakLimit && rel <= rmsLimit;
        _output.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"{stage}: max abs {maxAbs:E3}, relRMS {rel:E2}, reference peak {peak:E3}, peak-relative {peakRelative:E2} vs {peakLimit:E0}, relRMS limit {rmsLimit:E0}, {(pass ? "pass" : "FAIL")}"));
        if (!pass)
            failures.Add($"{stage}: peak-relative {peakRelative:E2}, relRMS {rel:E2}");
    }

    private static int[] Ints(Tensor t) => [.. t.AsReadOnlySpan<long>().ToArray().Select(v => (int)v)];

    private static string PtxDir()
    {
        for (DirectoryInfo? dir = new(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            string candidate = Path.Combine(dir.FullName, "src", "HartsyInference.Cuda", "Ptx");
            if (Directory.Exists(candidate))
                return candidate;
        }

        throw new DirectoryNotFoundException("src/HartsyInference.Cuda/Ptx not found above the test output directory.");
    }
}
