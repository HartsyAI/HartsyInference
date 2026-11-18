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

    /// <summary>The shipped precision of the Budget3D-Idol denoiser: block matrices and attention in F16 on CUDA, against the float32 reference at every budget.</summary>
    [Fact]
    [Trait("Category", "GpuIntegration")]
    public void EveryBudget_WithF16BlocksOnCuda_StaysCloseToTheFloat32Reference()
    {
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }

        using CudaBackend backend = new(0, PtxDir());
        Run(backend, cuda: true, [0.0625, 0.1, 0.25, 0.5, 1.0], peakLimit: 3e-2, rmsLimit: 6e-3, blockDType: DType.F16);
    }

    /// <summary>Budget3D-Idol-Turbo: its checkpoint embeds the phase experts in the block matrices (<c>.base</c> plus <c>slow_A</c>, <c>slow_B</c>, <c>fast_A</c>, <c>fast_B</c> at scale 4); one forward per budget at the slow (t=800) and
    /// fast (t=200) expert against the reference, with the image stream embedded under the expert of the phase.</summary>
    [Fact]
    [Trait("Category", "GpuIntegration")]
    public void TurboEveryBudgetAndPhase_MatchesTheReferenceOnCuda()
    {
        string? models = Environment.GetEnvironmentVariable(ModelsEnv);
        string? refDir = Environment.GetEnvironmentVariable(ReferenceEnv);
        string checkpoint = Path.Combine(models ?? "", "budget3d-idol-turbo", "budget3d_idol_turbo.safetensors");
        string conditioning = Path.Combine(refDir ?? "", "cond", "conditioning.safetensors");
        string inputs = Path.Combine(refDir ?? "", "den", "denoiser_inputs.safetensors");
        string layouts = Path.Combine(refDir ?? "", "b3d", "layouts.safetensors");
        string reference = Path.Combine(refDir ?? "", "b3d", "budget3d-idol-turbo.safetensors");
        string[] missing = new[] { checkpoint, conditioning, inputs, layouts, reference }.Where(p => !File.Exists(p)).ToArray();
        if (missing.Length > 0 || !CudaContext.IsAvailable())
        {
            string message = missing.Length > 0 ? $"required real-weight asset(s) missing: {string.Join(", ", missing)}" : "CUDA unavailable";
            if (missing.Length > 0 && Environment.GetEnvironmentVariable(RequireEnv) == "1")
                throw new InvalidOperationException($"{RequireEnv}=1 but {message}.");
            _output.WriteLine($"SKIPPED: {message}");
            return;
        }

        using CudaBackend backend = new(0, PtxDir());
        using IdolWeights weights = IdolWeights.Load(checkpoint);
        using BudgetAdapterWeights adapters = BudgetAdapterWeights.Load(checkpoint);
        using IdolTurboAdapters turbo = IdolTurboAdapters.AttachEmbedded(weights);
        using IdolDenoiser trunk = new(turbo.Weights);
        using BudgetDenoiser model = new(trunk, adapters);
        using SafeTensorsLoader condLoader = new(), inputLoader = new(), layoutLoader = new(), refLoader = new();
        condLoader.Load(conditioning);
        inputLoader.Load(inputs);
        layoutLoader.Load(layouts);
        refLoader.Load(reference);
        Dictionary<string, Tensor> cond = condLoader.GetAllTensors(), input = inputLoader.GetAllTensors(), layout = layoutLoader.GetAllTensors(), want = refLoader.GetAllTensors();
        backend.PreloadWeights(trunk.EnumerateWeights());
        backend.PreloadWeights(model.EnumerateWeights());
        List<string> failures = [];
        try
        {
            using PhaseContexts contexts = turbo.EmbedContexts(backend, trunk, cond["feature1"], cond["feature2"]);
            foreach (double budget in new[] { 0.0625, 0.25, 1.0 })
            {
                string key = ("b" + budget.ToString("0.####", CultureInfo.InvariantCulture)).Replace(".", "p");
                long[] k = layout[$"{key}_k4k2"].AsReadOnlySpan<long>().ToArray();
                BudgetLayout shape = BudgetLayout.FromIndices(Ints(layout[$"{key}_idx4"]), Ints(layout[$"{key}_idx2"]), Ints(layout[$"{key}_idx1"]), (int)k[0], (int)k[1]);
                foreach ((string phase, float t) in new[] { ("slow", 800f), ("fast", 200f) })
                {
                    using IDisposable scope = turbo.Phase(t);
                    (Tensor velocity, Tensor camera) = model.Forward(backend, input["latent"], input["camera"], t, contexts.At(t), shape);
                    using (velocity)
                    using (camera)
                    {
                        backend.Sync();
                        Report($"turbo {phase} budget {budget} ({shape.Tokens} tokens) v_latent", velocity, want[$"{key}_{phase}_v_latent"], 2e-2, 3e-3, failures);
                        Report($"turbo {phase} budget {budget} ({shape.Tokens} tokens) v_camera", camera, want[$"{key}_{phase}_v_camera"], 2e-2, 3e-3, failures);
                    }
                }
            }
        }
        finally
        {
            backend.Sync();
            backend.FreeWeights(trunk.EnumerateWeights());
            backend.FreeWeights(model.EnumerateWeights());
        }

        Assert.True(failures.Count == 0, string.Join("; ", failures));
    }

    /// <summary>The whole sampler against the reference's (<c>dump_idol_sampler.py</c>): Budget3D-Idol for 6 steps with guidance 3 and Budget3D-Idol-Turbo for 5 guidance-free steps, both at budget 0.5 from the reference's own noise,
    /// on CUDA with float32 blocks. The final latent and camera are compared by relative RMS, and the final layout's indices exactly where the run keeps the reference's layout.</summary>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    [Trait("Category", "GpuIntegration")]
    public void TheSamplerAtHalfBudget_MatchesTheReferenceOnCuda(bool turboModel)
    {
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }

        using CudaBackend backend = new(0, PtxDir());
        RunSampler(backend, cuda: true, turboModel, rmsLimit: turboModel ? 5e-2 : 2e-2);
    }

    /// <summary>The same sampler comparison in float32 on the CPU (hours: Budget3D-Idol runs twelve forwards, the turbo five and two image streams). Opt in with <c>HARTSY_RUN_SLOW_CPU=1</c>; progress is written to the test output after every step.</summary>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    [Trait("Category", "Slow")]
    public void TheSamplerAtHalfBudget_MatchesTheReferenceOnTheCpu(bool turboModel)
    {
        if (Environment.GetEnvironmentVariable("HARTSY_RUN_SLOW_CPU") != "1")
        {
            _output.WriteLine("SKIPPED: set HARTSY_RUN_SLOW_CPU=1 to run the CPU sampler comparison");
            return;
        }

        RunSampler(new CpuBackend(), cuda: false, turboModel, rmsLimit: 1e-2);
    }

    private void RunSampler(IBackend backend, bool cuda, bool turboModel, double rmsLimit)
    {
        string? models = Environment.GetEnvironmentVariable(ModelsEnv);
        string? refDir = Environment.GetEnvironmentVariable(ReferenceEnv);
        string checkpoint = Path.Combine(models ?? "", turboModel ? "budget3d-idol-turbo" : "budget3d-idol", turboModel ? "budget3d_idol_turbo.safetensors" : "budget3d_idol.safetensors");
        string conditioning = Path.Combine(refDir ?? "", "cond", "conditioning.safetensors");
        string inputs = Path.Combine(refDir ?? "", "den", "denoiser_inputs.safetensors");
        string reference = Path.Combine(refDir ?? "", "samp", turboModel ? "sampler-budget3d-idol-turbo.safetensors" : "sampler-budget3d-idol.safetensors");
        string[] missing = new[] { checkpoint, conditioning, inputs, reference }.Where(p => !File.Exists(p)).ToArray();
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
        using IdolTurboAdapters? turbo = turboModel ? IdolTurboAdapters.AttachEmbedded(weights) : null;
        using IdolDenoiser trunk = new(turbo?.Weights ?? weights);
        using BudgetDenoiser model = new(trunk, adapters);
        using SafeTensorsLoader condLoader = new(), inputLoader = new(), refLoader = new();
        condLoader.Load(conditioning);
        inputLoader.Load(inputs);
        refLoader.Load(reference);
        Dictionary<string, Tensor> cond = condLoader.GetAllTensors(), input = inputLoader.GetAllTensors(), want = refLoader.GetAllTensors();
        if (cuda)
        {
            backend.PreloadWeights(trunk.EnumerateWeights());
            backend.PreloadWeights(model.EnumerateWeights());
        }

        List<string> failures = [];
        try
        {
            using Tensor? positive = turbo is null ? trunk.EmbedContext(backend, cond["feature1"], cond["feature2"]) : null;
            using PhaseContexts? phases = turbo?.EmbedContexts(backend, trunk, cond["feature1"], cond["feature2"]);
            using Tensor zero1 = new(cond["feature1"].Shape, DType.F32), zero2 = new(cond["feature2"].Shape, DType.F32);
            using Tensor? blank = turbo is null ? trunk.EmbedContext(backend, zero1, zero2) : null;
            (Tensor latent, Tensor camera, BudgetLayout layout) = BudgetSampler.Sample(backend, model, positive ?? phases!.Slow, blank, input["latent"], input["camera"], BudgetLayout.SnapTokens(0.5 * 8192),
                steps: turboModel ? 5 : 6, guidance: turboModel ? 0f : 3f, rescaleT: 3.0, relayoutEvery: turboModel ? 1 : BudgetSampler.BaseRelayoutEvery, relayoutUntil: turboModel ? -1 : BudgetSampler.BaseRelayoutUntil,
                onStep: step => Progress($"{(turboModel ? "turbo" : "base")} sampler step {step} done {DateTime.Now:HH:mm:ss}"), passScope: turbo is null ? null : turbo.Phase, contextAt: phases is null ? null : phases.At);
            using (latent)
            using (camera)
            {
                backend.Sync();
                long[] k = want["k4k2"].AsReadOnlySpan<long>().ToArray();
                bool sameLayout = layout.K4 == k[0] && layout.K2 == k[1] && layout.Pairs.SequenceEqual(Ints(want["idx1"])) && layout.Cells.SequenceEqual(Ints(want["idx2"])) && layout.Cubes.SequenceEqual(Ints(want["idx4"]));
                _output.WriteLine($"final layout: {layout.Tokens} tokens, k4 {layout.K4}, k2 {layout.K2}; the reference's k4 {k[0]}, k2 {k[1]}; {(sameLayout ? "identical indices" : "the indices differ")}");
                Report($"{(turboModel ? "turbo 5-step" : "6-step guided")} latent", latent, want["latent"], double.PositiveInfinity, rmsLimit, failures);
                Report($"{(turboModel ? "turbo 5-step" : "6-step guided")} camera", camera, want["camera"], double.PositiveInfinity, rmsLimit, failures);
                // The base model's layout is refreshed twice and frozen, so it must be the reference's. The turbo rebuilds it from the running estimate at every step, which can swap cells whose scores
                // tie within float32 rounding of each other (the counts k4 and k2 are always the budget's); there only the counts are required.
                if (layout.K4 != k[0] || layout.K2 != k[1] || (!sameLayout && !turboModel))
                    failures.Add("the final layout differs from the reference's");
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

    // A long CPU run reports to the test output (shown when it ends) and, when HARTSY_PROGRESS_FILE names a file, to that file as it goes.
    private void Progress(string line)
    {
        _output.WriteLine(line);
        if (Environment.GetEnvironmentVariable("HARTSY_PROGRESS_FILE") is { Length: > 0 } path)
            File.AppendAllText(path, line + Environment.NewLine);
    }

    private void Run(IBackend backend, bool cuda, double[] budgets, double peakLimit, double rmsLimit, DType? blockDType = null)
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

        using IdolWeights weights = IdolWeights.Load(checkpoint, linearDType: blockDType);
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
