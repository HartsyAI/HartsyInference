using System.Diagnostics;
using System.Globalization;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ThreeD.Models.Idol;
using HartsyInference.ThreeD.Models.Idol.Budget;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.ThreeD.Tests.Parity;

/// <summary>Where a Budget3D forward spends its time on CUDA in the shipped precision (F16 blocks): the token embedding, the trunk, and the expansion, per budget. A measurement, not a gate; it prints the milliseconds of the second and third calls.</summary>
[Trait("Category", "GpuIntegration")]
[Collection(Budget3DCudaCollection.Name)]
public sealed class BudgetProfileTests
{
    private readonly ITestOutputHelper _output;

    public BudgetProfileTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void PrintTheStageTimesOfABudgetForward()
    {
        string? models = Environment.GetEnvironmentVariable("HARTSY_IDOL_MODELS");
        string? refDir = Environment.GetEnvironmentVariable("HARTSY_IDOL_REF");
        string checkpoint = Path.Combine(models ?? "", "budget3d-idol", "budget3d_idol.safetensors");
        string conditioning = Path.Combine(refDir ?? "", "cond", "conditioning.safetensors");
        string inputs = Path.Combine(refDir ?? "", "den", "denoiser_inputs.safetensors");
        string layouts = Path.Combine(refDir ?? "", "b3d", "layouts.safetensors");
        if (!new[] { checkpoint, conditioning, inputs, layouts }.All(File.Exists) || !CudaContext.IsAvailable())
        {
            _output.WriteLine($"SKIPPED: needs the Budget3D checkpoint, the reference dumps and a CUDA card (missing: {string.Join(", ", new[] { checkpoint, conditioning, inputs, layouts }.Where(f => !File.Exists(f)).DefaultIfEmpty("CUDA"))})");
            return;
        }

        using CudaBackend backend = new(0, PtxDir());
        using IdolWeights weights = IdolWeights.Load(checkpoint, linearDType: DType.F16);
        using BudgetAdapterWeights adapters = BudgetAdapterWeights.Load(checkpoint);
        using IdolDenoiser trunk = new(weights);
        using BudgetDenoiser model = new(trunk, adapters);
        using SafeTensorsLoader condLoader = new(), inputLoader = new(), layoutLoader = new();
        condLoader.Load(conditioning);
        inputLoader.Load(inputs);
        layoutLoader.Load(layouts);
        Dictionary<string, Tensor> cond = condLoader.GetAllTensors(), input = inputLoader.GetAllTensors(), layout = layoutLoader.GetAllTensors();
        backend.PreloadWeights(trunk.EnumerateWeights());
        backend.PreloadWeights(model.EnumerateWeights());
        BudgetTimer.Enabled = true;
        try
        {
            using Tensor context = trunk.EmbedContext(backend, cond["feature1"], cond["feature2"]);
            foreach (string key in new[] { "b1", "b0p5", "b0p25", "b0p0625" })
            {
                long[] k = layout[$"{key}_k4k2"].AsReadOnlySpan<long>().ToArray();
                BudgetLayout shape = BudgetLayout.FromIndices(Ints(layout[$"{key}_idx4"]), Ints(layout[$"{key}_idx2"]), Ints(layout[$"{key}_idx1"]), (int)k[0], (int)k[1]);
                for (int call = 0; call < 3; call++)
                {
                    BudgetTimer.Reset();
                    Stopwatch clock = Stopwatch.StartNew();
                    (Tensor v, Tensor c) = model.Forward(backend, input["latent"], input["camera"], 731f, context, shape);
                    backend.Sync();
                    using (v)
                    using (c)
                        _ = v.AsReadOnlySpan<float>()[0];
                    if (call > 0)
                        _output.WriteLine($"{key} ({shape.Tokens} tokens) call {call}: total {clock.Elapsed.TotalMilliseconds.ToString("F0", CultureInfo.InvariantCulture)} ms; " + string.Join(", ", BudgetTimer.Milliseconds.Select(p => $"{p.Key} {p.Value.ToString("F0", CultureInfo.InvariantCulture)}")));
                }
            }
        }
        finally
        {
            BudgetTimer.Enabled = false;
            backend.Sync();
            backend.FreeWeights(trunk.EnumerateWeights());
            backend.FreeWeights(model.EnumerateWeights());
        }
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

        throw new DirectoryNotFoundException("Ptx not found.");
    }
}
