using System.Text.Json;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Pipelines.LongCat;
using HartsyInference.Tests.Shared;
using Xunit;

namespace HartsyInference.Diffusion.Tests.Parity;

/// <summary>The LongCat-Video 720p refinement (image-conditioned and unconditioned) against the pinned reference's own
/// <c>generate_refine</c> loop and DiT (tests/python-reference/longcat_pipeline/dump_longcat_refine.py, @ 58eb336), on the tiny B1
/// checkpoint, on the CPU in F32 and on CUDA with exact F32 GEMMs. A dump's <c>attention</c> field says what the reference
/// attended with: its Triton kernel, or the stand-in of its own block selection with dense attention over the selected blocks.
/// Skipped, and so not evidence, until HARTSY_LONGCAT_REFINE_DIR names the dump directory.</summary>
[Trait("Category", "Integration")]
public sealed class LongCatRefinementParityTests
{
    private const string EnvName = "HARTSY_LONGCAT_REFINE_DIR";
    private const double Tolerance = 1e-5;

    [LongCatRefinementFact]
    public void EveryScenarioMatchesTheReference()
    {
        using CpuBackend backend = new();
        Compare(backend, "refinement-parity-report.txt");
    }

    [LongCatRefinementFact(cuda: true)]
    [Trait("Category", "GpuIntegration")]
    public void EveryScenarioMatchesTheReferenceOnCuda()
    {
        using CudaBackend backend = GpuDevices.OpenCuda();
        using ExactGemmScope exact = new(backend);
        Compare(backend, "refinement-parity-report-cuda.txt");
    }

    private static void Compare(IBackend backend, string report)
    {
        string dir = Environment.GetEnvironmentVariable(EnvName)!;
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(Path.Combine(dir, "meta.json")));
        JsonElement root = document.RootElement;
        int[] block = root.GetProperty("block").EnumerateArray().Select(static e => e.GetInt32()).ToArray();
        TokenGrid3D grid = new(block[0], block[1], block[2]);
        using IDisposable f32 = LongCatTestData.PinF32Dit();
        using LongCatCheckpoint checkpoint = LongCatCheckpoint.Open(Path.Combine(dir, "ckpt"));
        LongCatConfig config = checkpoint.Config with
        {
            SparseAttention = new LongCatSparseAttention(root.GetProperty("sparsity").GetDouble(), grid, grid),
        };
        using LongCatTransformer transformer = new(config);
        transformer.LoadWeights(checkpoint.Weights);
        LongCatVideoPipeline pipeline = new(backend, transformer);
        LongCatPrompt prompt = new(Read(dir, document, "text.pos_embeds"), root.GetProperty("text_len").GetInt32(),
            root.GetProperty("valid_pos").GetInt32());

        List<string> failures = [], passed = [];
        foreach (string name in (string[])["refine_i2v", "refine_t2v"])
        {
            JsonElement scenario = root.GetProperty("scenarios").GetProperty(name);
            int cond = scenario.GetProperty("num_cond").GetInt32();
            LongCatRefineRun run = new(config.InChannels, scenario.GetProperty("frames").GetInt32(),
                scenario.GetProperty("latent_height").GetInt32(), scenario.GetProperty("latent_width").GetInt32(),
                Read(dir, document, $"{name}.latent"), Read(dir, document, $"{name}.noise"), prompt, cond,
                cond > 0 ? Read(dir, document, $"{name}.cond") : null, root.GetProperty("steps").GetInt32(),
                root.GetProperty("shift").GetSingle(), root.GetProperty("threshold").GetSingle(), RecordTrace: true);

            LongCatResult result = pipeline.Refine(run);

            Compare(failures, passed, $"{name} sigmas", result.Schedule.Sigmas, Numbers(scenario.GetProperty("sigmas")));
            Compare(failures, passed, $"{name} timesteps", result.Schedule.Timesteps, Numbers(scenario.GetProperty("steps")));
            for (int i = 0; i < result.Steps.Count; i++)
            {
                Compare(failures, passed, $"{name} step{i} velocity", result.Steps[i].Velocity, Read(dir, document, $"{name}.step{i}.noise"));
                Compare(failures, passed, $"{name} step{i} sample", result.Steps[i].SampleOut,
                    Read(dir, document, $"{name}.step{i}.sample_out"));
            }
            Compare(failures, passed, $"{name} final", result.Latents, Read(dir, document, $"{name}.final"));
        }

        File.WriteAllLines(Path.Combine(dir, report),
            failures.Select(static l => "FAIL " + l).Concat(passed.Select(static l => "ok   " + l)));
        Assert.True(failures.Count == 0, $"{failures.Count} comparison(s) off, {passed.Count} matched:\n{string.Join('\n', failures)}");
    }

    private static float[] Read(string dir, JsonDocument document, string name) => LongCatReferenceDump.Read(dir, document, name);

    private static float[] Numbers(JsonElement array) => array.EnumerateArray().Select(static e => e.GetSingle()).ToArray();

    private static void Compare(List<string> failures, List<string> passed, string label, ReadOnlySpan<float> actual, float[] expected)
    {
        if (actual.Length != expected.Length)
        {
            failures.Add($"{label}: {actual.Length} values against {expected.Length}");
            return;
        }
        double error = LongCatTestData.RelRms(actual, expected);
        (error <= Tolerance ? passed : failures).Add($"{label}: relRMS {error:E2} (limit {Tolerance:E0})");
    }

    [AttributeUsage(AttributeTargets.Method)]
    private sealed class LongCatRefinementFactAttribute : FactAttribute
    {
        public LongCatRefinementFactAttribute(bool cuda = false)
        {
            string? dir = Environment.GetEnvironmentVariable(EnvName);
            if (string.IsNullOrWhiteSpace(dir) || !File.Exists(Path.Combine(dir, "meta.json")))
                Skip = $"SKIPPED: {EnvName} is unset or has no meta.json; the LongCat refinement reference comparison did not run.";
            else if (cuda && GpuDevices.CudaFailure is { } failure)
                Skip = $"SKIPPED: no usable CUDA device ({failure}).";
        }
    }
}
