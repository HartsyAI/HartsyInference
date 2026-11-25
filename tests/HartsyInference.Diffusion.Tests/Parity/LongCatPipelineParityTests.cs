using System.Text.Json;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Pipelines.LongCat;
using Xunit;

namespace HartsyInference.Diffusion.Tests.Parity;

/// <summary>The LongCat-Video base pipelines (text-to-video, distilled text-to-video, image-to-video, continuation with the enhance_hf schedule,
/// and a two-segment continuation chain) against the pinned reference's own pipeline loops
/// (tests/python-reference/longcat_pipeline/dump_longcat_pipeline.py, @ 58eb336), on the tiny B1 checkpoint, CPU F32.
/// Skipped, and so not evidence, until HARTSY_LONGCAT_PIPELINE_DIR names the dump directory.</summary>
/// <remarks>The reference's continuation default uses its KV cache; the no-cache variant of the same call is not compared here (its CFG
/// scale covers the discarded conditioning frames, a different sampler, which this pipeline does not implement).</remarks>
[Trait("Category", "Integration")]
public sealed class LongCatPipelineParityTests
{
    private const string EnvName = "HARTSY_LONGCAT_PIPELINE_DIR";
    private const double Tolerance = 1e-5;

    [LongCatPipelineFact]
    public void EveryScenarioMatchesTheReference()
    {
        string dir = Environment.GetEnvironmentVariable(EnvName)!;
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(Path.Combine(dir, "meta.json")));
        JsonElement root = document.RootElement;
        using LongCatCheckpoint checkpoint = LongCatCheckpoint.Open(Path.Combine(dir, "ckpt"));
        using LongCatTransformer transformer = new(checkpoint.Config);
        transformer.LoadWeights(checkpoint.Weights);
        using CpuBackend backend = new();
        LongCatVideoPipeline pipeline = new(backend, transformer);

        int channels = checkpoint.Config.InChannels, captionChannels = checkpoint.Config.CaptionChannels;
        int textLen = root.GetProperty("text_len").GetInt32();
        LongCatPrompt positive = new(Read(dir, document, "text.pos_embeds"), textLen, root.GetProperty("valid_pos").GetInt32());
        LongCatPrompt negative = new(Read(dir, document, "text.neg_embeds"), textLen, root.GetProperty("valid_neg").GetInt32());
        int steps = root.GetProperty("steps").GetInt32();
        float shift = root.GetProperty("shift").GetSingle(), guidance = root.GetProperty("guidance").GetSingle();

        List<string> failures = [], passed = [];
        (int frames, int height, int width) Shape(string name)
        {
            JsonElement shape = root.GetProperty("tensors").GetProperty($"{name}.init");
            return ((int)shape[2].GetInt64(), (int)shape[3].GetInt64(), (int)shape[4].GetInt64());
        }

        LongCatRun Run(string name, LongCatConditioningMode mode, bool distill, bool enhance, int numCond, float[]? cond)
        {
            (int frames, int height, int width) = Shape(name);
            return new LongCatRun(channels, frames, height, width, Read(dir, document, $"{name}.init"), positive, negative, steps,
                distill, shift, guidance, mode, numCond, cond, enhance);
        }

        void Compare(string name, LongCatRun run)
        {
            LongCatResult result = pipeline.Denoise(run);
            JsonElement scenario = root.GetProperty("scenarios").GetProperty(name);
            CompareVector(failures, passed, $"{name} sigmas", result.Schedule.Sigmas, Doubles(scenario.GetProperty("sigmas")));
            CompareVector(failures, passed, $"{name} timesteps", result.Schedule.Timesteps, Doubles(scenario.GetProperty("steps")));
            for (int i = 0; i < result.Steps.Count; i++)
            {
                CompareVector(failures, passed, $"{name} step{i} velocity", result.Steps[i].Velocity, Read(dir, document, $"{name}.step{i}.noise"));
                CompareVector(failures, passed, $"{name} step{i} sample", result.Steps[i].SampleOut, Read(dir, document, $"{name}.step{i}.sample_out"));
            }
            CompareVector(failures, passed, $"{name} final", result.Latents, Read(dir, document, $"{name}.final"));
        }

        Compare("t2v", Run("t2v", LongCatConditioningMode.Text, distill: false, enhance: false, numCond: 0, cond: null));
        Compare("t2v_distill", Run("t2v_distill", LongCatConditioningMode.Text, distill: true, enhance: false, numCond: 0, cond: null));
        Compare("i2v", Run("i2v", LongCatConditioningMode.Image, distill: false, enhance: false, numCond: 1, cond: Read(dir, document, "i2v.cond")));
        Compare("vc", Run("vc", LongCatConditioningMode.Continuation, distill: false, enhance: true, numCond: 2,
            cond: Read(dir, document, "vc.cond")));

        (int chainFrames, int chainHeight, int chainWidth) = Shape("chain0");
        LongCatRun template = new(channels, chainFrames, chainHeight, chainWidth, new float[(long)channels * chainFrames * chainHeight * chainWidth],
            positive, negative, steps, Distill: false, shift, guidance);
        IReadOnlyList<float[]> chain = pipeline.ContinuationChain(template, Read(dir, document, "chain0.cond"),
            [Read(dir, document, "chain0.init"), Read(dir, document, "chain1.init")], contextFrames: 2);
        CompareVector(failures, passed, "chain0 final", chain[0], Read(dir, document, "chain0.final"));
        CompareVector(failures, passed, "chain1 final", chain[1], Read(dir, document, "chain1.final"));

        File.WriteAllLines(Path.Combine(dir, "pipeline-parity-report.txt"),
            failures.Select(static l => "FAIL " + l).Concat(passed.Select(static l => "ok   " + l)));
        Assert.True(failures.Count == 0, $"{failures.Count} comparison(s) off, {passed.Count} matched:\n{string.Join('\n', failures)}");
    }

    private static float[] Read(string dir, JsonDocument document, string name) => LongCatReferenceDump.Read(dir, document, name);

    private static float[] Doubles(JsonElement array)
    {
        float[] values = new float[array.GetArrayLength()];
        int i = 0;
        foreach (JsonElement element in array.EnumerateArray())
            values[i++] = element.GetSingle();
        return values;
    }

    private static void CompareVector(List<string> failures, List<string> passed, string label, ReadOnlySpan<float> actual, float[] expected)
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
    private sealed class LongCatPipelineFactAttribute : FactAttribute
    {
        public LongCatPipelineFactAttribute()
        {
            string? dir = Environment.GetEnvironmentVariable(EnvName);
            if (string.IsNullOrWhiteSpace(dir) || !File.Exists(Path.Combine(dir, "meta.json")))
                Skip = $"SKIPPED: {EnvName} is unset or has no meta.json; the LongCat pipeline reference comparison did not run.";
        }
    }
}
