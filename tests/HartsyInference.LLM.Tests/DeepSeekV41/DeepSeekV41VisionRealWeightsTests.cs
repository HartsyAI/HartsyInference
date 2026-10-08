using System.Diagnostics;
using System.Text.Json;
using HartsyInference.Cpu;
using HartsyInference.Engine;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Services;
using HartsyInference.LLM.DeepSeekV41;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.LLM.Tests.DeepSeekV41;

/// <summary>The real DeepSeek-V4.1-Flash vision tower and aligner (pinned revision dba1be0a) through the CPU host reference, against the UNMODIFIED upstream <c>vision.py</c> run on the same weights by
/// <c>tests/python-reference/deepseek_v41/dump_real_vision.py</c> (set <c>DSV41_VISION_ORACLE</c> to its output directory). A missing checkpoint or dump skips unless <c>HARTSY_REQUIRE_REAL_WEIGHTS=1</c>, which fails instead.
/// Run alone, under a memory cap: <c>systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 env HARTSY_REQUIRE_REAL_WEIGHTS=1 HARTSY_CPU_THREADS=8 DSV41_VISION_ORACLE=&lt;dump dir&gt; dotnet test tests/HartsyInference.LLM.Tests -c Release -f net10.0 --filter "FullyQualifiedName~DeepSeekV41VisionRealWeightsTests"</c>.</summary>
[Trait("Category", "Integration")]
[Trait("Category", "RealWeights")]
public sealed class DeepSeekV41VisionRealWeightsTests
{
    // Gates fixed before the first comparison ran. Both sides are float32 on BF16 weights widened exactly, so a miss is a structural
    // difference, not rounding. The project plan's bar is correlation >= 0.9999 at the aligner output; every stage is additionally held
    // to a relative L2 and, element by element, to a fraction of the stage's largest value, so a localized error that a norm would hide
    // still fails.
    private const double OutputCosineFloor = 0.9999;
    private const double StageRelL2 = 1e-3;
    private const double StageMaxAbsFraction = 1e-3;

    private readonly ITestOutputHelper _output;

    public DeepSeekV41VisionRealWeightsTests(ITestOutputHelper output) => _output = output;

    private static string ModelDirectory() =>
        ModelResolver.Resolve(DeepSeekV41Catalog.Id, null, Modality.Text).LocalPath
        ?? Path.Combine(RepoPaths.ModelsRoot(), "llm", "deepseek-v4.1-flash");

    private static string Rss()
    {
        using Process process = Process.GetCurrentProcess();
        return $"{process.WorkingSet64 / (1L << 30)} GiB (peak {process.PeakWorkingSet64 / (1L << 30)} GiB)";
    }

    [Fact]
    public void RealTowerAndAligner_MatchTheUpstreamModules()
    {
        string dir = ModelDirectory();
        string? oracle = Environment.GetEnvironmentVariable("DSV41_VISION_ORACLE");
        string metaPath = oracle is null ? "DSV41_VISION_ORACLE-unset" : Path.Combine(oracle, "meta.json");
        if (!RealWeightGate.Require(_output.WriteLine, Path.Combine(dir, "config.json"), metaPath)) return;

        using JsonDocument meta = JsonDocument.Parse(File.ReadAllText(Path.Combine(oracle!, "meta.json")));
        Assert.Equal("float32", meta.RootElement.GetProperty("dtype").GetString());
        int layers = meta.RootElement.GetProperty("layers").GetInt32();

        using CpuBackend backend = new();
        using DeepSeekV41Checkpoint checkpoint = DeepSeekV41Checkpoint.Open(dir);
        Assert.Equal(new DeepSeekV41VisionConfig(layers, 1024, 16, 2816, 14, 3, 10000.0), checkpoint.Config.Vision);
        Stopwatch sw = Stopwatch.StartNew();
        using DeepSeekV41VisionModel model = DeepSeekV41VisionLoader.Load(backend, checkpoint);
        _output.WriteLine($"load (266 tensors, BF16 to F32) {sw.Elapsed.TotalSeconds:F1}s, RSS {Rss()}");
        Assert.Equal(checkpoint.Config.HiddenSize, model.OutputDim);

        (string Name, float[] Values)[] embeddings =
            [("image_start", model.ImageStart.ToArray()), ("image_end", model.ImageEnd.ToArray()), ("image_newline", model.ImageNewline.ToArray())];
        foreach ((string name, float[] actual) in embeddings)
        {
            float[] expected = ReadF32(Path.Combine(oracle!, name + ".f32"), model.OutputDim);
            double diff = DeepSeekV41VisionMetrics.MaxAbsDiff(actual, expected);
            _output.WriteLine($"{name}: maxDiff {diff:E2}");
            Assert.Equal(0.0, diff);
        }

        Dictionary<string, float[]> taps = [];
        model.Tower.Probe = (stage, values) => taps[stage] = values;
        model.Aligner.Probe = (stage, values) => taps[stage] = values;
        foreach (JsonElement grid in meta.RootElement.GetProperty("grids").EnumerateArray())
        {
            string tag = grid.GetProperty("tag").GetString()!;
            int height = grid.GetProperty("height").GetInt32(), width = grid.GetProperty("width").GetInt32();
            (int tokenHeight, int tokenWidth) = model.TokenGrid(height, width);
            Assert.Equal((grid.GetProperty("tokenHeight").GetInt32(), grid.GetProperty("tokenWidth").GetInt32()), (tokenHeight, tokenWidth));
            float[] patches = ReadF32(Path.Combine(oracle!, $"{tag}.patches.f32"), height * width * model.Tower.Config.PatchInputDim);
            taps.Clear();

            sw.Restart();
            float[] output = model.Encode(patches, height, width);
            _output.WriteLine($"{tag}: encode {sw.Elapsed.TotalSeconds:F1}s, RSS {Rss()}");

            Assert.Equal(tokenHeight * tokenWidth * model.OutputDim, output.Length);
            taps["out"] = output;
            string[] stages = ["patch_embed", .. Enumerable.Range(0, layers).Select(static i => $"block.{i}"), "norm", "unfold", "hidden", "out"];
            Assert.Equal(stages.Length, taps.Count);
            foreach (string stage in stages)
            {
                float[] actual = taps[stage];
                float[] expected = ReadF32(Path.Combine(oracle!, $"{tag}.{stage}.f32"), actual.Length);
                double rel = DeepSeekV41VisionMetrics.RelL2(actual, expected), cos = DeepSeekV41VisionMetrics.Cosine(actual, expected);
                double abs = DeepSeekV41VisionMetrics.MaxAbsDiff(actual, expected), scale = DeepSeekV41VisionMetrics.MaxAbs(expected);
                _output.WriteLine($"{tag} {stage,-11} relL2 {rel:E2} cos {cos:F8} maxAbs {abs:E2} (stage max {scale:F3})");
                Assert.All(actual, static v => Assert.True(float.IsFinite(v)));
                Assert.True(rel <= StageRelL2, $"{tag} {stage}: relL2 {rel:E3} > {StageRelL2:E1}");
                Assert.True(abs <= StageMaxAbsFraction * scale, $"{tag} {stage}: maxAbs {abs:E3} > {StageMaxAbsFraction * scale:E3}");
                if (stage == "out") Assert.True(cos >= OutputCosineFloor, $"{tag} aligner output cosine {cos:F6} < {OutputCosineFloor}");
            }
        }
    }

    private static float[] ReadF32(string path, int expected)
    {
        byte[] bytes = File.ReadAllBytes(path);
        Assert.Equal(expected * 4, bytes.Length);
        float[] values = new float[expected];
        Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length);
        return values;
    }
}
