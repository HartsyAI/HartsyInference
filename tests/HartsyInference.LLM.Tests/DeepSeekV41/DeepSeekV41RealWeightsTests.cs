using System.Diagnostics;
using HartsyInference.Cpu;
using HartsyInference.LLM.DeepSeekV41;
using HartsyInference.ModelAssets.BlockScale;
using HartsyInference.ModelAssets.Tokenizers;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.LLM.Tests.DeepSeekV41;

/// <summary>The official DeepSeek-V4.1-Flash checkpoint through the CPU host reference. Needs ~48 GiB of RAM and the weights under
/// <c>$HARTSYINFERENCE_MODELS_DIR/llm/deepseek-v4.1-flash</c>; a missing directory skips unless <c>HARTSY_REQUIRE_REAL_WEIGHTS=1</c>.
/// <c>HARTSY_REQUIRE_REAL_WEIGHTS=1 HARTSYINFERENCE_MODELS_DIR=/mnt/model-storage/Models dotnet test tests/HartsyInference.LLM.Tests -c Release -f net10.0 --filter "FullyQualifiedName~DeepSeekV41RealWeightsTests"</c>.</summary>
[Trait("Category", "Integration")]
[Trait("Category", "RealWeights")]
public sealed class DeepSeekV41RealWeightsTests
{
    private readonly ITestOutputHelper _output;

    public DeepSeekV41RealWeightsTests(ITestOutputHelper output) => _output = output;

    private static string ModelDirectory() =>
        Path.Combine(Environment.GetEnvironmentVariable("HARTSYINFERENCE_MODELS_DIR") ?? "", "llm", "deepseek-v4.1-flash");

    [Fact]
    public void Smoke_PrefillGivesFiniteLogitsAndGreedyTokens()
    {
        string dir = ModelDirectory();
        if (!RealWeightGate.Require(_output.WriteLine, Path.Combine(dir, "config.json"), Path.Combine(dir, "model.safetensors.index.json"))) return;

        using CpuBackend backend = new();
        Stopwatch sw = Stopwatch.StartNew();
        using DeepSeekV41LoadedModel loaded = DeepSeekV41HostModelLoader.Load(backend, dir, new DeepSeekV41LoadOptions(MaxTokens: 64));
        using FileStream tokStream = File.OpenRead(Path.Combine(dir, "tokenizer.json"));
        ILlmTokenizer tokenizer = HfTokenizerJson.LoadByteLevelBpe(tokStream, bosToken: DeepSeekV41TextModel.BosLiteral, eosToken: DeepSeekV41TextModel.EosLiteral);
        _output.WriteLine($"load {sw.Elapsed.TotalSeconds:F1}s, peak RSS {Process.GetCurrentProcess().PeakWorkingSet64 / (1L << 30)} GiB");

        DeepSeekV41HostModel model = loaded.Model;
        List<int> ids = [tokenizer.BosId!.Value, .. tokenizer.EncodeOrdinary("The capital of France is")];
        DeepSeekV41SequenceState state = model.CreateState(64);
        float[] hidden = new float[ids.Count * model.Dim];
        sw.Restart();
        model.Forward(ids.ToArray(), state, hidden);
        _output.WriteLine($"prefill {ids.Count} tokens {sw.Elapsed.TotalSeconds:F1}s");
        float[] logits = model.Logits(hidden.AsSpan((ids.Count - 1) * model.Dim, model.Dim));
        Assert.All(logits, v => Assert.True(float.IsFinite(v)));

        for (int step = 0; step < 8; step++)
        {
            int next = Array.IndexOf(logits, logits.Max());
            ids.Add(next);
            _output.WriteLine($"step {step}: {next} {tokenizer.Decode([next]).Replace("\n", "\\n")}");
            sw.Restart();
            float[] h = new float[model.Dim];
            model.Forward([next], state, h);
            logits = model.Logits(h);
            _output.WriteLine($"  {sw.Elapsed.TotalSeconds:F1}s/token, peak RSS {Process.GetCurrentProcess().PeakWorkingSet64 / (1L << 30)} GiB");
            Assert.All(logits, v => Assert.True(float.IsFinite(v)));
        }
    }

    /// <summary>Dequantized row slices dumped by <c>tests/python-reference/deepseek_v41/dump_real_slices.py</c> (set <c>DSV41_REAL_SLICES</c> to its output directory).</summary>
    [Fact]
    public void RealTensorSlices_DequantizeLikeTheReference()
    {
        string dir = ModelDirectory();
        string? slices = Environment.GetEnvironmentVariable("DSV41_REAL_SLICES");
        if (!RealWeightGate.Require(_output.WriteLine, Path.Combine(dir, "config.json"), slices ?? "DSV41_REAL_SLICES-unset")) return;

        using DeepSeekV41Checkpoint checkpoint = DeepSeekV41Checkpoint.Open(dir);
        foreach (System.Text.Json.JsonElement job in System.Text.Json.JsonDocument.Parse(File.ReadAllText(Path.Combine(slices!, "manifest.json"))).RootElement.EnumerateArray())
        {
            string key = job.GetProperty("key").GetString()!;
            int r0 = job.GetProperty("r0").GetInt32(), r1 = job.GetProperty("r1").GetInt32(), cols = job.GetProperty("cols").GetInt32();
            float[] expected = new float[(r1 - r0) * cols];
            Buffer.BlockCopy(File.ReadAllBytes(Path.Combine(slices!, job.GetProperty("file").GetString()!)), 0, expected, 0, expected.Length * 4);

            float[] all = WeightDequantizer.ToF32(checkpoint.GetWeight(key), checkpoint.GetQuant(key));
            Assert.Equal(0, all.Length % cols);
            int worst = 0; double maxDiff = 0;
            for (int i = 0; i < expected.Length; i++)
            {
                double diff = Math.Abs(all[(long)r0 * cols + i] - expected[i]);
                if (diff > maxDiff) { maxDiff = diff; worst = i; }
            }
            _output.WriteLine($"{key}[{r0}:{r1}] cols {cols} (logical {all.Length / cols} rows) maxDiff {maxDiff:E2} at {worst}");
            Assert.Equal(0.0, maxDiff);
        }
    }
}
