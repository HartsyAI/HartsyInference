using System.Diagnostics;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Sampling;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Tier-1 direct-engine decode throughput benchmark (the Phase 1 harness planned but never built in
/// <c>docs/Checklists/LLM_THROUGHPUT_BENCHMARK.md</c>). Loads a real GGUF, runs warmup + N reps of prefill +
/// 128-token greedy decode through the actual production path (<see cref="TextGenerationPipeline.Generate"/>,
/// not a hand-rolled loop), and reports median decode tok/s — with and without CUDA-graph decode
/// (<c>HARTSY_GRAPH_DECODE=1</c>, greedy-only) since that is the fair "ours, best config" comparison point.
/// Gated on <c>HARTSY_TEST_GGUF_MODELS</c> (semicolon/comma-separated paths) + CUDA availability; skips cleanly
/// otherwise, same pattern as <see cref="SchedulerGraphDecodeTests"/>.</summary>
[Trait("Category", "Slow")]
[Collection("CudaSerial")]
public sealed class TextDecodeThroughputBenchmark
{
    private readonly ITestOutputHelper _output;
    public TextDecodeThroughputBenchmark(ITestOutputHelper output) => _output = output;

    private const int MaxTokens = 128;
    private const int Reps = 5;
    private const string Prompt = "Write a detailed, multi-paragraph technical explanation of how a modern " +
        "CPU executes a single machine instruction, covering the fetch, decode, execute, memory-access, and " +
        "write-back stages, including pipelining, hazards, branch prediction, and out-of-order execution.";

    private static string[] ModelPaths()
    {
        string? env = Environment.GetEnvironmentVariable("HARTSY_TEST_GGUF_MODELS");
        if (string.IsNullOrWhiteSpace(env)) return [];
        return [.. env
            .Split([';', ','], StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Where(File.Exists)];
    }

    private readonly record struct RunResult(double TgTps, double TtftMs, int TokenCount);

    private static RunResult RunOnce(TextGenerationPipeline pipeline, bool graphDecode)
    {
        List<long> stamps = [];
        Stopwatch sw = Stopwatch.StartNew();
        GenerationResult result = pipeline.Generate(new GenerationRequest
        {
            Prompt = Prompt,
            MaxTokens = MaxTokens,
            Sampling = SamplingOptions.Default with { Greedy = true },
            GraphDecode = graphDecode,
        }, onToken: _ => stamps.Add(sw.ElapsedTicks));
        if (stamps.Count < 2)
            return new RunResult(0, 0, stamps.Count);
        double ticksPerSec = Stopwatch.Frequency;
        double ttftMs = stamps[0] / ticksPerSec * 1000.0;
        double decodeWindowSec = (stamps[^1] - stamps[0]) / ticksPerSec;
        double tg = decodeWindowSec > 0 ? (stamps.Count - 1) / decodeWindowSec : 0;
        return new RunResult(tg, ttftMs, result.TokenIds.Count);
    }

    private static double Median(IEnumerable<double> xs)
    {
        double[] sorted = [.. xs];
        Array.Sort(sorted);
        int n = sorted.Length;
        return n % 2 == 1 ? sorted[n / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0;
    }

    [Fact]
    public void DecodeThroughput_GreedyGraphOffAndOn_ReportsMedianTgTps()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        string[] models = ModelPaths();
        if (models.Length == 0) { _output.WriteLine("SKIPPED: HARTSY_TEST_GGUF_MODELS not set"); return; }

        foreach (string path in models)
        {
            using CudaBackend backend = new(0, Path.Combine(AppContext.BaseDirectory, "Ptx"));
            using GgufLanguageModel model = GgufLanguageModel.Load(path, lowVramQuant: true);
            TextGenerationPipeline pipeline = new(model.Transformer, model.Tokenizer, backend, model.Template);
            bool graphEligible = model.Transformer.SupportsGraphDecode(backend);

            // Warmup (JIT, weight preload, first-launch overhead) — discarded.
            RunOnce(pipeline, graphDecode: false);

            List<double> tgOff = [];
            for (int i = 0; i < Reps; i++) tgOff.Add(RunOnce(pipeline, graphDecode: false).TgTps);

            List<double> tgOn = [];
            if (graphEligible)
            {
                RunOnce(pipeline, graphDecode: true);   // warmup + capture
                for (int i = 0; i < Reps; i++) tgOn.Add(RunOnce(pipeline, graphDecode: true).TgTps);
            }

            string name = $"{Path.GetFileName(path)} (arch={model.Architecture})";
            _output.WriteLine($"{name}: graph-off tg median = {Median(tgOff):F2} tok/s (reps: {string.Join(", ", tgOff.Select(x => x.ToString("F1")))})");
            if (graphEligible)
                _output.WriteLine($"{name}: graph-on  tg median = {Median(tgOn):F2} tok/s (reps: {string.Join(", ", tgOn.Select(x => x.ToString("F1")))})");
            else
                _output.WriteLine($"{name}: graph decode NOT eligible for this architecture (structural check failed)");

            Assert.True(Median(tgOff) > 0, $"{name}: graph-off produced no measurable decode throughput");
        }
    }
}
