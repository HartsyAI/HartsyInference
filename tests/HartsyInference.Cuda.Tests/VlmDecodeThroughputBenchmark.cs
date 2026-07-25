using System.Diagnostics;
using System.Linq;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Multimodal;
using HartsyInference.LLM.Sampling;
using HartsyInference.ModelAssets.Gguf;
using HartsyInference.Vision.Codec;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>VLM perf pass companion to <see cref="TextDecodeThroughputBenchmark"/>: that harness is text-only
/// (bare <c>Prompt</c> string, no image), so it measures LLaVA's Vicuna-7B/Llama LANGUAGE backbone decode speed
/// but never exercises the vision tower + projector cost real VLM usage pays every request. This drives the
/// SAME production path Swarm/the CLI use (<c>TextService.RunVision</c>: <c>VlmImagePreprocessor.Preprocess</c>
/// → <c>MultimodalGenerator.Generate</c>) against a real photo, loaded once, N reps, reporting prefill
/// (image encode + projector + text prefill, all folded into one <c>ForwardEmbeds</c> call) and decode
/// wall-clock separately. <c>MultimodalGenerator</c>'s decode loop has no CUDA-graph capture and D2H-syncs
/// every step (unlike <see cref="TextGenerationPipeline"/>'s optimized path) — this number is expected to be
/// slower than the text-only figure for that structural reason, not a LLaVA-specific regression; the gap
/// itself is a perf-pass finding, not noise. Gated on <c>HARTSY_TEST_VLM_MODELS</c> (semicolon/comma-separated
/// "textGguf|mmprojGguf" pairs) + CUDA availability; skips cleanly otherwise.</summary>
[Trait("Category", "Slow")]
[Collection("CudaSerial")]
public sealed class VlmDecodeThroughputBenchmark
{
    private readonly ITestOutputHelper _output;
    public VlmDecodeThroughputBenchmark(ITestOutputHelper output) => _output = output;

    private const int MaxTokens = 64;
    private const int Reps = 5;
    private const string Question = "Describe this image in detail.";

    private static string TestImagePath => Path.Combine(
        Path.GetDirectoryName(typeof(VlmDecodeThroughputBenchmark).Assembly.Location)!, "TestData", "bus.png");

    private static (string text, string mmproj)[] ModelPairs()
    {
        string? env = Environment.GetEnvironmentVariable("HARTSY_TEST_VLM_MODELS");
        if (string.IsNullOrWhiteSpace(env)) return [];
        List<(string, string)> pairs = [];
        foreach (string entry in env.Split([';', ','], StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            string[] parts = entry.Split('|');
            if (parts.Length == 2 && File.Exists(parts[0]) && File.Exists(parts[1]))
                pairs.Add((parts[0], parts[1]));
        }
        return [.. pairs];
    }

    private readonly record struct RunResult(double PrefillMs, double DecodeMs, int TokenCount);

    [Fact]
    public void VlmThroughput_ImagePlusPrompt_ReportsPrefillAndDecode()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        (string text, string mmproj)[] pairs = ModelPairs();
        if (pairs.Length == 0) { _output.WriteLine("SKIPPED: HARTSY_TEST_VLM_MODELS not set (\"text.gguf|mmproj.gguf;...\")."); return; }
        if (!File.Exists(TestImagePath)) { _output.WriteLine($"SKIPPED: test image not found: {TestImagePath}"); return; }

        (byte[] rgb, int width, int height) = PngDecoder.DecodeFromFile(TestImagePath);

        foreach ((string textPath, string mmprojPath) in pairs)
        {
            using CudaBackend backend = new(0, Path.Combine(AppContext.BaseDirectory, "Ptx"));
            bool lowVram = new FileInfo(textPath).Length > 2L << 30;
            using GgufLanguageModel model = GgufLanguageModel.Load(textPath, lowVramQuant: lowVram);
            backend.PreloadWeights(model.Transformer.EnumerateWeights());
            using IVlmImageEncoder vision = IsLlavaNext(mmprojPath) ? LlavaNextEncoder.Load(mmprojPath) : SiglipVlmEncoder.Load(mmprojPath);
            using Tensor pixelValues = vision is LlavaNextEncoder
                ? LlavaNextImagePreprocessor.RawToNativeTensor(rgb, width, height)
                : VlmImagePreprocessor.Preprocess(rgb, width, height, vision.ImageSize, vision.ImageMean, vision.ImageStd);

            MultimodalGenerator gen = new(model, vision, backend);
            SamplingOptions greedy = SamplingOptions.Default with { Greedy = true };

            RunResult RunOnce()
            {
                // MultimodalGenerator.Generate has no token-level progress callback — time prefill (image
                // encode + projector + first-token) and total wall-clock separately by wrapping Generate at
                // MaxTokens=1 (prefill-only proxy) then again at the real MaxTokens, matching how the doc's
                // TTFT/decode split works elsewhere without needing to change production code for a benchmark.
                Stopwatch swPrefill = Stopwatch.StartNew();
                _ = gen.Generate(pixelValues, Question, maxTokens: 1, greedy);
                swPrefill.Stop();

                Stopwatch swFull = Stopwatch.StartNew();
                string full = gen.Generate(pixelValues, Question, maxTokens: MaxTokens, greedy);
                swFull.Stop();

                int tokenCount = model.Tokenizer.EncodeOrdinary(full).Length;
                double decodeMs = swFull.Elapsed.TotalMilliseconds - swPrefill.Elapsed.TotalMilliseconds;
                return new RunResult(swPrefill.Elapsed.TotalMilliseconds, decodeMs, tokenCount);
            }

            // Warmup (JIT, weight preload, image-encoder first-launch overhead) — discarded.
            RunOnce();

            List<double> prefillMs = [];
            List<double> decodeMs = [];
            List<double> tgTps = [];
            for (int i = 0; i < Reps; i++)
            {
                RunResult r = RunOnce();
                prefillMs.Add(r.PrefillMs);
                decodeMs.Add(r.DecodeMs);
                tgTps.Add(r.TokenCount > 1 && r.DecodeMs > 0 ? (r.TokenCount - 1) / (r.DecodeMs / 1000.0) : 0);
            }

            string name = $"{Path.GetFileName(textPath)} + {Path.GetFileName(mmprojPath)} (arch={model.Architecture}, family={vision.Family})";
            _output.WriteLine($"{name}: prefill (image encode + projector + text prefill) median = {Median(prefillMs):F1} ms " +
                $"(reps: {string.Join(", ", prefillMs.Select(x => x.ToString("F0")))})");
            _output.WriteLine($"{name}: decode median = {Median(tgTps):F2} tok/s over {MaxTokens} tokens " +
                $"(reps: {string.Join(", ", tgTps.Select(x => x.ToString("F1")))}) — no CUDA-graph decode, D2H-sync every step (structural, not model-specific)");

            Assert.True(Median(prefillMs) > 0, $"{name}: prefill produced no measurable time");
        }
    }

    private static double Median(IEnumerable<double> xs)
    {
        double[] sorted = [.. xs];
        Array.Sort(sorted);
        int n = sorted.Length;
        return n % 2 == 1 ? sorted[n / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0;
    }

    /// <summary>Mirrors <c>TextService.IsLlavaNext</c>: LLaVA-NeXT/1.6 anyres checkpoints are distinguished from
    /// plain LLaVA-1.5 (same CLIP tower + projector tensors) by the presence of the
    /// <c>clip.vision.image_grid_pinpoints</c> metadata key.</summary>
    private static bool IsLlavaNext(string mmprojPath)
    {
        using GgufLoader probe = new GgufLoader();
        probe.Load(mmprojPath);
        return probe.Metadata.ContainsKey("clip.vision.image_grid_pinpoints");
    }
}
