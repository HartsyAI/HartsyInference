using HartsyInference.Cuda;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Sampling;
using HartsyInference.LLM.Transformer;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Phase 4 gate for the graph-decode-into-scheduler retrofit (see
/// docs/Checklists/LLM_DECODE_PERF_GRIND.md's "NEW PLAN" section): the first point in this retrofit where a
/// REAL CUDA-graph capture actually happens through <see cref="DynamicBatchScheduler.SubmitAsync"/> (Phases
/// 0-3 could only prove the surrounding logic — heterogeneous cache arrays, admission gating, the circuit
/// breaker — since the CPU backend can't construct a real <see cref="Generation.GraphDecodeSession"/> at all).
/// Gated on <c>HARTSY_TEST_GGUF_MODELS</c> + CUDA availability, same pattern as
/// <see cref="GraphDecodeRepetitionPenaltyTests"/>; skips cleanly otherwise.</summary>
[Collection("CudaSerial")]
public sealed class SchedulerGraphDecodeTests
{
    private readonly ITestOutputHelper _output;
    public SchedulerGraphDecodeTests(ITestOutputHelper output) => _output = output;

    private const string PromptA = "Continue this repeating pattern with more numbers, comma separated, no other text: 1, 2, 3, 4, 1, 2, 3, 4,";
    private const string PromptB = "Say hello in exactly three words.";

    private static string[] ModelPaths()
    {
        string? env = Environment.GetEnvironmentVariable("HARTSY_TEST_GGUF_MODELS");
        if (string.IsNullOrWhiteSpace(env)) return [];
        return [.. env
            .Split([';', ','], StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Where(File.Exists)];
    }

    private static PagedKvPool NewPool(TransformerConfig cfg) =>
        new(cfg.NumLayers, cfg.NumKvHeads, cfg.HeadDim, pageSize: 16, maxPages: 512);

    /// <summary>Regression gate for a real bug found live-testing this retrofit: a genuinely cold model's
    /// FIRST-EVER solo admission (no prior eager decode of any kind) failed capture with
    /// CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED — see <see cref="DynamicBatchScheduler.CaptureGraphSession"/>'s
    /// warm-up doc for the root cause and fix. This is exactly the realistic "first request after a server
    /// just loaded a model" scenario, and neither of this file's other two tests actually exercised it (both
    /// incidentally warm up the shared backend via an earlier eager call before their own capture attempt).</summary>
    [Fact]
    public async Task SoloAdmission_SucceedsColdWithNoPriorEagerWarmup()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        string[] models = ModelPaths();
        if (models.Length == 0) { _output.WriteLine("SKIPPED: HARTSY_TEST_GGUF_MODELS not set"); return; }
        List<string> failures = [];
        foreach (string path in models)
        {
            using CudaBackend backend = new(0, Path.Combine(AppContext.BaseDirectory, "Ptx"));
            using GgufLanguageModel model = GgufLanguageModel.Load(path, dequantizeToF32: false);
            if (!model.Transformer.SupportsGraphDecode(backend)) { _output.WriteLine($"SKIPPED (not graph-decode-eligible): {Path.GetFileName(path)}"); continue; }
            SamplingOptions sampling = SamplingOptions.Default with { Greedy = true };
            using PagedKvPool pool = NewPool(model.Transformer.Config);
            using DynamicBatchScheduler scheduler = new(model.Transformer, model.Tokenizer, backend, pool, model.Template);
            // The VERY FIRST thing this fresh backend/model ever does is a solo graph-eligible admission.
            GenerationResult result = await scheduler.SubmitAsync(new GenerationRequest
            { Prompt = PromptA, MaxTokens = 40, Sampling = sampling, GraphDecode = true }, onToken: null, CancellationToken.None);
            _output.WriteLine($"{Path.GetFileName(path)}: cold capture => [{string.Join(",", result.TokenIds)}]");
            if (result.TokenIds.Count == 0)
                failures.Add($"{Path.GetFileName(path)}: cold solo admission produced no tokens");
        }
        Assert.True(failures.Count == 0, string.Join("\n", failures));
    }

    [Fact]
    public async Task SoloSchedulerRequest_MatchesTextGenerationPipeline_WithGraphDecodeOn()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        string[] models = ModelPaths();
        if (models.Length == 0) { _output.WriteLine("SKIPPED: HARTSY_TEST_GGUF_MODELS not set"); return; }

        List<string> failures = [];
        foreach (string path in models)
        {
            using CudaBackend backend = new(0, Path.Combine(AppContext.BaseDirectory, "Ptx"));
            using GgufLanguageModel model = GgufLanguageModel.Load(path, dequantizeToF32: false);
            if (!model.Transformer.SupportsGraphDecode(backend))
            {
                _output.WriteLine($"SKIPPED (not graph-decode-eligible): {Path.GetFileName(path)}");
                continue;
            }
            string name = $"{Path.GetFileName(path)} (arch={model.Architecture})";
            SamplingOptions sampling = SamplingOptions.Default with { Greedy = true };

            TextGenerationPipeline pipeline = new(model.Transformer, model.Tokenizer, backend, model.Template);
            GenerationResult reference = pipeline.Generate(new GenerationRequest
            { Prompt = PromptA, MaxTokens = 40, Sampling = sampling, GraphDecode = true });

            using PagedKvPool pool = NewPool(model.Transformer.Config);
            using DynamicBatchScheduler scheduler = new(model.Transformer, model.Tokenizer, backend, pool, model.Template);
            GenerationResult scheduled = await scheduler.SubmitAsync(new GenerationRequest
            { Prompt = PromptA, MaxTokens = 40, Sampling = sampling, GraphDecode = true }, onToken: null, CancellationToken.None);

            _output.WriteLine($"{name}: pipeline=[{string.Join(",", reference.TokenIds)}]");
            _output.WriteLine($"{name}: scheduler=[{string.Join(",", scheduled.TokenIds)}]");
            if (!reference.TokenIds.SequenceEqual(scheduled.TokenIds))
                failures.Add($"{name}: solo scheduler request diverged from TextGenerationPipeline.Generate with graph decode on");
        }
        Assert.True(failures.Count == 0, string.Join("\n", failures));
    }

    [Fact]
    public async Task SoloSchedulerRequest_SameOutput_GraphDecodeOnOrOff()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        string[] models = ModelPaths();
        if (models.Length == 0) { _output.WriteLine("SKIPPED: HARTSY_TEST_GGUF_MODELS not set"); return; }

        List<string> failures = [];
        foreach (string path in models)
        {
            using CudaBackend backend = new(0, Path.Combine(AppContext.BaseDirectory, "Ptx"));
            using GgufLanguageModel model = GgufLanguageModel.Load(path, dequantizeToF32: false);
            if (!model.Transformer.SupportsGraphDecode(backend))
            {
                _output.WriteLine($"SKIPPED (not graph-decode-eligible): {Path.GetFileName(path)}");
                continue;
            }
            string name = $"{Path.GetFileName(path)} (arch={model.Architecture})";
            SamplingOptions sampling = SamplingOptions.Default with { Greedy = true };

            using PagedKvPool poolOff = NewPool(model.Transformer.Config);
            using DynamicBatchScheduler schedulerOff = new(model.Transformer, model.Tokenizer, backend, poolOff, model.Template);
            GenerationResult withoutGraph = await schedulerOff.SubmitAsync(new GenerationRequest
            { Prompt = PromptA, MaxTokens = 40, Sampling = sampling, GraphDecode = false }, onToken: null, CancellationToken.None);

            using PagedKvPool poolOn = NewPool(model.Transformer.Config);
            using DynamicBatchScheduler schedulerOn = new(model.Transformer, model.Tokenizer, backend, poolOn, model.Template);
            GenerationResult withGraph = await schedulerOn.SubmitAsync(new GenerationRequest
            { Prompt = PromptA, MaxTokens = 40, Sampling = sampling, GraphDecode = true }, onToken: null, CancellationToken.None);

            _output.WriteLine($"{name}: withoutGraph=[{string.Join(",", withoutGraph.TokenIds)}]");
            _output.WriteLine($"{name}: withGraph=[{string.Join(",", withGraph.TokenIds)}]");
            if (!withoutGraph.TokenIds.SequenceEqual(withGraph.TokenIds))
                failures.Add($"{name}: enabling graph decode changed scheduler output — the optimization must never change output");
        }
        Assert.True(failures.Count == 0, string.Join("\n", failures));
    }

    [Fact]
    public async Task Transition_SoloThenCrowded_BothSequencesCorrect_ARetirementDoesNotCorruptOutput()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        string[] models = ModelPaths();
        if (models.Length == 0) { _output.WriteLine("SKIPPED: HARTSY_TEST_GGUF_MODELS not set"); return; }

        List<string> failures = [];
        foreach (string path in models)
        {
            using CudaBackend backend = new(0, Path.Combine(AppContext.BaseDirectory, "Ptx"));
            using GgufLanguageModel model = GgufLanguageModel.Load(path, dequantizeToF32: false);
            if (!model.Transformer.SupportsGraphDecode(backend))
            {
                _output.WriteLine($"SKIPPED (not graph-decode-eligible): {Path.GetFileName(path)}");
                continue;
            }
            string name = $"{Path.GetFileName(path)} (arch={model.Architecture})";
            SamplingOptions sampling = SamplingOptions.Default with { Greedy = true };

            // Reference for A: alone, graph decode OFF the whole time — the ground truth this test's A must
            // match regardless of the graph→eager splice it actually goes through below.
            TextGenerationPipeline refPipeline = new(model.Transformer, model.Tokenizer, backend, model.Template);
            GenerationResult referenceA = refPipeline.Generate(new GenerationRequest
            { Prompt = PromptA, MaxTokens = 60, Sampling = sampling, GraphDecode = false });
            // Reference for B: alone, plain (never graph-eligible in this test — admitted while A is active).
            GenerationResult referenceB = refPipeline.Generate(new GenerationRequest
            { Prompt = PromptB, MaxTokens = 40, Sampling = sampling, GraphDecode = false });

            using PagedKvPool pool = NewPool(model.Transformer.Config);
            using DynamicBatchScheduler scheduler = new(model.Transformer, model.Tokenizer, backend, pool, model.Template);

            // A admitted alone (captures a graph — scheduler is idle at this instant). B submitted immediately
            // after, before A can possibly finish (A's MaxTokens=60 guarantees many rounds of runway), forcing
            // A's session to retire and the round to go through the heterogeneous eager path (FixedKvCache for
            // A + PagedKvCache for B) for the rest of both generations.
            Task<GenerationResult> taskA = scheduler.SubmitAsync(new GenerationRequest
            { Prompt = PromptA, MaxTokens = 60, Sampling = sampling, GraphDecode = true }, onToken: null, CancellationToken.None);
            Task<GenerationResult> taskB = scheduler.SubmitAsync(new GenerationRequest
            { Prompt = PromptB, MaxTokens = 40, Sampling = sampling, GraphDecode = true }, onToken: null, CancellationToken.None);

            GenerationResult[] results = await Task.WhenAll(taskA, taskB);
            GenerationResult resultA = results[0], resultB = results[1];

            _output.WriteLine($"{name}: A reference=[{string.Join(",", referenceA.TokenIds)}]");
            _output.WriteLine($"{name}: A scheduled =[{string.Join(",", resultA.TokenIds)}]");
            _output.WriteLine($"{name}: B reference=[{string.Join(",", referenceB.TokenIds)}]");
            _output.WriteLine($"{name}: B scheduled =[{string.Join(",", resultB.TokenIds)}]");

            if (!referenceA.TokenIds.SequenceEqual(resultA.TokenIds))
                failures.Add($"{name}: sequence A (solo-admitted, retired mid-generation) diverged from its graph-decode-off reference — the graph→eager splice may have corrupted KV state");
            if (!referenceB.TokenIds.SequenceEqual(resultB.TokenIds))
                failures.Add($"{name}: sequence B (admitted while A was active) diverged from its own solo reference");
        }
        Assert.True(failures.Count == 0, string.Join("\n", failures));
    }
}
