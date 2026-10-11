using HartsyInference.Core.Configuration;
using HartsyInference.Cuda;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Sampling;
using HartsyInference.LLM.Transformer;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>The scheduler's graph-eligible solo sequence follows <c>vram.kvF16</c> like <see cref="TextGenerationPipeline"/> does:
/// graph decode writes the F16 cache through the F16 scatter kernels, and a session retired by a joining request continues
/// eagerly on the same F16 cache. Gated on <c>HARTSY_TEST_GGUF_MODELS</c> + CUDA, same as <see cref="SchedulerGraphDecodeTests"/>.</summary>
[Collection("CudaSerial")]
public sealed class SchedulerKvF16Tests
{
    private readonly ITestOutputHelper _output;
    public SchedulerKvF16Tests(ITestOutputHelper output) => _output = output;

    private const string PromptA = "Continue this repeating pattern with more numbers, comma separated, no other text: 1, 2, 3, 4, 1, 2, 3, 4,";
    private const string PromptB = "Say hello in exactly three words.";
    private const int PrefixTokens = 8;

    private static string[] ModelPaths()
    {
        string? env = Environment.GetEnvironmentVariable("HARTSY_TEST_GGUF_MODELS");
        if (string.IsNullOrWhiteSpace(env)) return [];
        return [.. env.Split([';', ','], StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).Where(File.Exists)];
    }

    private static PagedKvPool NewPool(TransformerConfig cfg) =>
        new(cfg.NumLayers, cfg.NumKvHeads, cfg.HeadDim, pageSize: 16, maxPages: 512);

    private static GenerationRequest Req(string prompt, int max) => new()
    { Prompt = prompt, MaxTokens = max, Sampling = SamplingOptions.Default with { Greedy = true }, GraphDecode = true };

    private static async Task<GenerationResult> Scheduled(GgufLanguageModel model, CudaBackend backend, string prompt, int max)
    {
        using PagedKvPool pool = NewPool(model.Transformer.Config);
        using DynamicBatchScheduler scheduler = new(model.Transformer, model.Tokenizer, backend, pool, model.Template);
        return await scheduler.SubmitAsync(Req(prompt, max), onToken: null, CancellationToken.None);
    }

    /// <summary>True when the first <paramref name="n"/> tokens (or all of a shorter result) agree; F16 storage may legitimately end a run earlier or later, so lengths are not compared.</summary>
    private static bool PrefixEqual(IReadOnlyList<int> a, IReadOnlyList<int> b, int n) =>
        a.Count > 0 && b.Count > 0 && a.Take(n).SequenceEqual(b.Take(n));

    [Trait("Category", "GpuIntegration")]
    [Fact]
    public async Task SoloGraphSequence_WithKvF16_MatchesPipelineF16_AndTracksF32()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        string[] models = ModelPaths();
        if (models.Length == 0) { _output.WriteLine("SKIPPED: HARTSY_TEST_GGUF_MODELS not set"); return; }
        List<string> failures = [];
        foreach (string path in models)
        {
            using CudaBackend backend = new(0, Path.Combine(AppContext.BaseDirectory, "Ptx"));
            using GgufLanguageModel model = GgufLanguageModel.Load(path, dequantizeToF32: false);
            if (!model.Transformer.SupportsGraphDecode(backend)) { _output.WriteLine($"SKIPPED (not graph-eligible): {Path.GetFileName(path)}"); continue; }
            string name = Path.GetFileName(path);
            GenerationResult f32 = await Scheduled(model, backend, PromptA, 40);
            GenerationResult f16, pipeF16;
            try
            {
                KnobStore.Set(EngineKnobs.KvF16, true);
                f16 = await Scheduled(model, backend, PromptA, 40);
                pipeF16 = new TextGenerationPipeline(model.Transformer, model.Tokenizer, backend, model.Template).Generate(Req(PromptA, 40));
            }
            finally { KnobStore.Clear(EngineKnobs.KvF16); }
            _output.WriteLine($"{name}: f32 =[{string.Join(",", f32.TokenIds)}]");
            _output.WriteLine($"{name}: f16 =[{string.Join(",", f16.TokenIds)}]");
            _output.WriteLine($"{name}: pipe=[{string.Join(",", pipeF16.TokenIds)}]");
            if (!f16.TokenIds.SequenceEqual(pipeF16.TokenIds)) failures.Add($"{name}: scheduler F16 output differs from TextGenerationPipeline F16 output");
            if (!PrefixEqual(f16.TokenIds, f32.TokenIds, PrefixTokens)) failures.Add($"{name}: F16-KV scheduler output diverges from F32-KV in the first {PrefixTokens} tokens");
        }
        Assert.True(failures.Count == 0, string.Join("\n", failures));
    }

    [Trait("Category", "GpuIntegration")]
    [Fact]
    public async Task JoinMidGeneration_WithKvF16_RetiredGraphSessionContinuesEagerlyOnF16Cache()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        string[] models = ModelPaths();
        if (models.Length == 0) { _output.WriteLine("SKIPPED: HARTSY_TEST_GGUF_MODELS not set"); return; }
        List<string> failures = [];
        foreach (string path in models)
        {
            using CudaBackend backend = new(0, Path.Combine(AppContext.BaseDirectory, "Ptx"));
            using GgufLanguageModel model = GgufLanguageModel.Load(path, dequantizeToF32: false);
            if (!model.Transformer.SupportsGraphDecode(backend)) { _output.WriteLine($"SKIPPED (not graph-eligible): {Path.GetFileName(path)}"); continue; }
            string name = Path.GetFileName(path);
            TextGenerationPipeline pipe = new(model.Transformer, model.Tokenizer, backend, model.Template);
            GenerationResult refA32 = pipe.Generate(Req(PromptA, 60) with { GraphDecode = false });
            GenerationResult refB32 = pipe.Generate(Req(PromptB, 40) with { GraphDecode = false });
            GenerationResult[] r;
            try
            {
                KnobStore.Set(EngineKnobs.KvF16, true);
                using PagedKvPool pool = NewPool(model.Transformer.Config);
                using DynamicBatchScheduler scheduler = new(model.Transformer, model.Tokenizer, backend, pool, model.Template);
                Task<GenerationResult> a = scheduler.SubmitAsync(Req(PromptA, 60), onToken: null, CancellationToken.None);
                Task<GenerationResult> b = scheduler.SubmitAsync(Req(PromptB, 40), onToken: null, CancellationToken.None);
                r = await Task.WhenAll(a, b);
            }
            finally { KnobStore.Clear(EngineKnobs.KvF16); }
            _output.WriteLine($"{name}: A f32 ref=[{string.Join(",", refA32.TokenIds)}]");
            _output.WriteLine($"{name}: A f16 sch=[{string.Join(",", r[0].TokenIds)}]");
            _output.WriteLine($"{name}: B f32 ref=[{string.Join(",", refB32.TokenIds)}]");
            _output.WriteLine($"{name}: B f16 sch=[{string.Join(",", r[1].TokenIds)}]");
            if (r[0].TokenIds.Count < PrefixTokens || !PrefixEqual(r[0].TokenIds, refA32.TokenIds, PrefixTokens))
                failures.Add($"{name}: sequence A (F16 cache, graph session retired mid-generation) diverged from F32 reference");
            if (!PrefixEqual(r[1].TokenIds, refB32.TokenIds, PrefixTokens))
                failures.Add($"{name}: sequence B (joined while A active) diverged from its F32 reference");
        }
        Assert.True(failures.Count == 0, string.Join("\n", failures));
    }
}
