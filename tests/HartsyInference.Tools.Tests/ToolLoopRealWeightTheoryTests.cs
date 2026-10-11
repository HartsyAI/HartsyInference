using HartsyInference.Cuda;
using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.Tests.Common;
using HartsyInference.Tools.Parsing;
using System.Text;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Tools.Tests;

/// <summary>The regression matrix on real checkpoints: each model runs one tool turn through <see cref="ToolLoop"/> and must emit a parsed <c>get_time</c> call in the expected dialect, with the tool prompt injected only where its template has no tools slot, then take a second round. Models that are missing skip with a line (<c>HARTSY_REQUIRE_REAL_WEIGHTS=1</c> turns that into a failure). The device is CPU unless <c>TOOLCALL_PROBE_DEVICE</c> says otherwise.</summary>
public sealed class ToolLoopRealWeightTheoryTests
{
    private readonly ITestOutputHelper _output;

    public ToolLoopRealWeightTheoryTests(ITestOutputHelper output) => _output = output;

    /// <summary>Fast checkpoints: key, catalog id, expected dialect, and whether the tool prompt is injected.</summary>
    public static TheoryData<string, string, ToolCallFormat, bool> FastCases => new()
    {
        { "qwen3-0.6b", "qwen3", ToolCallFormat.Hermes, false },
        { "qwen2.5-0.5b", "qwen2.5", ToolCallFormat.Hermes, false },
        { "qwen3.5-0.8b", "qwen35", ToolCallFormat.QwenXml, false },
        { "llama-3.2-1b", "llama32", ToolCallFormat.Llama3, false },
        { "gemma-3-1b", "gemma3", ToolCallFormat.Hermes, true },
        { "phi-3.5-mini", "phi3", ToolCallFormat.Hermes, true },
    };

    /// <summary>Larger checkpoints, slow on CPU: run with the Slow category.</summary>
    public static TheoryData<string, string, ToolCallFormat, bool> SlowCases => new()
    {
        { "mistral-7b-v0.3", "mistral", ToolCallFormat.Hermes, true },
        { "glm-4-9b-0414", "glm4", ToolCallFormat.Hermes, false },
    };

    /// <summary>Checkpoints that run only on CUDA. Qwen3.5's Q5_K dense weights have no CPU matmul path; the 3.8B-and-larger
    /// models widen their weights to F32 on the CPU (about 15 GB or more), which the OOM killer ends on a 39 GB host.</summary>
    private static readonly HashSet<string> CudaOnly = new(StringComparer.Ordinal) { "qwen3.5-0.8b", "phi-3.5-mini", "mistral-7b-v0.3", "glm-4-9b-0414" };

    [Theory]
    [MemberData(nameof(FastCases))]
    [Trait("Category", "Integration")]
    [Trait("Category", "RealWeights")]
    public async Task FastCheckpointCallsTheToolInItsDialect(string key, string modelId, ToolCallFormat expected, bool injected)
        => await RunCase(key, modelId, expected, injected);

    [Theory]
    [MemberData(nameof(SlowCases))]
    [Trait("Category", "Integration")]
    [Trait("Category", "RealWeights")]
    [Trait("Category", "Slow")]
    public async Task SlowCheckpointCallsTheToolInItsDialect(string key, string modelId, ToolCallFormat expected, bool injected)
        => await RunCase(key, modelId, expected, injected);

    private static string PathFor(string key) => key switch
    {
        "qwen3-0.6b" => TestPaths.Llm.Qwen3_06BQ4KM,
        "qwen2.5-0.5b" => TestPaths.Llm.Qwen25_05BQ4KM,
        "qwen3.5-0.8b" => TestPaths.Llm.Qwen35_08BQ4KM,
        "llama-3.2-1b" => TestPaths.Llm.Llama32_1BQ8Flat,
        "gemma-3-1b" => TestPaths.Llm.Gemma3_1BItQ4KM,
        "phi-3.5-mini" => TestPaths.Llm.Phi35MiniQ4KM,
        "mistral-7b-v0.3" => TestPaths.Llm.Mistral7BInstructV0_3Q4KM,
        "glm-4-9b-0414" => TestPaths.Llm.Glm4_9BQ4KM,
        _ => throw new ArgumentOutOfRangeException(nameof(key), key, "Unknown checkpoint key."),
    };

    private async Task RunCase(string key, string modelId, ToolCallFormat expected, bool expectInjection)
    {
        string path = PathFor(key);
        if (!RealWeightGate.Require(_output.WriteLine, path)) return;
        string device = Environment.GetEnvironmentVariable("TOOLCALL_PROBE_DEVICE") is { Length: > 0 } chosen ? chosen : "cpu";
        if (device == "cuda" && !CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable.");
            return;
        }
        if (device == "cpu" && CudaOnly.Contains(key))
        {
            _output.WriteLine($"SKIPPED: {key} needs CUDA (CPU widens its weights to F32, or its quant has no CPU matmul path).");
            return;
        }

        List<(ToolCallFormat? Format, bool Injected)> resolved = [];
        StringBuilder raw = new();
        EngineOptions options = new();
        options.TextStreamFilterFactory = context =>
        {
            ITextStreamFilter? filter = ToolCalling.CreateFilter(context);
            resolved.Add((filter is ToolCallStreamFilter parser ? parser.Format : null, context.ToolPromptInjected));
            return filter is null ? null : new RecordingFilter(filter, raw);
        };
        using InferenceEngine engine = new(device, options);
        ModelSpec spec = ModelResolver.Resolve(modelId, path, Modality.Text);
        ToolRegistry registry = new ToolRegistry().Add("get_time", "Returns the current time.", "{\"type\":\"object\",\"properties\":{}}",
            (_, _) => Task.FromResult("{\"time\":\"14:05\"}"));
        TextRequest request = new()
        {
            Messages = [new TextMessage { Role = TextRole.User, Content = "What time is it? Call the get_time function now. Do not answer in text." }],
            Tools = registry.Definitions,
            EnableThinking = false,
            Greedy = true,
            Temperature = 0,
            MaxTokens = 160,
            Seed = 42,
        };
        ToolLoopRun run = ToolLoop.Create(engine.Text, spec, request, registry, new ToolLoopOptions { MaxRounds = 2 });
        List<TextChunk> chunks = [];
        await foreach (TextChunk chunk in run.RunAsync()) chunks.Add(chunk);

        NativeToolCall? call = chunks.Where(c => c.Kind == TextChunkKind.NativeToolCall).Select(c => c.ToolCall).FirstOrDefault();
        _output.WriteLine($"{key}: rounds={run.Rounds} stop={run.Stop} format={resolved.FirstOrDefault().Format} injected={resolved.FirstOrDefault().Injected} call={call?.Name ?? "none"}");
        _output.WriteLine($"{key}: raw=[{raw}]");
        Assert.NotEmpty(resolved);
        Assert.NotNull(call);
        Assert.Equal("get_time", call!.Name);
        Assert.Equal(expected, resolved[0].Format);
        Assert.Equal(expectInjection, resolved[0].Injected);
        Assert.True(run.Rounds >= 2, $"expected a second round after the tool result, got {run.Rounds}");
        Assert.Contains(chunks, c => c.Kind == TextChunkKind.ToolResult);
        // Soft: small models are unreliable at echoing a tool result, so this is logged rather than asserted.
        _output.WriteLine($"{key}: final text mentions the result: {run.VisibleText.Contains("14:05", StringComparison.Ordinal)}");
    }

    /// <summary>Records every decoded delta the parser sees, then forwards it, so a failing case shows what the model wrote.</summary>
    private sealed class RecordingFilter(ITextStreamFilter inner, StringBuilder raw) : ITextStreamFilter
    {
        public IReadOnlyCollection<string> MarkerLiterals => inner.MarkerLiterals;

        public TextFilterResult OnDelta(string delta)
        {
            raw.Append(delta);
            return inner.OnDelta(delta);
        }

        public TextFilterResult OnEnd() => inner.OnEnd();
    }
}
