using System.Text;
using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Tools.Tests;

/// <summary>Runs one tool-offering turn on a real GGUF (CPU unless TOOLCALL_PROBE_DEVICE says otherwise) and reports the raw streamed text, the parsed call and the stop reason. Skips when the file is absent (set the path env var).</summary>
public sealed class ToolCallCpuProbeTests
{
    private static readonly ToolDefinition HangUp = ToolSchema.FromDelegate("hang_up", static () => "call ended", "Ends the current phone call immediately.");

    private readonly ITestOutputHelper _output;

    public ToolCallCpuProbeTests(ITestOutputHelper output) => _output = output;

    private static TextRequest Request() => new()
    {
        Messages = [new TextMessage { Role = TextRole.User, Content = "Call the hang_up function now. Do not answer in text." }],
        Tools = [HangUp],
        EnableThinking = false,
        Greedy = true,
        Temperature = 0,
        MaxTokens = 96,
        Seed = 42,
    };

    [Fact]
    [Trait("Category", "Integration")]
    public async Task Qwen3SmallCpuTurnEmitsAParsedHangUp()
        => await RunProbe("QWEN3_06B_GGUF_PATH", "qwen3");

    [Fact]
    [Trait("Category", "Integration")]
    public async Task Qwen35SmallCpuTurnEmitsAParsedHangUpInXml()
        => await RunProbe("QWEN35_08B_GGUF_PATH", "qwen35");

    private async Task RunProbe(string envVar, string modelId)
    {
        string? path = Environment.GetEnvironmentVariable(envVar);
        if (string.IsNullOrEmpty(path) || !File.Exists(path))
        {
            _output.WriteLine($"SKIPPED: set {envVar} to a GGUF for {modelId}.");
            return;
        }
        EngineOptions options = new();
        StringBuilder raw = new();
        options.TextStreamFilterFactory = context => ToolCalling.CreateFilter(context) is { } inner ? new RecordingFilter(inner, raw) : null;
        string device = Environment.GetEnvironmentVariable("TOOLCALL_PROBE_DEVICE") is { Length: > 0 } chosen ? chosen : "cpu";
        using InferenceEngine engine = new(device, options);
        ModelSpec spec = ModelResolver.Resolve(modelId, path, Modality.Text);
        StringBuilder text = new();
        NativeToolCall? call = null;
        StopReason? stop = null;
        await foreach (TextChunk chunk in engine.Text.StreamAsync(spec, Request()))
        {
            if (chunk.Kind == TextChunkKind.Chunk) text.Append(chunk.Text);
            if (chunk.Kind == TextChunkKind.NativeToolCall) call = chunk.ToolCall;
            if (chunk.Kind == TextChunkKind.StopReason) stop = chunk.Stop;
        }
        _output.WriteLine($"TEXT: {text}");
        _output.WriteLine("RAW_BEGIN" + raw + "RAW_END");
        _output.WriteLine($"CALL: {call?.Name ?? "none"} {call?.Arguments}");
        _output.WriteLine($"STOP: {stop}");
        Assert.NotNull(call);
        Assert.Equal("hang_up", call!.Name);
        Assert.Equal(StopReason.ToolCall, stop);
    }

    /// <summary>Records every decoded delta the parser sees, then forwards it, so a run's raw completion can be captured as a fixture.</summary>
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
