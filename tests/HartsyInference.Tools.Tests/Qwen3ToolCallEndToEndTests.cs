using System.Text;
using HartsyInference.Cuda;
using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using HartsyInference.LLM.ChatTemplates;
using HartsyInference.LLM.Generation;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Tools.Tests;

/// <summary>The PR6 bring-up gate on real weights: Qwen3-4B (catalog id <c>qwen3</c>, Q4_K_M GGUF) loaded through <see cref="InferenceEngine"/> on CUDA with <see cref="ToolCalling.Install"/>, offered one <c>hang_up</c> tool and asked to hang up, must stream a parsed <see cref="NativeToolCall"/> and end as <see cref="StopReason.ToolCall"/>; and the checkpoint's own <c>chat_template</c> must compile to <see cref="JinjaChatTemplate"/> (not the silent ChatML fallback) and render the <c>&lt;tools&gt;</c> block. Run by the orchestrator on the 3060: <c>CUDA_VISIBLE_DEVICES=1 HARTSY_REQUIRE_REAL_WEIGHTS=1 dotnet test tests/HartsyInference.Tools.Tests -c Release --filter "FullyQualifiedName~Qwen3ToolCallEndToEndTests"</c>.</summary>
[Trait("Category", "RealWeights")]
[Trait("Category", "GpuIntegration")]
public sealed class Qwen3ToolCallEndToEndTests
{
    private const string UserTurn = "please hang up now";
    private const int MaxTokens = 96;

    private static readonly ToolDefinition HangUp = ToolSchema.FromDelegate("hang_up", static () => "call ended", "Ends the current phone call immediately.");

    private readonly ITestOutputHelper _output;

    public Qwen3ToolCallEndToEndTests(ITestOutputHelper output) => _output = output;

    private static TextRequest Request() => new()
    {
        Messages = [new TextMessage { Role = TextRole.User, Content = UserTurn }],
        Tools = [HangUp],
        EnableThinking = false,
        Greedy = true,
        Temperature = 0,
        MaxTokens = MaxTokens,
        Seed = 42,
    };

    /// <summary>Null (with a SKIPPED line) when CUDA or the checkpoint is unavailable; <c>HARTSY_REQUIRE_REAL_WEIGHTS=1</c> turns the checkpoint miss into a failure.</summary>
    private ModelSpec? Gate()
    {
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable.");
            return null;
        }
        string checkpoint = TestPaths.Llm.Qwen3_4BQ4KM;
        if (!RealWeightGate.Require(_output.WriteLine, checkpoint)) return null;
        ModelSpec spec = ModelResolver.Resolve("qwen3", checkpoint, Modality.Text);
        Assert.NotNull(spec.Catalog);
        Assert.Equal("qwen3", spec.Catalog.Id);
        Assert.Equal(checkpoint, spec.LocalPath);
        return spec;
    }

    [Fact]
    public void CheckpointTemplateIsJinjaAndRendersTheToolsBlock()
    {
        ModelSpec? spec = Gate();
        if (spec is null) return;
        using GgufLanguageModel model = GgufLanguageModel.Load(spec.LocalPath!);
        Assert.IsType<JinjaChatTemplate>(model.Template);
        ToolSpec tool = ToolSpec.FromJson(
            "{\"type\":\"function\",\"function\":{\"name\":\"hang_up\",\"description\":\"Ends the current phone call immediately.\",\"parameters\":"
            + HangUp.JsonSchema + "}}");
        int[] ids = model.Template.Encode(model.Tokenizer, [ChatMessage.User(UserTurn)], addGenerationPrompt: true, enableThinking: false, tools: [tool]);
        string rendered = model.Tokenizer.Decode(ids);
        _output.WriteLine(rendered);
        Assert.Contains("<tools>", rendered, StringComparison.Ordinal);
        Assert.Contains("\"hang_up\"", rendered, StringComparison.Ordinal);
        Assert.Contains(UserTurn, rendered, StringComparison.Ordinal);
    }

    [Fact]
    public async Task StreamYieldsAParsedHangUpCallAndStopsAsToolCall()
    {
        ModelSpec? spec = Gate();
        if (spec is null) return;
        EngineOptions options = new();
        ToolCalling.Install(options);
        using InferenceEngine engine = new("cuda", options);
        List<TextChunk> chunks = [];
        await foreach (TextChunk chunk in engine.Text.StreamAsync(spec, Request()))
        {
            chunks.Add(chunk);
            _output.WriteLine($"{chunk.Kind}: {chunk.Text ?? chunk.ToolCall?.Name ?? chunk.Stop?.ToString() ?? ""}");
        }
        string forwarded = string.Concat(chunks.Where(c => c.Kind == TextChunkKind.Chunk).Select(c => c.Text));
        Assert.DoesNotContain("<think>", forwarded, StringComparison.Ordinal);
        Assert.DoesNotContain("<tool_call>", forwarded, StringComparison.Ordinal);
        TextChunk call = Assert.Single(chunks, c => c.Kind == TextChunkKind.NativeToolCall);
        Assert.Equal("hang_up", call.ToolCall!.Name);
        Assert.Equal("call_0", call.ToolCall.Id);
        Assert.Equal(TextChunkKind.StopReason, chunks[^1].Kind);
        Assert.Equal(StopReason.ToolCall, chunks[^1].Stop);
        Assert.Equal(TextChunkKind.Result, chunks[^2].Kind);
        Assert.DoesNotContain("<think>", chunks[^2].Text ?? "", StringComparison.Ordinal);
    }

    [Fact]
    public async Task GenerateReturnsTheCallAndTheLoopFeedsItsResultBack()
    {
        ModelSpec? spec = Gate();
        if (spec is null) return;
        EngineOptions options = new();
        ToolCalling.Install(options);
        using InferenceEngine engine = new("cuda", options);

        TextResult result = await engine.Text.GenerateAsync(spec, Request());
        _output.WriteLine($"generate: stop={result.Stop} text=\"{result.Text}\" call={result.ToolCall?.Name}");
        Assert.Equal(StopReason.ToolCall, result.Stop);
        Assert.Equal("hang_up", result.ToolCall?.Name);

        int invoked = 0;
        ToolRegistry registry = new ToolRegistry().Add("hang_up", () => { invoked++; return "call ended"; }, "Ends the current phone call immediately.");
        StringBuilder visible = new();
        List<TextChunk> chunks = [];
        await foreach (TextChunk chunk in ToolLoop.RunAsync(engine.Text, spec, Request() with { Tools = null }, registry, maxRounds: 2))
        {
            chunks.Add(chunk);
            if (chunk.Kind == TextChunkKind.Chunk) visible.Append(chunk.Text);
            _output.WriteLine($"loop {chunk.Kind}: {chunk.Text ?? chunk.ToolCall?.Name ?? chunk.Stop?.ToString() ?? ""}");
        }
        Assert.Equal(1, invoked);
        Assert.Contains(chunks, c => c.Kind == TextChunkKind.Status && c.Text == ToolLoop.ToolResultPrefix + "call ended");
        Assert.Equal(TextChunkKind.StopReason, chunks[^1].Kind);
        Assert.NotEqual(StopReason.Error, chunks[^1].Stop);
        Assert.DoesNotContain("<think>", visible.ToString(), StringComparison.Ordinal);
    }
}
