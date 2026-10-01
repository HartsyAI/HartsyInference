using System.Diagnostics;
using System.Text;
using HartsyInference.Core.Configuration;
using HartsyInference.Cuda;
using HartsyInference.LLM.ChatTemplates;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Sampling;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.LLM.Tests;

/// <summary>perf/llm-short-reply-decode investigation step 1: per-decode-step timestamps straight out of
/// <see cref="TextGenerationPipeline.Generate"/> — no <c>StreamAsync</c>, no <c>ToolLoop</c>, no detokenizer/
/// filter/channel, a bare <c>onToken</c> callback that only timestamps — to isolate "per-generation startup cost"
/// (CUDA state warm-up, KV-cache allocation, sampler-chain construction: hypothesis A) from the sampler's own
/// per-token CPU cost. Three axes, all on Qwen3-4B Q4_K_M / CUDA ordinal 0 (the 4090):
/// <list type="bullet">
/// <item>fresh vs. second (same process, right after) vs. long (128 tokens) generation — the per-generation-cost
/// axis;</item>
/// <item>tools on (sentinel grammar matching <c>ToolCalling</c>'s "&lt;tool_call&gt;" sentinel) vs. tools off;</item>
/// <item>greedy (<see cref="SamplingOptions.GreedyPreset"/> — what the Phase-1 voice-turn probe measured) vs. the
/// voice session's ACTUAL default sampling. <c>VoiceAgentSession.Turns.cs</c>'s <c>BuildRequest</c> (PR #202,
/// <c>feat/voice-package</c>) never sets <c>Greedy</c>/<c>Temperature</c>/<c>TopP</c> on its <c>TextRequest</c>, so
/// <c>TextService.BuildSampling</c> carries <c>TextRequest</c>'s own defaults straight through: Temperature=0.7,
/// TopP=0.95, Greedy=false (see <c>TextRequest.cs</c> and <c>TextService.BuildSampling</c>). The Phase-1 probe
/// (<c>VoiceLlmTurnBenchTests</c>, same branch) measured ONLY <see cref="SamplingOptions.GreedyPreset"/> for both
/// its "tools on" and "tools off" rows — it never exercised the session's real (non-greedy) sampler path.</item>
/// </list>
/// Opt-in with <c>HARTSY_SHORTREPLY_BENCH=1</c>; otherwise returns early. Opens <c>CudaBackend(0)</c> and FAILS
/// unless the device name contains "4090" (engine ordinal 0 is the fastest card on the reference box).</summary>
[Trait("Category", "GpuIntegration")]
[Trait("Category", "RealWeights")]
public sealed class ShortReplyDecodeStepBenchTests
{
    private const string GateEnvVar = "HARTSY_SHORTREPLY_BENCH";
    private const string OutEnvVar = "HARTSY_SHORTREPLY_BENCH_OUT";
    private const string RequiredDeviceSubstring = "4090";
    private const string ToolSentinel = "<tool_call>";
    private const string SystemPrompt = "You are a phone agent. Answer briefly and plainly.";
    private const string HangUpToolJson =
        "{\"type\":\"function\",\"function\":{\"name\":\"hang_up\",\"description\":\"End the call.\","
        + "\"parameters\":{\"type\":\"object\",\"properties\":{\"reason\":{\"type\":\"string\"}},\"required\":[\"reason\"]}}}";

    // Deliberately asks for a longer answer than a real voice reply would give: this test's job is to see steps
    // 1..32 AND a steady-state window in the SAME generation, which a natural 9-token voice reply cannot show.
    private const string Question = "Explain in detail what the caller needs and what we should do about each point.";

    private readonly ITestOutputHelper _out;

    public ShortReplyDecodeStepBenchTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public void PerStepGaps_FreshSecondAndLong_GreedyVsSessionDefaultSampling()
    {
        if (Environment.GetEnvironmentVariable(GateEnvVar) != "1")
        {
            _out.WriteLine($"SKIPPED: set {GateEnvVar}=1 to run the short-reply decode-step bench.");
            return;
        }
        string checkpoint = TestPaths.Llm.Qwen3_4BQ4KM;
        if (!RealWeightGate.Require(_out.WriteLine, checkpoint)) return;
        Assert.True(CudaContext.IsAvailable(), $"CUDA unavailable: {CudaContext.LastUnavailableReason}");
        string? ptx = BackendGate.KernelDir("Ptx", "HartsyInference.Cuda");
        Assert.False(ptx is null, "no compiled PTX directory beside the tests or in the repo");

        using CudaBackend backend = new CudaBackend(0, ptx);
        string device = backend.Capabilities.DeviceName;
        _out.WriteLine($"CUDA ordinal 0: {device}");
        Assert.True(device.Contains(RequiredDeviceSubstring, StringComparison.Ordinal),
            $"ordinal 0 is '{device}', not a {RequiredDeviceSubstring}.");

        Stopwatch load = Stopwatch.StartNew();
        using GgufLanguageModel model = GgufLanguageModel.Load(checkpoint);
        TextGenerationPipeline pipeline = new TextGenerationPipeline(model.Transformer, model.Tokenizer, backend, model.Template);
        _out.WriteLine($"{Path.GetFileName(checkpoint)} loaded in {load.Elapsed.TotalSeconds:F1}s (arch {model.Architecture}); "
            + $"graphDecode knob={EngineKnobs.GraphDecode.Value} specDecode knob={EngineKnobs.SpecDecode.Value}");

        StringBuilder table = new StringBuilder();
        table.AppendLine("### Decode-step bench — Qwen3-4B Q4_K_M (4090, in-process, bare `pipeline.Generate`, thinking off)");
        table.AppendLine();
        table.AppendLine("| scenario | run | tokens | first-gap median ms (steps 1..9) | steady-gap median ms (last <=12) | overall tok/s |");
        table.AppendLine("|---|---|---:|---:|---:|---:|");

        foreach ((string label, bool tools, SamplingOptions sampling) in Scenarios())
        {
            GenerationRequest fresh = BuildRequest(tools, sampling, maxTokens: 40);
            RunAndReport(pipeline, fresh, label, "fresh", table);

            GenerationRequest second = BuildRequest(tools, sampling, maxTokens: 40);
            RunAndReport(pipeline, second, label, "second", table);

            GenerationRequest longGen = BuildRequest(tools, sampling, maxTokens: 128);
            RunAndReport(pipeline, longGen, label, "long(128)", table);
        }
        Emit(table);
    }

    private static IEnumerable<(string Label, bool Tools, SamplingOptions Sampling)> Scenarios()
    {
        yield return ("greedy, tools on (probe-equivalent)", true, SamplingOptions.GreedyPreset with { JsonModeSentinel = ToolSentinel });
        yield return ("session-default (temp .7/topP .95), tools on", true, SamplingOptions.Default with { Temperature = 0.7f, TopP = 0.95f, JsonModeSentinel = ToolSentinel });
        yield return ("session-default (temp .7/topP .95), tools off", false, SamplingOptions.Default with { Temperature = 0.7f, TopP = 0.95f });
    }

    private void RunAndReport(TextGenerationPipeline pipeline, GenerationRequest request, string scenario, string run, StringBuilder table)
    {
        List<double> timestamps = new(request.MaxTokens + 1);
        Stopwatch clock = Stopwatch.StartNew();
        GenerationResult result = pipeline.Generate(request, _ => timestamps.Add(clock.Elapsed.TotalSeconds));
        List<double> gaps = new(timestamps.Count);
        for (int i = 1; i < timestamps.Count; i++)
        {
            gaps.Add((timestamps[i] - timestamps[i - 1]) * 1000.0);
        }
        double firstMedian = Median(gaps.Take(Math.Min(9, gaps.Count)).ToList());
        int steadyCount = Math.Min(12, gaps.Count);
        double steadyMedian = Median(gaps.Skip(Math.Max(0, gaps.Count - steadyCount)).ToList());
        double overallRate = gaps.Count > 0 ? gaps.Count / gaps.Sum() * 1000.0 : 0.0;
        _out.WriteLine($"{scenario} / {run}: {result.TokenIds.Count} tokens, gaps(ms)=[{string.Join(", ", gaps.Select(g => g.ToString("F1")))}]");
        table.AppendLine($"| {scenario} | {run} | {result.TokenIds.Count} | {firstMedian:F1} | {steadyMedian:F1} | {overallRate:F1} |");
    }

    private static double Median(List<double> values)
    {
        if (values.Count == 0) return 0.0;
        List<double> sorted = values.OrderBy(v => v).ToList();
        int mid = sorted.Count / 2;
        return sorted.Count % 2 == 0 ? (sorted[mid - 1] + sorted[mid]) / 2.0 : sorted[mid];
    }

    private static GenerationRequest BuildRequest(bool tools, SamplingOptions sampling, int maxTokens)
    {
        List<ChatMessage> messages = new(2);
        if (tools)
        {
            messages.Add(ChatMessage.System(SystemPrompt + "\n\nTools available (call with <tool_call>{...}</tool_call>):\n" + HangUpToolJson)
                with { Tools = [ToolSpec.FromJson(HangUpToolJson)] });
        }
        else
        {
            messages.Add(ChatMessage.System(SystemPrompt));
        }
        messages.Add(ChatMessage.User(Question));
        return new GenerationRequest
        {
            Messages = messages,
            EnableThinking = false,
            MaxTokens = maxTokens,
            Sampling = sampling,
            // Matches ToolCalling's production gate: graph decode bypasses the CPU sampler chain (incl. grammar),
            // so it is structurally excluded whenever JSON masking is live, regardless of this knob. Forcing it
            // off for the tools-on rows keeps the request explicit, like the Phase-1 probe did.
            GraphDecode = tools ? false : null,
            SpeculativeDecode = tools ? false : null,
        };
    }

    private void Emit(StringBuilder table)
    {
        string text = table.ToString();
        _out.WriteLine(text);
        string? outPath = Environment.GetEnvironmentVariable(OutEnvVar);
        if (!string.IsNullOrEmpty(outPath))
        {
            File.AppendAllText(outPath, text + Environment.NewLine);
            _out.WriteLine($"appended to {outPath}");
        }
    }
}
