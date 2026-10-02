using System.Diagnostics;
using System.Text;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.MemoryManagement;
using HartsyInference.Cuda;
using HartsyInference.Cuda.Profiling;
using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.LLM.Tests;

/// <summary>TTFT probe for the LLMAssistant 3060 slowness report: the SAME templated prompt (&gt;= 554 ordinary
/// tokens, counted via <see cref="ITextService.CountTokens"/> against the real Qwen3 tokenizer once a slot is
/// warm — a cold slot's <c>CountTokens</c> is a chars/4 heuristic, not the tokenizer), through the SAME
/// <see cref="InferenceEngine"/>/<see cref="TextRequest"/> construction <see cref="LLMAssistantBackToBackTurnTests"/>
/// uses (VramMode=Auto -&gt; VramPolicy=null, greedy, thinking off), run once on cuda:1 (3060) and once on cuda:0
/// (4090) so the two numbers are directly comparable. <c>diagnostics.profile</c> is turned on via
/// <see cref="KnobStore.Set"/> (env vars are NOT read by the knob store any more) and the accumulated per-op
/// table is dumped after each run, so a 10x gap shows WHICH op class owns it instead of leaving that to guesswork.
///
/// <para>Opt-in (<c>HARTSY_LLM_TTFT_PROBE=1</c>): touches both physical GPUs for a timed measurement, so it must
/// only run under this repo's swarm-quiet-window + lock protocol, never as part of an ordinary suite pass.</para></summary>
[Trait("Category", "GpuIntegration")]
[Trait("Category", "RealWeights")]
[Trait("Category", "Slow")]
public sealed class LlmAssistant3060TtftProbeTests
{
    private const string GateEnvVar = "HARTSY_LLM_TTFT_PROBE";
    private const int TargetPromptTokens = 554;
    private const string ContextParagraph =
        "Call notes so far: the caller is phoning about an order placed last week. They want to change the delivery "
        + "address to their office, ask whether the courier can call ahead, and confirm the refund on a returned item "
        + "has been issued. The account shows one open order, one completed return, and a note that the previous agent "
        + "promised a callback that never happened. Company policy: address changes are allowed until the parcel is "
        + "scanned at the depot, refunds take three to five business days, and any promise of a callback must be logged. ";
    private const string Question = "Summarize what the caller needs in two short sentences.";

    private readonly ITestOutputHelper _out;

    public LlmAssistant3060TtftProbeTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public async Task Qwen3_4B_Ttft_3060_vs_4090_SameRequest()
    {
        string checkpoint = TestPaths.Llm.Qwen3_4BQ4KM;
        if (!RealWeightGate.Require(_out.WriteLine, checkpoint)) return;
        if (Environment.GetEnvironmentVariable(GateEnvVar) != "1")
        {
            _out.WriteLine($"SKIPPED: set {GateEnvVar}=1 to run (touches both GPUs — follow the swarm-quiet-window "
                + "+ orch_bench.lock protocol before setting it).");
            return;
        }
        Assert.True(CudaContext.IsAvailable(), $"CUDA unavailable: {CudaContext.LastUnavailableReason}");
        Assert.True(CudaContext.GetDeviceCount() >= 2, "needs two CUDA devices (ordinals 0 and 1).");

        using InferenceEngine engine = new("cuda", new EngineOptions { VramPolicy = null });
        ModelSpec spec = new() { Requested = "qwen3-4b", Modality = Modality.Text, LocalPath = checkpoint };

        // Warm BOTH devices before any timed measurement: each device gets its own TextService slot (its own cold
        // GGUF load + PreloadWeights), and a timed run against a never-yet-loaded device would bake that one-time
        // load into its "TTFT" — comparing a warm 3060 against a cold 4090 is not a TTFT comparison at all.
        await RunOnceAsync(engine, spec, "Hi.", 1, profile: false);
        await RunOnceAsync(engine, spec, "Hi.", 0, profile: false);

        string prompt = BuildPrompt(engine, spec, out int promptTokens);
        _out.WriteLine($"prompt: {promptTokens} ordinary tokens (target >= {TargetPromptTokens})");

        // Both slots stay resident (no Unload between devices) for the same reason — only the FIRST timed call on
        // either device may legitimately pay a load, and both already have.
        foreach (int ordinal in new[] { 1, 0 })
        {
            KnobStore.Set(EngineKnobs.Profile, true);
            NvtxRange.ResetProfile();
            Stopwatch clock = Stopwatch.StartNew();
            (double ttftSeconds, int completionTokens) = await RunOnceAsync(engine, spec, prompt, ordinal, profile: true, clock);
            _out.WriteLine($"cuda:{ordinal}: TTFT {ttftSeconds * 1000:F1} ms, {completionTokens} completion token(s), "
                + $"{promptTokens} prompt tokens.");

            string dumpPath = Path.Combine(Path.GetTempPath(), $"llmfix_profile_cuda{ordinal}.txt");
            NvtxRange.DumpProfile(dumpPath);
            if (File.Exists(dumpPath))
            {
                _out.WriteLine($"--- per-op profile (cuda:{ordinal}, top by total ms) ---");
                _out.WriteLine(File.ReadAllText(dumpPath));
            }
            else
            {
                _out.WriteLine($"(no profile table written for cuda:{ordinal} — NvtxRange may be disabled: "
                    + "check for the 'NVTX disabled' stderr line.)");
            }
            KnobStore.Set(EngineKnobs.Profile, false);
            NvtxRange.ResetProfile();
        }
        // Both devices' slots are released together at the end, not between timed runs.
        engine.Text.Unload();
    }

    private static string BuildPrompt(InferenceEngine engine, ModelSpec spec, out int promptTokens)
    {
        StringBuilder sb = new();
        int tokens = 0;
        for (int copies = 0; copies < 40; copies++)
        {
            sb.Append(ContextParagraph);
            tokens = engine.Text.CountTokens(spec, sb.ToString() + "\n" + Question);
            if (tokens >= TargetPromptTokens)
            {
                break;
            }
        }
        promptTokens = tokens;
        return sb.ToString() + "\n" + Question;
    }

    private static async Task<(double TtftSeconds, int CompletionTokens)> RunOnceAsync(
        InferenceEngine engine, ModelSpec spec, string userMessage, int ordinal, bool profile, Stopwatch? clock = null)
    {
        TextRequest request = new()
        {
            Messages = [new TextMessage { Role = TextRole.User, Content = userMessage }],
            Temperature = 0,
            TopP = 1.0,
            TopK = 40,
            MinP = null,
            RepetitionPenalty = 1.1,
            MaxTokens = profile ? 32 : 4,
            Seed = -1,
            Greedy = true,
            EnableThinking = false,
            Device = $"cuda:{ordinal}",
        };
        double ttft = double.NaN;
        int count = 0;
        await foreach (TextChunk chunk in engine.Text.StreamAsync(spec, request))
        {
            if (chunk.Kind == TextChunkKind.Chunk)
            {
                if (clock is not null && double.IsNaN(ttft))
                {
                    ttft = clock.Elapsed.TotalSeconds;
                }
                count++;
            }
        }
        return (ttft, count);
    }
}
