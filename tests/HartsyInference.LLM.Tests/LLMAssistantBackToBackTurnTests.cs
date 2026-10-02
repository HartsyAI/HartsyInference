using HartsyInference.Core.MemoryManagement;
using HartsyInference.Cuda;
using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.LLM.Tests;

/// <summary>Reproduces HartsyLocalLLMProvider's exact engine construction (<c>new InferenceEngine("cuda", options)</c>
/// with <c>VramMode=Auto</c> -&gt; <c>VramPolicy=null</c>, matching every default LLMAssistant install) and
/// per-request shape (<c>TextRequest.Device="cuda:1"</c>, greedy, <c>EnableThinking=false</c>, <c>TopK=40</c>,
/// <c>RepetitionPenalty=1.1</c>), then runs several two-turn sequences back to back through
/// <see cref="ITextService.StreamAsync"/> (the entry point <c>LLMAssistantVoiceTurnWS</c> uses) with a ~30ms gap
/// between turn 1's stream completing and turn 2 starting — turn 2 carries turn 1's user message, turn 1's own
/// reply, and a new, much longer user message, the same growing-conversation shape the 2026-10-01 19:58:54Z
/// incident logged.
///
/// <para>This sanity check alone does not reproduce that incident's intermittent
/// <c>CUDA_ERROR_INVALID_VALUE</c> — 60 isolated sequences on an otherwise idle card passed cleanly, and the race
/// (<see cref="GpuTransferHelper.State.StreamHandle"/> zeroed by a concurrent backend retirement) needs a second
/// thread actually retiring the backend mid-op, which <see cref="GpuTransferHelperStreamRetirementTests"/> in the
/// Cuda test project reproduces deterministically instead. This test exists so the exact LLMAssistant request
/// shape is exercised somewhere end to end, independent of that race.</para></summary>
[Trait("Category", "GpuIntegration")]
[Trait("Category", "RealWeights")]
public sealed class LLMAssistantBackToBackTurnTests
{
    private const int Ordinal = 1; // engine ordinal 1 = RTX 3060 on the dev box (fastest-first enumeration).
    private const int Sequences = 10;

    private readonly ITestOutputHelper _out;

    public LLMAssistantBackToBackTurnTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public async Task TwoTurnSequence_RepeatedBackToBack_DoesNotThrow()
    {
        string checkpoint = TestPaths.Llm.Qwen3_4BQ4KM;
        if (!RealWeightGate.Require(_out.WriteLine, checkpoint)) return;
        Assert.True(CudaContext.IsAvailable(), $"CUDA unavailable: {CudaContext.LastUnavailableReason}");
        Assert.True(CudaContext.GetDeviceCount() > Ordinal, $"needs a CUDA ordinal {Ordinal} device.");

        // Mirrors HartsyLocalLLMProvider.OnProviderInit: VramMode "Auto" (the provider's own default) maps to a
        // null VramPolicy, and the engine's own backend selector carries no ordinal — only TextRequest.Device picks
        // cuda:1, exactly like every real LLMAssistant request.
        using InferenceEngine engine = new("cuda", new EngineOptions { VramPolicy = null });
        ModelSpec spec = new() { Requested = "qwen3-4b", Modality = Modality.Text, LocalPath = checkpoint };

        string longContext = string.Concat(Enumerable.Repeat(
            "The caller said their account number is on file and they want to reschedule a delivery for next week. ", 40));
        int failures = 0;
        Exception? firstFailure = null;
        for (int i = 0; i < Sequences; i++)
        {
            try
            {
                string turn1User = "Hi, quick question for you.";
                TextResult turn1 = await RunTurnAsync(engine, spec, [
                    new TextMessage { Role = TextRole.User, Content = turn1User },
                ]);

                await Task.Delay(30);

                // Turn 2 approximates the real incident's shape: turn 1 + its reply + a new, much longer user
                // message (growing conversation, ~500+ templated tokens), not a trivially short follow-up.
                TextResult turn2 = await RunTurnAsync(engine, spec, [
                    new TextMessage { Role = TextRole.User, Content = turn1User },
                    new TextMessage { Role = TextRole.Assistant, Content = turn1.Text },
                    new TextMessage { Role = TextRole.User, Content = longContext + "Given all that, what should happen next?" },
                ]);
                _out.WriteLine($"seq {i}: turn1 {turn1.CompletionTokens} tok, turn2 {turn2.CompletionTokens} tok — ok");
            }
            catch (Exception ex)
            {
                failures++;
                firstFailure ??= ex;
                _out.WriteLine($"seq {i}: FAILED — {ex.GetType().Name}: {ex.Message}");
            }
        }
        _out.WriteLine($"{failures}/{Sequences} sequences failed.");
        if (firstFailure is not null)
        {
            _out.WriteLine(firstFailure.ToString());
        }
        Assert.Equal(0, failures);
    }

    private static async Task<TextResult> RunTurnAsync(InferenceEngine engine, ModelSpec spec, List<TextMessage> messages)
    {
        TextRequest request = new()
        {
            Messages = messages,
            Temperature = 0,
            TopP = 1.0,
            TopK = 40,
            MinP = null,
            RepetitionPenalty = 1.1,
            MaxTokens = 48,
            Seed = -1,
            Greedy = true,
            EnableThinking = false,
            Device = $"cuda:{Ordinal}",
        };
        string text = "";
        StopReason stop = StopReason.Stop;
        int completionTokens = 0;
        await foreach (TextChunk chunk in engine.Text.StreamAsync(spec, request))
        {
            switch (chunk.Kind)
            {
                case TextChunkKind.Chunk:
                    text += chunk.Text;
                    completionTokens++;
                    break;
                case TextChunkKind.Result:
                    text = chunk.Text ?? text;
                    break;
                case TextChunkKind.StopReason:
                    stop = chunk.Stop ?? stop;
                    break;
            }
        }
        return new TextResult { Text = text, Stop = stop, CompletionTokens = completionTokens };
    }
}
