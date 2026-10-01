using System.Diagnostics;
using HartsyInference.Core.Logging;
using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;
using HartsyInference.Tests.Common;
using HartsyInference.Tools;
using HartsyInference.Voice.Gpu;
using HartsyInference.Voice.Tests.Fakes;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Voice.Tests;

/// <summary>The production split with a real language model: the audio engine on the RTX 3060 (engine <c>cuda:1</c> with
/// every card visible) and Qwen3-4B Q4_K_M, asked for by its catalog id <c>qwen3</c>, on the RTX 4090 through
/// <see cref="TextRequest.Device"/> = <c>cuda:0</c>, tool calling installed. The caller's first JFK utterance must be
/// heard, answered without a think block, spoken, and heard back. The 4090 is shared with SwarmUI, so beyond the Slow
/// category this runs only with
/// <c>HARTSY_VOICE_LLM_GPU=1</c>, set by the orchestrator when it grants the card:
/// <c>HARTSY_VOICE_LLM_GPU=1 HARTSY_REQUIRE_REAL_WEIGHTS=1 HARTSYINFERENCE_MODELS_DIR=/mnt/model-storage/Models dotnet test tests/HartsyInference.Voice.Tests -c Release --filter "FullyQualifiedName~VoiceSessionQwen3EndToEndTests"</c>.</summary>
[Trait("Category", "GpuIntegration")]
[Trait("Category", "RealWeights")]
[Trait("Category", "Slow")]
public sealed class VoiceSessionQwen3EndToEndTests
{
    private const string GrantEnvVar = "HARTSY_VOICE_LLM_GPU";

    private readonly ITestOutputHelper _output;

    public VoiceSessionQwen3EndToEndTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public async Task ACallerIsAnsweredByQwen3OnTheOtherCard()
    {
        if (Environment.GetEnvironmentVariable(GrantEnvVar) != "1")
        {
            _output.WriteLine($"SKIPPED: set {GrantEnvVar}=1 once the 4090 is granted for this run.");
            return;
        }
        string checkpoint = TestPaths.Llm.Qwen3_4BQ4KM;
        if (!RealWeightGate.Require(_output.WriteLine, [.. GpuVoiceRig.Assets(), checkpoint]))
        {
            return;
        }
        EngineOptions engineOptions = new();
        ToolCalling.Install(engineOptions);
        using InferenceEngine? engine = GpuVoiceRig.OpenAudioEngine(_output.WriteLine, ordinal: 1, engineOptions);
        if (engine is null)
        {
            return;
        }
        VoiceAgentOptions options = GpuVoiceRig.Options(engine, llmDevice: "cuda:0") with { LlmModel = "qwen3" };
        ModelSpec llm = VoiceModelSet.ResolveLlm(options);
        _output.WriteLine($"qwen3 resolves to {llm.LocalPath}");
        Assert.Equal(Path.GetFullPath(checkpoint), Path.GetFullPath(llm.LocalPath ?? ""));
        List<string> log = [];
        Logs.SetLogger((_, message) =>
        {
            lock (log)
            {
                log.Add(message);
            }
        });
        try
        {
            await RunAsync(engine, options, log);
        }
        finally
        {
            Logs.SetLogger(null!);
        }
    }

    private async Task RunAsync(InferenceEngine engine, VoiceAgentOptions options, List<string> log)
    {
        await using VoiceModelSet models = await VoiceModelSet.LoadAsync(engine, options, VoiceAssets.WakeRoot);
        long warmStarted = Stopwatch.GetTimestamp();
        await models.WarmAsync(engine.Text);
        _output.WriteLine($"warm-up (speech on the 3060 and one Qwen3 token on the 4090, in parallel): {Stopwatch.GetElapsedTime(warmStarted).TotalMilliseconds:F0} ms");
        lock (log)
        {
            _output.WriteLine(Assert.Single(log, message => message.StartsWith("[Voice] Warm-up on ", StringComparison.Ordinal)));
        }
        ToolRegistry tools = new ToolRegistry().Add("get_time", () => DateTime.Now.ToString("h:mm tt"), "Tells the current local time.");

        float[] jfk = VoiceAssets.Jfk16k();
        TurnEndpointerRealVadTests.Utterance first = TurnEndpointerRealVadTests.Run(jfk, options).Utterances[0];
        float[] slice = jfk[..(int)Math.Min(jfk.Length, first.Start + first.Length)];
        await using VoiceHarness harness = await VoiceHarness.StartAsync(models, engine.Text, TimeSpan.FromMilliseconds(2), tools);
        harness.Push(slice);
        harness.PushSilence(1.5);
        VoiceTurnMetrics turn = (await harness.TurnCompletedAsync(1, 180)).Metrics!.Value;
        float[] replyAudio = harness.Reader.Samples;
        string reply = harness.Session.Transcript.Single(e => e.Role == TextRole.Assistant).Text;
        string replyHeard = await models.Gpu.RunAsync(VoiceGpuJobKind.Transcribe, () => models.Transcribe(replyAudio), CancellationToken.None);
        string caller = harness.Session.Transcript.Single(e => e.Role == TextRole.User).Text;
        string[] replyWords = [.. VoiceAssets.Words(reply).Distinct()];
        _output.WriteLine(turn.ToLogLine());
        _output.WriteLine($"caller: \"{caller}\" (recall of \"fellow Americans\": {VoiceAssets.Recall(caller, ["fellow", "americans"]):P0})");
        _output.WriteLine($"Qwen3: \"{reply}\"; heard back: \"{replyHeard}\" (recall of the reply's words: {VoiceAssets.Recall(replyHeard, replyWords):P0})");

        Assert.DoesNotContain("<think>", reply, StringComparison.Ordinal);
        Assert.NotNull(turn.LlmTtftMs);
        Assert.NotNull(turn.TotalMs);
        Assert.True(VoiceAssets.Words(replyHeard).Count() >= 3, $"the spoken reply was not intelligible: \"{replyHeard}\"");
    }
}
