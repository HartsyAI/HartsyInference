using System.Diagnostics;
using System.Globalization;
using System.Text;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Cuda;
using HartsyInference.LLM.ChatTemplates;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Sampling;
using HartsyInference.ModelAssets.Tokenizers;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.LLM.Tests;

/// <summary>Phase 1 voice-turn LLM probe: Qwen3-4B Q4_K_M in-process on the 4090, the numbers the plan's
/// "TTFT ≤ 150 ms and ≥ 60 tok/s with tools on (graph/spec decode off)" gate is evaluated against. Opt-in with
/// <c>HARTSY_VOICE_BENCH_LLM=1</c>; otherwise returns early. Opens <c>CudaBackend(0)</c> and FAILS unless the
/// device name contains "4090" (engine ordinal 0 is the fastest card on the reference box; nvidia-smi index 1).
///
/// <para>Request shape: a ~500-token prompt (a call-notes paragraph repeated until the model's own tokenizer
/// counts at least 500 ordinary tokens; the templated prompt length is reported from
/// <see cref="GenerationResult.PromptTokens"/>), <c>EnableThinking = false</c>, greedy, 64 max tokens, 2 warm-up +
/// 5 timed runs, twice: (1) "tools on" — a hang-up tool definition rendered into the system turn, the same
/// <c>JsonModeSentinel = "&lt;tool_call&gt;"</c> grammar <c>TextService</c> arms when <c>Tools</c> are present, and graph /
/// speculative decode forced OFF (the grammar makes them ineligible anyway; forcing keeps the row honest if that
/// rule ever changes); (2) "no tools" — no tool definition, no grammar, graph / speculative decode left at the
/// engine knob defaults, which are printed. The tool JSON is written into the system-prompt TEXT and attached as
/// <see cref="ChatMessage.Tools"/>, which the template ignores: that is the shape the 2026-09-30 gate row was
/// measured with (631 templated tokens). <c>main</c> now renders <see cref="GenerationRequest.Tools"/> through the
/// Jinja template natively (#193); this probe deliberately keeps the measured shape so the row stays comparable,
/// and moving it to <see cref="GenerationRequest.Tools"/> is the next probe revision.</para>
///
/// <para>Paths: the checkpoint comes from <see cref="TestPaths.Llm.Qwen3_4BQ4KM"/>, whose default root is
/// <c>&lt;repo&gt;/Models</c>; on the reference box run with <c>HARTSYINFERENCE_MODELS_DIR=/mnt/model-storage/Models</c>
/// (the 2026-09-30 run needed it).</para>
///
/// <para>Timing: <c>t_prefill</c> is <see cref="GenerationRequest.OnPrefillCompleted"/>, which fires AFTER the first
/// sample (prefill + first-token sampling, not prefill alone — the pipeline has no earlier hook), and is
/// effectively coincident with the first <c>onToken</c>; TTFT is first <c>onToken</c> minus request start; decode
/// rate is <c>(n − 1) / (t_last − t_first)</c> over the streamed tokens. Median / p95 (linear interpolation) / min.
/// Tables go to the test output and, when <c>HARTSY_VOICE_BENCH_OUT</c> names a file, are appended as markdown.</para></summary>
[Trait("Category", "GpuIntegration")]
[Trait("Category", "RealWeights")]
public sealed class VoiceLlmTurnBenchTests
{
    private const string GateEnvVar = "HARTSY_VOICE_BENCH_LLM";
    private const string OutEnvVar = "HARTSY_VOICE_BENCH_OUT";
    private const string RequiredDeviceSubstring = "4090";
    private const int Ordinal = 0;
    private const int WarmRuns = 2;
    private const int TimedRuns = 5;
    private const int MaxTokens = 64;
    private const int TargetPromptTokens = 500;
    private const string ToolSentinel = "<tool_call>";
    private const string ContextParagraph =
        "Call notes so far: the caller is phoning about an order placed last week. They want to change the delivery "
        + "address to their office, ask whether the courier can call ahead, and confirm the refund on a returned item "
        + "has been issued. The account shows one open order, one completed return, and a note that the previous agent "
        + "promised a callback that never happened. Company policy: address changes are allowed until the parcel is "
        + "scanned at the depot, refunds take three to five business days, and any promise of a callback must be logged. ";
    private const string Question = "Summarize what the caller needs in two short sentences.";
    private const string SystemPrompt = "You are a phone agent. Answer briefly and plainly.";
    private const string HangUpToolJson =
        "{\"type\":\"function\",\"function\":{\"name\":\"hang_up\",\"description\":\"End the call.\","
        + "\"parameters\":{\"type\":\"object\",\"properties\":{\"reason\":{\"type\":\"string\"}},\"required\":[\"reason\"]}}}";

    private readonly ITestOutputHelper _out;

    public VoiceLlmTurnBenchTests(ITestOutputHelper output) => _out = output;

    /// <summary>Qwen3-4B TTFT and decode rate, tools on (grammar armed, graph/spec off) and tools off.</summary>
    [Fact]
    public void Qwen3_4B_Ttft_And_Decode_WithAndWithoutTools_4090()
    {
        if (Environment.GetEnvironmentVariable(GateEnvVar) != "1")
        {
            _out.WriteLine($"SKIPPED: set {GateEnvVar}=1 to run the voice-turn LLM probe.");
            return;
        }
        string checkpoint = TestPaths.Llm.Qwen3_4BQ4KM;
        if (!RealWeightGate.Require(_out.WriteLine, checkpoint)) return;
        Assert.True(CudaContext.IsAvailable(), $"CUDA unavailable: {CudaContext.LastUnavailableReason}");
        string? ptx = BackendGate.KernelDir("Ptx", "HartsyInference.Cuda");
        Assert.False(ptx is null, "no compiled PTX directory beside the tests or in the repo");

        using CudaBackend backend = new CudaBackend(Ordinal, ptx);
        string device = backend.Capabilities.DeviceName;
        _out.WriteLine($"CUDA ordinal {Ordinal}: {device}");
        Assert.True(device.Contains(RequiredDeviceSubstring, StringComparison.Ordinal),
            $"ordinal {Ordinal} is '{device}', not a {RequiredDeviceSubstring}.");

        Stopwatch load = Stopwatch.StartNew();
        using GgufLanguageModel model = GgufLanguageModel.Load(checkpoint);
        TextGenerationPipeline pipeline = new TextGenerationPipeline(model.Transformer, model.Tokenizer, backend, model.Template);
        _out.WriteLine($"{Path.GetFileName(checkpoint)} loaded in {load.Elapsed.TotalSeconds:F1}s (arch {model.Architecture}, "
            + $"template {model.Template.GetType().Name}); graphDecode knob={EngineKnobs.GraphDecode.Value} specDecode knob={EngineKnobs.SpecDecode.Value}");

        (string prompt, int promptTokens) = BuildPrompt(model.Tokenizer);
        _out.WriteLine($"user prompt: {promptTokens} ordinary tokens (target ≥ {TargetPromptTokens})");

        StringBuilder table = new StringBuilder();
        table.AppendLine("### LLM probe — Qwen3-4B Q4_K_M (4090, in-process, thinking off, greedy, 64 max tokens)");
        table.AppendLine();
        table.AppendLine("| Variant | prompt tokens (templated) | TTFT median ms | p95 | min | prefill+first-sample median ms | decode tok/s median | p95 | min | tokens | stopped on EOS | reply head |");
        table.AppendLine("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|");

        foreach (bool tools in new[] { true, false })
        {
            string variant = tools ? "tools on (sentinel grammar, graph/spec OFF)" : "no tools (knob defaults)";
            GenerationRequest request = BuildRequest(prompt, tools);
            List<double> ttft = new List<double>(TimedRuns);
            List<double> prefill = new List<double>(TimedRuns);
            List<double> decode = new List<double>(TimedRuns);
            GenerationResult? last = null;
            int lastPromptTokens = 0;
            for (int i = 0; i < WarmRuns + TimedRuns; i++)
            {
                Run run = Execute(pipeline, request);
                if (i >= WarmRuns)
                {
                    ttft.Add(run.Ttft);
                    prefill.Add(run.PrefillAndFirstSample);
                    decode.Add(run.DecodeTokensPerSecond);
                }
                last = run.Result;
                lastPromptTokens = run.Result.PromptTokens;
                _out.WriteLine($"{variant} {(i < WarmRuns ? "warm" : "timed")} {i + 1}: ttft {run.Ttft * 1000:F1} ms, prefill+sample {run.PrefillAndFirstSample * 1000:F1} ms, "
                    + $"decode {run.DecodeTokensPerSecond:F1} tok/s over {run.Result.TokenIds.Count} tokens, prompt {run.Result.PromptTokens} tok");
            }
            Stats t = Stats.Of(ttft);
            Stats p = Stats.Of(prefill);
            Stats d = Stats.Of(decode);
            table.AppendLine($"| {variant} | {lastPromptTokens} | {t.Median * 1000:F1} | {t.P95 * 1000:F1} | {t.Min * 1000:F1} | {p.Median * 1000:F1} | "
                + $"{d.Median:F1} | {d.P95:F1} | {d.Min:F1} | {last!.TokenIds.Count} | {last.StoppedOnStopToken} | {Cell(last.Text)} |");
        }
        Emit(table);
    }

    private static (string Prompt, int Tokens) BuildPrompt(ILlmTokenizer tokenizer)
    {
        StringBuilder sb = new StringBuilder();
        int tokens = 0;
        for (int copies = 0; copies < 40; copies++)
        {
            sb.Append(ContextParagraph);
            tokens = tokenizer.Encode(sb.ToString() + "\n" + Question, addSpecial: false).Length;
            if (tokens >= TargetPromptTokens)
            {
                break;
            }
        }
        return (sb.ToString() + "\n" + Question, tokens);
    }

    private static GenerationRequest BuildRequest(string prompt, bool tools)
    {
        List<ChatMessage> messages = new List<ChatMessage>(2);
        if (tools)
        {
            messages.Add(ChatMessage.System(SystemPrompt + "\n\nTools available (call with <tool_call>{...}</tool_call>):\n" + HangUpToolJson)
                with { Tools = [ToolSpec.FromJson(HangUpToolJson)] });
        }
        else
        {
            messages.Add(ChatMessage.System(SystemPrompt));
        }
        messages.Add(ChatMessage.User(prompt));
        SamplingOptions sampling = tools ? SamplingOptions.GreedyPreset with { JsonModeSentinel = ToolSentinel } : SamplingOptions.GreedyPreset;
        return new GenerationRequest
        {
            Messages = messages,
            EnableThinking = false,
            MaxTokens = MaxTokens,
            Sampling = sampling,
            GraphDecode = tools ? false : null,
            SpeculativeDecode = tools ? false : null,
        };
    }

    private static Run Execute(TextGenerationPipeline pipeline, GenerationRequest request)
    {
        Stopwatch clock = Stopwatch.StartNew();
        double prefillDone = double.NaN;
        double first = double.NaN;
        double last = double.NaN;
        int count = 0;
        GenerationRequest timed = request with { OnPrefillCompleted = _ => prefillDone = clock.Elapsed.TotalSeconds };
        GenerationResult result = pipeline.Generate(timed, _ =>
        {
            double now = clock.Elapsed.TotalSeconds;
            if (count == 0)
            {
                first = now;
            }
            last = now;
            count++;
        });
        Assert.True(count >= 2, $"generation produced {count} token(s); need at least 2 for a decode rate");
        double decodeRate = (count - 1) / (last - first);
        return new Run(result, first, double.IsNaN(prefillDone) ? first : prefillDone, decodeRate);
    }

    private static string Cell(string text)
    {
        string flat = text.Trim().Replace("|", "\\|").Replace('\n', ' ');
        return flat.Length > 120 ? flat[..120] + "…" : flat;
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

    private readonly record struct Run(GenerationResult Result, double Ttft, double PrefillAndFirstSample, double DecodeTokensPerSecond);

    private readonly record struct Stats(double Median, double P95, double Min)
    {
        public static Stats Of(List<double> samples)
        {
            List<double> sorted = samples.OrderBy(s => s).ToList();
            return new Stats(Percentile(sorted, 0.5), Percentile(sorted, 0.95), sorted[0]);
        }

        private static double Percentile(List<double> sorted, double p)
        {
            if (sorted.Count == 1)
            {
                return sorted[0];
            }
            double position = p * (sorted.Count - 1);
            int low = (int)Math.Floor(position);
            int high = Math.Min(low + 1, sorted.Count - 1);
            return sorted[low] + (sorted[high] - sorted[low]) * (position - low);
        }
    }
}
