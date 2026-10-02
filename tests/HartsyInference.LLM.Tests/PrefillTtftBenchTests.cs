using System.Diagnostics;
using System.Globalization;
using System.Text;
using HartsyInference.Cuda;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Sampling;
using HartsyInference.ModelAssets.Tokenizers;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.LLM.Tests;

/// <summary>Time to first token for the prefills a voice turn makes, with a cancellable token as the voice session
/// passes one. Opt-in with <c>HARTSY_PREFILL_TTFT_BENCH=1</c>; otherwise returns early. Opens
/// <c>CudaBackend(ordinal)</c> with <c>HARTSY_PREFILL_TTFT_BENCH_CUDA_ORDINAL</c> (default 1, the 3060's engine ordinal on
/// the reference box) and fails unless the device is a 3060; the checkpoint is <c>HARTSY_PREFILL_TTFT_BENCH_GGUF</c>,
/// default Qwen3-4B Q4_K_M.
///
/// <para>Cases: a suffix-only prefill of 20, 40 and 60 new tokens behind a retained ~600-token conversation prefix (the
/// voice session's prefix-cache reuse; each run's suffix differs from the last, so exactly the suffix is prefilled),
/// and a whole 650-token prompt with no reuse. Greedy, one generated token, so a run is the prefill, the head and
/// the first sample; timed from the call to the first token callback. 3 warm-up + 15 timed runs per case, median and
/// p95. The same file runs against any build, so one build's table is comparable with another's.</para></summary>
[Trait("Category", "GpuIntegration")]
[Trait("Category", "RealWeights")]
public sealed class PrefillTtftBenchTests
{
    private const string GateEnvVar = "HARTSY_PREFILL_TTFT_BENCH";
    private const string OrdinalEnvVar = "HARTSY_PREFILL_TTFT_BENCH_CUDA_ORDINAL";
    private const string CheckpointEnvVar = "HARTSY_PREFILL_TTFT_BENCH_GGUF";
    private const string RequiredDeviceSubstring = "3060";
    private const int PrefixTokens = 600;
    private const int FullPromptTokens = 650;
    private const int WarmRuns = 3;
    private const int TimedRuns = 15;
    private static readonly int[] SuffixTokens = [20, 40, 60];
    private const string Text =
        "The caller wants to move the delivery to their office on Thursday, asks whether the courier can phone ahead, "
        + "and wants to know when the refund for the returned jacket will reach their card. The account shows one open "
        + "order, one completed return and a callback that was promised but never logged. Address changes are allowed "
        + "until the parcel is scanned at the depot, refunds take three to five business days, and every promised "
        + "callback must be written into the notes before the call ends. ";

    private readonly ITestOutputHelper _out;

    public PrefillTtftBenchTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public void VoiceTurnPrefills_TimeToFirstToken_WithACancellableToken()
    {
        if (Environment.GetEnvironmentVariable(GateEnvVar) != "1")
        {
            _out.WriteLine($"SKIPPED: set {GateEnvVar}=1 to run the prefill TTFT bench.");
            return;
        }
        string checkpoint = Environment.GetEnvironmentVariable(CheckpointEnvVar) is { Length: > 0 } path ? path : TestPaths.Llm.Qwen3_4BQ4KM;
        if (!RealWeightGate.Require(_out.WriteLine, checkpoint)) return;
        Assert.True(CudaContext.IsAvailable(), $"CUDA unavailable: {CudaContext.LastUnavailableReason}");
        string? ptx = BackendGate.KernelDir("Ptx", "HartsyInference.Cuda");
        Assert.False(ptx is null, "no compiled PTX directory beside the tests or in the repo");
        string? ordinalText = Environment.GetEnvironmentVariable(OrdinalEnvVar);
        int ordinal = string.IsNullOrEmpty(ordinalText) ? 1 : int.Parse(ordinalText, CultureInfo.InvariantCulture);

        using CudaBackend backend = new CudaBackend(ordinal, ptx);
        string device = backend.Capabilities.DeviceName;
        Assert.True(device.Contains(RequiredDeviceSubstring, StringComparison.Ordinal), $"ordinal {ordinal} is '{device}', not a {RequiredDeviceSubstring}.");
        using GgufLanguageModel model = GgufLanguageModel.Load(checkpoint);
        TextGenerationPipeline pipeline = new TextGenerationPipeline(model.Transformer, model.Tokenizer, backend, model.Template);
        int[] corpus = Corpus(model.Tokenizer, FullPromptTokens + PrefixTokens + 64 * (WarmRuns + TimedRuns));
        using CancellationTokenSource live = new CancellationTokenSource();
        _out.WriteLine($"{device}, ordinal {ordinal}; {Path.GetFileName(checkpoint)}; token cancellable: {live.Token.CanBeCanceled}");

        StringBuilder table = new StringBuilder();
        table.AppendLine("| case | prefilled tokens | TTFT median ms | p95 ms | min ms |");
        table.AppendLine("|---|---:|---:|---:|---:|");
        int[] prefix = corpus[..PrefixTokens];
        int[] markers = corpus[..3];
        Assert.Equal(3, markers.Distinct().Count());
        foreach (int suffix in SuffixTokens)
        {
            using RetainedSequence retained = new RetainedSequence();
            GenerationResult primed = pipeline.Generate(Request(prefix), retained, onToken: null, live.Token);
            int afterPrefix = primed.TokenIds.Count > 0 ? primed.TokenIds[0] : -1;
            List<double> times = [];
            for (int run = 0; run < WarmRuns + TimedRuns; run++)
            {
                // The suffix opens with a token unlike the one the retained sequence holds after the prefix, so exactly
                // the suffix is prefilled every run.
                int start = PrefixTokens + run * 64;
                int[] tail = corpus[start..(start + suffix)];
                tail[0] = markers.First(m => m != afterPrefix);
                afterPrefix = tail[0];
                (double ms, int reused) = Ttft(pipeline, Request([.. prefix, .. tail]), retained, live.Token);
                Assert.Equal(PrefixTokens, reused);
                if (run >= WarmRuns) times.Add(ms);
            }
            table.AppendLine(Row($"suffix after a {PrefixTokens}-token retained prefix", suffix, times));
        }
        List<double> full = [];
        for (int run = 0; run < WarmRuns + TimedRuns; run++)
        {
            int start = run % 8;
            (double ms, _) = Ttft(pipeline, Request(corpus[start..(start + FullPromptTokens)]), null, live.Token);
            if (run >= WarmRuns) full.Add(ms);
        }
        table.AppendLine(Row("whole prompt, no reuse", FullPromptTokens, full));
        _out.WriteLine(table.ToString());
    }

    private static GenerationRequest Request(int[] prompt) => new GenerationRequest
    {
        RawTokenIds = prompt,
        MaxTokens = 1,
        Sampling = SamplingOptions.GreedyPreset,
        GraphDecode = false,
        SpeculativeDecode = false,
        PrefixCacheCapacityHint = 1024,
    };

    private static (double Ms, int Reused) Ttft(TextGenerationPipeline pipeline, GenerationRequest request,
        RetainedSequence? retained, CancellationToken cancel)
    {
        Stopwatch clock = Stopwatch.StartNew();
        double first = double.NaN;
        GenerationResult result = pipeline.Generate(request, retained, id =>
        {
            if (double.IsNaN(first)) first = clock.Elapsed.TotalMilliseconds;
        }, cancel);
        Assert.False(double.IsNaN(first), "no token was generated");
        return (first, result.ReusedPromptTokens);
    }

    /// <summary>Real token ids from the model's own tokenizer, enough of them for every case.</summary>
    private static int[] Corpus(ILlmTokenizer tokenizer, int tokens)
    {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < 60; i++)
        {
            sb.Append(Text).Append("Note ").Append(i.ToString(CultureInfo.InvariantCulture)).Append(". ");
        }
        int[] ids = tokenizer.Encode(sb.ToString(), addSpecial: false);
        Assert.True(ids.Length >= tokens, $"corpus has {ids.Length} tokens, need {tokens}");
        return ids;
    }

    private static string Row(string name, int tokens, List<double> times)
    {
        List<double> sorted = times.OrderBy(t => t).ToList();
        double median = sorted.Count % 2 == 1 ? sorted[sorted.Count / 2] : (sorted[sorted.Count / 2 - 1] + sorted[sorted.Count / 2]) / 2;
        double position = 0.95 * (sorted.Count - 1);
        int low = (int)Math.Floor(position);
        double p95 = sorted[low] + (sorted[Math.Min(low + 1, sorted.Count - 1)] - sorted[low]) * (position - low);
        return $"| {name} | {tokens} | {median:F2} | {p95:F2} | {sorted[0]:F2} |";
    }
}
