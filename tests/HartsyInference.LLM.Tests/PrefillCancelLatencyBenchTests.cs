using System.Diagnostics;
using System.Globalization;
using System.Text;
using HartsyInference.Cuda;
using HartsyInference.LLM.ChatTemplates;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Sampling;
using HartsyInference.ModelAssets.Tokenizers;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.LLM.Tests;

/// <summary>How fast a request cancelled during its prompt prefill lets go of the GPU. Opt-in with
/// <c>HARTSY_PREFILL_CANCEL_BENCH=1</c>; otherwise returns early. Opens <c>CudaBackend(ordinal)</c> with
/// <c>HARTSY_PREFILL_CANCEL_BENCH_CUDA_ORDINAL</c> (default 1, the 3060's engine ordinal on the reference box) and fails
/// unless the device is a 3060; the checkpoint is <c>HARTSY_PREFILL_CANCEL_BENCH_GGUF</c>, default Qwen3-4B Q4_K_M.
///
/// <para>Method: a ~650-token greedy prompt; the full prefill (prefill + first sample, then a device sync) is timed
/// over 2 warm-up + 5 runs. Then, for each cancel point (a fraction of that median), 5 requests are cancelled from
/// another thread at that point and two intervals are taken from the cancel: until <c>Generate</c> throws, and until a
/// <see cref="Core.Backends.IBackend.Sync"/> issued right after the throw returns, which is when the card is free for
/// the next request. Each run also records whether the prompt prefill finished before the stop. Medians are reported;
/// the same file runs against any build, so one build's table is comparable with another's.</para></summary>
[Trait("Category", "GpuIntegration")]
[Trait("Category", "RealWeights")]
public sealed class PrefillCancelLatencyBenchTests
{
    private const string GateEnvVar = "HARTSY_PREFILL_CANCEL_BENCH";
    private const string OrdinalEnvVar = "HARTSY_PREFILL_CANCEL_BENCH_CUDA_ORDINAL";
    private const string CheckpointEnvVar = "HARTSY_PREFILL_CANCEL_BENCH_GGUF";
    private const string RequiredDeviceSubstring = "3060";
    private const int TargetPromptTokens = 650;
    private const int WarmRuns = 2;
    private const int TimedRuns = 5;
    private const string Paragraph =
        "Call notes so far: the caller is phoning about an order placed last week. They want to change the delivery "
        + "address to their office, ask whether the courier can call ahead, and confirm the refund on a returned item "
        + "has been issued. The account shows one open order, one completed return, and a note that the previous agent "
        + "promised a callback that never happened. ";
    private static readonly double[] CancelPoints = [0.1, 0.3, 0.6];

    private readonly ITestOutputHelper _out;

    public PrefillCancelLatencyBenchTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public void CancelDuringThePromptPrefill_TimeToThrowAndToAFreeCard()
    {
        if (Environment.GetEnvironmentVariable(GateEnvVar) != "1")
        {
            _out.WriteLine($"SKIPPED: set {GateEnvVar}=1 to run the prefill cancel-latency probe.");
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
        GenerationRequest request = BuildRequest(model.Tokenizer);

        List<double> full = [];
        int promptTokens = 0;
        for (int i = 0; i < WarmRuns + TimedRuns; i++)
        {
            Stopwatch clock = Stopwatch.StartNew();
            GenerationResult result = pipeline.Generate(request with { MaxTokens = 1 });
            backend.Sync();
            promptTokens = result.PromptTokens;
            if (i >= WarmRuns) full.Add(clock.Elapsed.TotalMilliseconds);
        }
        double fullMs = Median(full);
        _out.WriteLine($"{device}, ordinal {ordinal}; {Path.GetFileName(checkpoint)}; {promptTokens} prompt tokens; "
            + $"full prefill + first sample + sync: median {fullMs:F1} ms ({string.Join(" / ", full.Select(v => v.ToString("F1", CultureInfo.InvariantCulture)))})");

        StringBuilder table = new StringBuilder();
        table.AppendLine($"| cancel at | cancel → throw, median ms | cancel → card free, median ms | prefill finished before the stop |");
        table.AppendLine("|---:|---:|---:|---:|");
        foreach (double point in CancelPoints)
        {
            List<double> toThrow = [];
            List<double> toFree = [];
            int finished = 0;
            for (int i = 0; i < TimedRuns; i++)
            {
                Stop stop = CancelOnce(pipeline, backend, request, point * fullMs);
                toThrow.Add(stop.ThrowMs);
                toFree.Add(stop.FreeMs);
                if (stop.PrefillFinished) finished++;
                _out.WriteLine($"cancel at {point:P0} ({point * fullMs:F1} ms): throw +{stop.ThrowMs:F1} ms, card free +{stop.FreeMs:F1} ms, "
                    + $"prefill finished {stop.PrefillFinished}");
            }
            table.AppendLine($"| {point:P0} of {fullMs:F0} ms | {Median(toThrow):F1} | {Median(toFree):F1} | {finished}/{TimedRuns} |");
        }
        _out.WriteLine(table.ToString());
    }

    /// <summary>Runs <paramref name="request"/> on its own thread and cancels it <paramref name="afterMs"/> after it started.</summary>
    private static Stop CancelOnce(TextGenerationPipeline pipeline, CudaBackend backend, GenerationRequest request, double afterMs)
    {
        using CancellationTokenSource cancel = new CancellationTokenSource();
        Stopwatch clock = new Stopwatch();
        double started = -1;
        double thrown = double.NaN;
        double free = double.NaN;
        bool prefillFinished = false;
        Exception? unexpected = null;
        GenerationRequest watched = request with { OnPrefillCompleted = _ => prefillFinished = true };
        Thread worker = new Thread(() =>
        {
            try
            {
                Volatile.Write(ref started, clock.Elapsed.TotalMilliseconds);
                pipeline.Generate(watched, onToken: null, cancel.Token);
            }
            catch (OperationCanceledException)
            {
                thrown = clock.Elapsed.TotalMilliseconds;
                backend.Sync();
                free = clock.Elapsed.TotalMilliseconds;
            }
            catch (Exception ex)
            {
                unexpected = ex;
            }
        });
        clock.Start();
        worker.Start();
        while (Volatile.Read(ref started) < 0) Thread.SpinWait(10);
        double cancelAt = Volatile.Read(ref started) + afterMs;
        while (clock.Elapsed.TotalMilliseconds < cancelAt) Thread.SpinWait(10);
        cancelAt = clock.Elapsed.TotalMilliseconds;
        cancel.Cancel();
        worker.Join();
        Assert.Null(unexpected);
        Assert.False(double.IsNaN(thrown), "the request finished without observing the cancel");
        return new Stop(thrown - cancelAt, free - cancelAt, prefillFinished);
    }

    private static GenerationRequest BuildRequest(ILlmTokenizer tokenizer)
    {
        StringBuilder sb = new StringBuilder();
        for (int copies = 0; copies < 60 && tokenizer.Encode(sb.ToString(), addSpecial: false).Length < TargetPromptTokens; copies++)
        {
            sb.Append(Paragraph);
        }
        return new GenerationRequest
        {
            Messages = [ChatMessage.System("You are a phone agent. Answer briefly and plainly."), ChatMessage.User(sb + "\nSummarize what the caller needs.")],
            EnableThinking = false,
            MaxTokens = 32,
            Sampling = SamplingOptions.GreedyPreset,
            GraphDecode = false,
            SpeculativeDecode = false,
        };
    }

    private static double Median(List<double> values)
    {
        List<double> sorted = values.OrderBy(v => v).ToList();
        int mid = sorted.Count / 2;
        return sorted.Count % 2 == 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
    }

    private readonly record struct Stop(double ThrowMs, double FreeMs, bool PrefillFinished);
}
