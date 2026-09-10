using System.Diagnostics;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Logging;
using HartsyInference.Cuda;
using HartsyInference.Vulkan;
using HartsyInference.Core.Configuration;
using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;
using HartsyInference.BenchmarkRunner.Contracts;
using HartsyInference.BenchmarkRunner.Serialization;

namespace HartsyInference.BenchmarkRunner.Execution;
/// <summary>One model/session per process. Journals survive timeout, crashes, and later resume attempts.</summary>
public static class Worker
{
    /// <summary>Known memory/unsupported errors are explicit; a process kill remains an unexplained crash.</summary>
    public static string ClassifyFailure(Exception error) => error switch
    {
        OutOfMemoryException or OutOfVramException => "oom",
        CudaException { ErrorCode: 2 } => "oom",
        VulkanException { ErrorCode: -1 or -2 } => "oom",
        OperationCanceledException => "cancelled",
        NotSupportedException or UnsupportedModelException => "unsupported",
        _ => "failed",
    };
    public static async Task<int> RunAsync(string root, string cache, string suiteId, string caseId, string selector, int session,
        int attempt, CancellationToken cancel)
    {
        SuiteDefinition suite = Suites.Load(suiteId);
        CaseDefinition definition = suite.Cases.Single(c => c.Id == caseId);
        string relative = $"sessions/{caseId}/{session}/{attempt}";
        string directory = Hashes.SafePath(root, relative);
        Directory.CreateDirectory(directory);
        string journal = Path.Combine(directory, "session.json");
        List<Measurement> measurements = [];
        BenchDiagnostics diagnostics = new(selector);
        SessionRecord record = new()
        {
            CaseId = caseId,
            Session = session,
            Attempt = attempt,
            Status = "running",
            StartedUtc = DateTimeOffset.UtcNow,
            Measurements = []
        };
        BenchJson.Write(journal, record, BenchJson.Default.SessionRecord);
        try
        {
            Assets.Verify(cache, definition.Asset);
            using IDisposable defaults = Hardware.Defaults(out _).Push();
            using InferenceEngine engine = new(selector, new EngineOptions { Diagnostics = diagnostics });
            IBenchmarkAdapter adapter = definition.Adapter switch
            {
                "text" => new TextAdapter(),
                "image" => new ImageAdapter(),
                _ => throw new NotSupportedException("Adapter is not registered."),
            };
            ModelSpec spec = new()
            {
                Requested = definition.Model,
                LocalPath = Assets.PathFor(cache, definition.Asset),
                Modality = adapter.Modality
            };
            for (int step = 0; step < suite.Warmups + definition.Inputs.Length; step++)
            {
                cancel.ThrowIfCancellationRequested();
                string lane = step == 0 ? "first" : step < suite.Warmups ? "warmup" : "warm";
                int input = step < suite.Warmups ? step : step - suite.Warmups;
                string prompt = string.Concat(Enumerable.Repeat(definition.InputPrefix, definition.PrefixRepeats)) + definition.Inputs[input];
                diagnostics.Reset();
                long start = Stopwatch.GetTimestamp();
                BenchmarkOutput generated = await adapter.GenerateAsync(engine, spec, definition, prompt, input, selector, cancel);
                TextResult? text = generated.Text;
                ImageResult? image = generated.Image;
                long end = Stopwatch.GetTimestamp();
                double elapsed = Stopwatch.GetElapsedTime(start, end).TotalMilliseconds;
                string output = relative + $"/{lane}-{input}" + (text is null ? ".png" : ".txt");
                string file = Hashes.SafePath(root, output);
                bool quality;
                string qualityDetail;
                if (text is not null)
                {
                    await File.WriteAllTextAsync(file, text.Text, cancel);
                    quality = !string.IsNullOrWhiteSpace(text.Text) && !text.Text.Contains('�') && text.CompletionTokens > 0 && text
                        .CompletionTokens == diagnostics.Tokens && text.PromptTokens == diagnostics.Prompt && text.Stop is StopReason
                        .Stop or StopReason.Length;
                    qualityDetail = "Nonempty UTF-8 and native token counts; semantic correctness requires output review.";
                }
                else
                {
                    ImageResult result = image!;
                    await File.WriteAllBytesAsync(file, PngEncoder.Encode(result.Rgb, result.Width, result.Height), cancel);
                    double mean = result.Rgb.Average(b => (double)b);
                    double variance = result.Rgb.Average(b => ((double)b - mean) * (b - mean));
                    quality = result.Width == definition.Width && result.Height == definition.Height && result.Rgb.Length == definition
                        .Width * definition.Height * 3 && result.Seed == definition.Seed + input && variance > 1;
                    qualityDetail = "Dimensions, seed, nonconstant RGB; semantic correctness requires output review.";
                }

                measurements.Add(new Measurement { Input = input, Lane = lane, ElapsedMs = elapsed, StopwatchFrequency = Stopwatch
                    .Frequency, StartedTicks = start, CompletedTicks = end, RequestStartedTicks = diagnostics.Start,
                    PrefillTicks = diagnostics.Prefill, TokenTimestamps = diagnostics.TokenTimestamps, FirstTokenMs = diagnostics.FirstMs,
                    DecodeTokensPerSecond = diagnostics.DecodeRate, PromptTokens = text?.PromptTokens ?? 0, CompletionTokens = text?
                    .CompletionTokens ?? 0, StopReason = text?.Stop.ToString() ?? "completed", Output = output, OutputSha256 = Hashes
                    .FileHash(file), QualityPassed = quality, QualityDetail = qualityDetail, HostPeakBytes = Process.GetCurrentProcess()
                    .PeakWorkingSet64 });
                record = record with
                {
                    Measurements = measurements.ToArray(),
                    ActualDevice = diagnostics.Device
                };
                BenchJson.Write(journal, record, BenchJson.Default.SessionRecord);
            }

            record = record with
            {
                Status = measurements.All(m => m.QualityPassed) ? "completed" : "quality-failed",
                NativeLibraries = Hardware.NativeLibraries()
            };
        }
        catch (Exception error)
        {
            Logs.Error("Benchmark worker failed; details are retained only in the local worker log.", error);
            // The process boundary owns failures. Store types, never exception text containing private paths or tokens.
            record = record with
            {
                Status = ClassifyFailure(error),
                Failure = error.GetType().Name
            };
        }

        BenchJson.Write(journal, record, BenchJson.Default.SessionRecord);
        return record.Status == "completed" ? 0 : 1;
    }
}
