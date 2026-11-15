using System.Diagnostics;
using System.Reflection;
using System.Security.Cryptography;
using System.Text.Json;
using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Configuration;
using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;

namespace HartsyInference.GenCapture;

/// <summary>Exercises frozen ordinary Engine requests; scoring and forced-token paths are never primary measurements.</summary>
internal static class AttentionRequestEvidence
{
    public static async Task<int> RunAsync(Dictionary<string, string> options)
    {
        string manifestPath = options["workload"];
        using JsonDocument document = JsonDocument.Parse(File.ReadAllBytes(manifestPath));
        JsonElement manifest = document.RootElement;
        string family = manifest.GetProperty("family").GetString()!;
        if (family is not ("qwen" or "krea")) throw new InvalidDataException("Unknown primary workload.");
        string model = manifest.GetProperty("model_path").GetString()!;
        string promptPath = manifest.GetProperty("prompt_path").GetString()!;
        Verify(model, manifest.GetProperty("model_sha256").GetString()!);
        Verify(promptPath, manifest.GetProperty("prompt_sha256").GetString()!);
        string prompt = File.ReadAllText(promptPath);
        string selector = options.GetValueOrDefault("backend", "cuda:0");
        string lane = options.GetValueOrDefault("lane", "paged");
        if (lane is not ("paged" or "batched" or "dense" or "graph")) throw new ArgumentException("Unknown request lane.");
        int batchSize = lane == "batched" ? 2 : 1;
        bool paged = family == "qwen" && (lane is "paged" or "batched");
        bool graph = family == "qwen" && options.GetValueOrDefault("lane", "paged") == "graph";
        Dictionary<string, string> auxiliary = new(StringComparer.Ordinal);
        if (manifest.TryGetProperty("aux", out JsonElement aux))
            foreach (JsonProperty item in aux.EnumerateObject())
            {
                string path = item.Value.GetProperty("path").GetString()!;
                Verify(path, item.Value.GetProperty("sha256").GetString()!);
                auxiliary.Add(item.Name, path);
            }
        ModelSpec spec = new()
        {
            Requested = manifest.GetProperty("model_id").GetString()!, LocalPath = model,
            Modality = family == "qwen" ? Modality.Text : Modality.Image, Aux = auxiliary,
        };
        KnobProfile profile = HartsyInference.Benchmarks.Shared.BenchmarkKnobProfiles.Capture(false,
            out SortedDictionary<string, string> settings).With(EngineKnobs.ContinuousBatching, paged);
        settings[EngineKnobs.ContinuousBatching.Id] = paged.ToString();
        using (profile.Push())
        {
            using InferenceEngine engine = new(selector, new EngineOptions { KvPoolBytesBudget = 4L * 1024 * 1024 * 1024 });
            TextRequest text = new()
            {
                Messages = [new TextMessage { Role = TextRole.User, Content = prompt }],
                Temperature = 0, Greedy = true, TopP = 1, TopK = 0, MinP = 0, RepetitionPenalty = 1,
                MaxTokens = 128, Seed = 42, EnableThinking = false, Device = selector, GraphDecode = graph,
                SpeculativeDecode = false, AlwaysFreeMemory = false,
            };
            ImageRequest image = new()
            {
                Prompt = prompt, Width = 1024, Height = 1024, Steps = 8, Seed = 42, Batch = 1, CfgScale = 1,
            };
            IDisposable? evidence = null;
            if (options.ContainsKey("observe"))
            {
                Type? observer = typeof(AttentionCall).Assembly.GetType("HartsyInference.Core.Backends.Attention.AttentionEvidence");
                evidence = (IDisposable)(observer?.GetMethod("Begin")?.Invoke(null, [262144])
                    ?? throw new NotSupportedException("The loaded pristine assembly has no telemetry."));
            }
            try
            {
                for (int i = 0; i < 3; i++)
                {
                    if (family == "qwen")
                    {
                        if (batchSize == 1) Check(await engine.Text.GenerateAsync(spec, text));
                        else foreach (TimedRequest result in await BatchAsync(engine, spec, text)) Check(result.Result);
                    }
                    else _ = await engine.Images.GenerateAsync(spec, image);
                }
                long measuredObservationStart = evidence is null ? 0
                    : (long)evidence.GetType().GetProperty("Attempted")!.GetValue(evidence)!;
                long start = Stopwatch.GetTimestamp();
                TextResult? completion = family == "qwen" && batchSize == 1 ? await engine.Text.GenerateAsync(spec, text) : null;
                TimedRequest[]? batched = family == "qwen" && batchSize > 1 ? await BatchAsync(engine, spec, text) : null;
                ImageResult? pixels = family == "krea" ? await engine.Images.GenerateAsync(spec, image) : null;
                double milliseconds = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
                if (completion is not null) Check(completion);
                if (batched is not null) foreach (TimedRequest result in batched) Check(result.Result);
                if (pixels is not null && (pixels.Width != 1024 || pixels.Height != 1024 || pixels.Seed != 42))
                    throw new InvalidDataException("The image workload changed dimensions or seed.");
                if (pixels is not null && (pixels.Recipe?.FamilyId != "krea2" || pixels.Recipe.VariantId != "turbo"))
                    throw new InvalidDataException("The frozen image workload must execute the Krea2 Turbo recipe variant.");
                string? imageCapture = null;
                if (pixels is not null)
                {
                    imageCapture = Path.ChangeExtension(options["out"], ".png");
                    using FileStream capture = new(imageCapture, FileMode.CreateNew);
                    capture.Write(PngEncoder.Encode(pixels.Rgb, pixels.Width, pixels.Height));
                }
                if (evidence is not null)
                {
                    evidence.Dispose();
                    using FileStream observations = new(options["observe"], FileMode.CreateNew);
                    evidence.GetType().GetMethod("WriteJson")!.Invoke(evidence, [observations]);
                }
                using FileStream output = new(options["out"], FileMode.CreateNew);
                using Utf8JsonWriter writer = new(output, new JsonWriterOptions { Indented = true });
                writer.WriteStartObject();
                writer.WriteNumber("schema", 1);
                writer.WriteString("family", family);
                writer.WriteString("backend", selector);
                writer.WriteString("workload_sha256", Hash(manifestPath));
                writer.WriteString("model_sha256", Hash(model));
                writer.WriteString("core_sha256", Hash(typeof(AttentionCall).Assembly.Location));
                writer.WriteString("engine_sha256", Hash(typeof(InferenceEngine).Assembly.Location));
                writer.WriteStartObject("resolved_settings");
                foreach ((string key, string value) in settings) writer.WriteString(key, value);
                writer.WriteEndObject();
                writer.WriteString("lane", family == "krea" ? "image" : lane);
                writer.WriteString("telemetry", evidence is null ? "disabled" : "enabled");
                writer.WriteNumber("measured_observation_start", measuredObservationStart);
                writer.WriteString("observations_include", "model loading, two request warmups, then the measured request");
                writer.WriteNumber("milliseconds", milliseconds);
                writer.WriteNumber("request_warmups", 2);
                if (completion is not null)
                {
                    writer.WriteNumber("prompt_tokens", completion.PromptTokens);
                    writer.WriteNumber("completion_tokens", completion.CompletionTokens);
                    writer.WriteString("text", completion.Text);
                    writer.WriteNumber("prefill_ms", completion.PrefillMilliseconds);
                    writer.WriteNumber("decode_ms", completion.DecodeMilliseconds);
                }
                if (pixels is not null)
                {
                    writer.WriteString("rgb_sha256", Convert.ToHexString(SHA256.HashData(pixels.Rgb)));
                    writer.WriteString("png_path", imageCapture);
                    writer.WriteString("png_sha256", Hash(imageCapture!));
                    writer.WriteString("recipe", pixels.Recipe?.ToString());
                    writer.WriteNumber("actual_seed", pixels.Seed);
                    writer.WriteStartObject("resolved_image_settings");
                    foreach ((string key, HartsyInference.Engine.Planning.ResolvedParameter value) in pixels.Provenance)
                        writer.WriteString(key, value.ToString());
                    writer.WriteEndObject();
                }
                if (batched is not null)
                {
                    writer.WriteNumber("batch_size", batched.Length);
                    writer.WriteNumber("tokens_per_second", 128 * batched.Length / milliseconds * 1000);
                    writer.WriteStartArray("requests");
                    foreach (TimedRequest sample in batched)
                    {
                        TextResult result = sample.Result;
                        writer.WriteStartObject();
                        writer.WriteNumber("milliseconds", sample.Milliseconds);
                        writer.WriteNumber("prompt_tokens", result.PromptTokens);
                        writer.WriteNumber("completion_tokens", result.CompletionTokens);
                        writer.WriteString("text", result.Text);
                        writer.WriteNumber("prefill_ms", result.PrefillMilliseconds);
                        writer.WriteNumber("decode_ms", result.DecodeMilliseconds);
                        writer.WriteEndObject();
                    }
                    writer.WriteEndArray();
                }
                writer.WriteEndObject();
                return 0;
            }
            finally { evidence?.Dispose(); }
        }
    }

    private static Task<TimedRequest[]> BatchAsync(InferenceEngine engine, ModelSpec spec, TextRequest request) =>
        Task.WhenAll(TimedAsync(engine, spec, request), TimedAsync(engine, spec, request));

    private static async Task<TimedRequest> TimedAsync(InferenceEngine engine, ModelSpec spec, TextRequest request)
    {
        long start = Stopwatch.GetTimestamp();
        TextResult result = await engine.Text.GenerateAsync(spec, request);
        return new(result, Stopwatch.GetElapsedTime(start).TotalMilliseconds);
    }

    private readonly record struct TimedRequest(TextResult Result, double Milliseconds);

    private static void Check(TextResult result)
    {
        if (result.PromptTokens != 8192 || result.CompletionTokens != 128 || result.Stop != StopReason.Length)
            throw new InvalidDataException("Qwen primary requires exactly 8192 prompt and 128 generated tokens ending at the limit.");
    }

    private static void Verify(string path, string expected)
    {
        if (!File.Exists(path) || Hash(path) != expected) throw new InvalidDataException("Missing or changed workload asset: " + path);
    }

    private static string Hash(string path)
    {
        using FileStream input = File.OpenRead(path);
        return Convert.ToHexString(SHA256.HashData(input)).ToLowerInvariant();
    }
}
