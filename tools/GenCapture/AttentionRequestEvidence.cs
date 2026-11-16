using System.Diagnostics;
using System.Globalization;
using System.Reflection;
using System.Security.Cryptography;
using System.Text.Json;
using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Configuration;
using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;

namespace HartsyInference.GenCapture;

/// <summary>Exercises frozen ordinary Engine requests; scoring and forced-token paths are never primary measurements.
/// Diagnostic controls are opt-in. With none given, the run is the primary workload: 8192 prompt tokens, 128 generated
/// tokens, a 4 GiB KV pool and the registry-default numerics. <c>--kv-pool-bytes</c> and <c>--expect-prompt-tokens</c> change
/// the pool and the prompt length only for a separately named configuration. <c>--sdpa-no-tf32</c> and <c>--no-tf32</c>
/// pin one TF32 control each and record it separately. <c>--require-batched-decode</c> fails the run unless the measured
/// window contains a decode step with two active sequences.</summary>
internal static class AttentionRequestEvidence
{
    private const long PrimaryPoolBytes = 4L * 1024 * 1024 * 1024;
    private const int PrimaryPromptTokens = 8192;
    private const int GeneratedTokens = 128;

    public static async Task<int> RunAsync(Dictionary<string, string> options)
    {
        byte[] protocolBytes = File.ReadAllBytes(Path.Combine(AppContext.BaseDirectory, "attention-protocol.json"));
        using JsonDocument protocol = JsonDocument.Parse(protocolBytes);
        int requestWarmups = protocol.RootElement.GetProperty("request_warmups").GetInt32();
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(requestWarmups);
        string protocolHash = Convert.ToHexString(SHA256.HashData(protocolBytes)).ToLowerInvariant();
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
        long poolBytes = ParseLong(options, "kv-pool-bytes", PrimaryPoolBytes);
        int expectPrompt = checked((int)ParseLong(options, "expect-prompt-tokens", PrimaryPromptTokens));
        bool sdpaNoTf32 = ParseFlag(options, "sdpa-no-tf32");
        bool noTf32 = ParseFlag(options, "no-tf32");
        bool requireBatchedDecode = ParseFlag(options, "require-batched-decode");
        if (poolBytes <= 0 || expectPrompt <= 0) throw new ArgumentOutOfRangeException(nameof(options), "Pool bytes and prompt tokens must be positive.");
        if (requireBatchedDecode && (lane != "batched" || !options.ContainsKey("observe")))
            throw new ArgumentException("--require-batched-decode needs --lane batched and --observe.");
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
        if (sdpaNoTf32)
        {
            profile = profile.With(EngineKnobs.SdpaNoTf32, true);
            settings[EngineKnobs.SdpaNoTf32.Id] = true.ToString();
        }
        if (noTf32)
        {
            profile = profile.With(EngineKnobs.NoTf32, true);
            settings[EngineKnobs.NoTf32.Id] = true.ToString();
        }
        using (profile.Push())
        {
            using InferenceEngine engine = new(selector, new EngineOptions { KvPoolBytesBudget = poolBytes });
            TextRequest text = new()
            {
                Messages = [new TextMessage { Role = TextRole.User, Content = prompt }],
                Temperature = 0, Greedy = true, TopP = 1, TopK = 0, MinP = 0, RepetitionPenalty = 1,
                MaxTokens = GeneratedTokens, Seed = 42, EnableThinking = false, Device = selector, GraphDecode = graph,
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
                for (int i = 0; i < requestWarmups; i++)
                {
                    if (family == "qwen")
                    {
                        if (batchSize == 1) Check(await engine.Text.GenerateAsync(spec, text), expectPrompt);
                        else foreach (TimedRequest result in await BatchAsync(engine, spec, text)) Check(result.Result, expectPrompt);
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
                if (completion is not null) Check(completion, expectPrompt);
                if (batched is not null) foreach (TimedRequest result in batched) Check(result.Result, expectPrompt);
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
                long dropped = 0;
                if (evidence is not null)
                {
                    dropped = (long)evidence.GetType().GetProperty("Dropped")!.GetValue(evidence)!;
                    evidence.Dispose();
                    using FileStream observations = new(options["observe"], FileMode.CreateNew);
                    evidence.GetType().GetMethod("WriteJson")!.Invoke(evidence, [observations]);
                }
                AttentionDecodeProof? proof = evidence is not null && batched is not null
                    ? AttentionDecodeProof.Read(options["observe"], measuredObservationStart) : null;
                bool textsIdentical = batched is null || batched[0].Result.Text == batched[1].Result.Text;
                bool batchProven = proof is not null && proof.Proven && dropped == 0 && textsIdentical;
                using FileStream output = new(options["out"], FileMode.CreateNew);
                using Utf8JsonWriter writer = new(output, new JsonWriterOptions { Indented = true });
                writer.WriteStartObject();
                writer.WriteNumber("schema", 1);
                writer.WriteNumber("report_schema", 2);
                writer.WriteString("tool_sha256", Hash(typeof(AttentionRequestEvidence).Assembly.Location));
                writer.WriteString("protocol_sha256", protocolHash);
                writer.WriteString("family", family);
                writer.WriteString("backend", selector);
                writer.WriteString("workload_sha256", Hash(manifestPath));
                writer.WriteString("model_sha256", Hash(model));
                writer.WriteString("core_sha256", Hash(typeof(AttentionCall).Assembly.Location));
                writer.WriteString("engine_sha256", Hash(typeof(InferenceEngine).Assembly.Location));
                writer.WriteStartObject("resolved_settings");
                foreach ((string key, string value) in settings) writer.WriteString(key, value);
                writer.WriteEndObject();
                writer.WriteStartObject("diagnostic_overrides");
                writer.WriteBoolean("numerics.sdpaNoTf32", sdpaNoTf32);
                writer.WriteBoolean("numerics.noTf32", noTf32);
                writer.WriteEndObject();
                writer.WriteNumber("kv_pool_budget_bytes", poolBytes);
                writer.WriteNumber("expect_prompt_tokens", expectPrompt);
                writer.WriteString("lane", family == "krea" ? "image" : lane);
                writer.WriteString("telemetry", evidence is null ? "disabled" : "enabled");
                writer.WriteNumber("telemetry_dropped", dropped);
                writer.WriteNumber("measured_observation_start", measuredObservationStart);
                writer.WriteString("observations_include", "request warmups (including lazy model loading), then the measured request");
                writer.WriteNumber("milliseconds", milliseconds);
                writer.WriteNumber("request_warmups", requestWarmups);
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
                    writer.WriteBoolean("texts_identical", textsIdentical);
                    writer.WriteNumber("tokens_per_second", GeneratedTokens * batched.Length / milliseconds * 1000);
                    writer.WriteStartArray("requests");
                    foreach (TimedRequest sample in batched)
                    {
                        TextResult result = sample.Result;
                        writer.WriteStartObject();
                        writer.WriteNumber("milliseconds", sample.Milliseconds);
                        writer.WriteNumber("prompt_tokens", result.PromptTokens);
                        writer.WriteNumber("completion_tokens", result.CompletionTokens);
                        writer.WriteString("stop", result.Stop.ToString());
                        writer.WriteString("text", result.Text);
                        writer.WriteNumber("prefill_ms", result.PrefillMilliseconds);
                        writer.WriteNumber("decode_ms", result.DecodeMilliseconds);
                        writer.WriteEndObject();
                    }
                    writer.WriteEndArray();
                }
                if (proof is not null)
                {
                    int pageSize = new EngineOptions().KvPageSize;
                    proof.WriteBatchProof(writer, requireBatchedDecode, batchProven);
                    writer.WriteStartObject("paged_observations");
                    writer.WriteNumber("paged_admission_events", proof.PagedAdmissionEvents);
                    writer.WriteNumber("paged_layers_observed", proof.PagedLayers);
                    writer.WriteString("geometry_in_payload_events", KvCapacity.ObservedEventsReason(proof));
                    writer.WriteEndObject();
                    writer.WritePropertyName("kv_capacity");
                    KvCapacityFor(model, poolBytes, pageSize, expectPrompt).WriteJson(writer);
                }
                writer.WriteEndObject();
                if (requireBatchedDecode && !batchProven)
                {
                    Console.Error.WriteLine("batched decode not proven: needs a measured scheduler.decode.batch with Count >= 2, no dropped telemetry and identical texts.");
                    return 3;
                }
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

    private static void Check(TextResult result, int expectPrompt)
    {
        if (result.PromptTokens != expectPrompt || result.CompletionTokens != GeneratedTokens || result.Stop != StopReason.Length)
            throw new InvalidDataException($"Qwen workload requires exactly {expectPrompt} prompt and {GeneratedTokens} generated tokens ending at the limit.");
    }

    private static long ParseLong(Dictionary<string, string> options, string name, long fallback) =>
        options.TryGetValue(name, out string? text) ? long.Parse(text, NumberStyles.Integer, CultureInfo.InvariantCulture) : fallback;

    private static bool ParseFlag(Dictionary<string, string> options, string name) =>
        options.TryGetValue(name, out string? text) && text switch
        {
            "true" => true,
            "false" => false,
            _ => throw new ArgumentException($"--{name} takes true or false."),
        };

    /// <summary>KV capacity from the model's GGUF header, identified by the hash the run records. Unknown geometry is reported as
    /// unavailable with its reason; no page figure is written.</summary>
    private static KvCapacity KvCapacityFor(string model, long poolBytes, int pageSize, long promptTokens)
    {
        (KvGeometry? geometry, string? reason) = KvCapacity.ReadGgufGeometry(model, Hash(model));
        return geometry is null
            ? KvCapacity.Unavailable(reason!, poolBytes, pageSize)
            : KvCapacity.Compute(geometry, poolBytes, pageSize, promptTokens, GeneratedTokens);
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
