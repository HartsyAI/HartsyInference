using System.Globalization;
using System.Security.Cryptography;
using System.Text.Json;
using HartsyInference.Engine;

namespace HartsyInference.GenCapture;

/// <summary>Writes a new, write-once batch-capacity report from an existing batched-run result. The result and its
/// observation file are read and never changed. The decode proof is recomputed from the observation file and must match the
/// original result's counts. KV capacity comes from the model's GGUF header, which must hash to the recorded model_sha256.
/// This command loads no weights and touches no GPU.</summary>
internal static class AttentionCapacityDerive
{
    // The primary request generates this many tokens (AttentionRequestEvidence.GeneratedTokens).
    private const int GeneratedTokens = 128;

    public static int Run(Dictionary<string, string> options)
    {
        string resultPath = options["result"];
        string observationsPath = options["observations"];
        string modelPath = options["model"];
        using FileStream output = KvCapacity.CreateFreshOutput(options["out"]);

        byte[] resultBytes = File.ReadAllBytes(resultPath);
        using JsonDocument document = JsonDocument.Parse(resultBytes);
        JsonElement result = document.RootElement;
        string modelSha = result.GetProperty("model_sha256").GetString()!;
        long poolBytes = result.GetProperty("kv_pool_budget_bytes").GetInt64();
        long promptTokens = result.GetProperty("expect_prompt_tokens").GetInt64();
        long measuredStart = result.GetProperty("measured_observation_start").GetInt64();
        JsonElement original = result.GetProperty("batch_proof");
        int pageSize = new EngineOptions().KvPageSize;
        if (original.GetProperty("kv_page_size").GetInt32() != pageSize)
            throw new InvalidDataException("recorded KV page size differs from the engine default; refusing to derive capacity");
        foreach (JsonElement request in result.GetProperty("requests").EnumerateArray())
        {
            if (request.GetProperty("completion_tokens").GetInt32() != GeneratedTokens)
                throw new InvalidDataException("a request generated a different token count; refusing to derive capacity");
        }

        AttentionDecodeProof proof = AttentionDecodeProof.Read(observationsPath, measuredStart);
        bool countsMatch = CountsMatch(original, proof);
        if (!countsMatch) throw new InvalidDataException("recomputed decode proof differs from the original result; refusing to derive");

        (KvGeometry? geometry, string? reason) = KvCapacity.ReadGgufGeometry(modelPath, modelSha);
        KvCapacity capacity = geometry is null
            ? KvCapacity.Unavailable(reason!, poolBytes, pageSize)
            : KvCapacity.Compute(geometry, poolBytes, pageSize, promptTokens, GeneratedTokens);

        using Utf8JsonWriter writer = new(output, new JsonWriterOptions { Indented = true });
        writer.WriteStartObject();
        writer.WriteNumber("schema", 1);
        writer.WriteString("report", "batch-capacity-derived");
        writer.WriteNumber("report_schema", 2);
        writer.WriteStartObject("source_result");
        writer.WriteString("path", Path.GetFullPath(resultPath));
        writer.WriteString("sha256", Convert.ToHexString(SHA256.HashData(resultBytes)).ToLowerInvariant());
        writer.WriteString("status", "read only; not modified");
        writer.WriteEndObject();
        writer.WriteStartObject("source_observations");
        writer.WriteString("path", Path.GetFullPath(observationsPath));
        writer.WriteString("sha256", KvCapacity.Sha256Hex(observationsPath));
        writer.WriteEndObject();
        writer.WriteString("model_sha256", modelSha);
        writer.WriteBoolean("proof_counts_match_original", countsMatch);
        proof.WriteBatchProof(writer, original.GetProperty("required").GetBoolean(), original.GetProperty("proven").GetBoolean());
        writer.WriteStartObject("paged_observations");
        writer.WriteNumber("paged_admission_events", proof.PagedAdmissionEvents);
        writer.WriteNumber("paged_layers_observed", proof.PagedLayers);
        writer.WriteString("geometry_in_payload_events", KvCapacity.ObservedEventsReason(proof));
        writer.WriteEndObject();
        writer.WritePropertyName("kv_capacity");
        capacity.WriteJson(writer);
        writer.WriteEndObject();
        writer.Flush();
        return 0;
    }

    /// <summary>True when the recomputed decode counts equal the counts the original run recorded.</summary>
    private static bool CountsMatch(JsonElement original, AttentionDecodeProof proof)
    {
        if (original.GetProperty("decode_batch_events_total").GetInt32() != proof.DecodeEvents) return false;
        if (original.GetProperty("decode_batch_events_measured").GetInt32() != proof.MeasuredDecodeEvents) return false;
        if (original.GetProperty("max_active_sequences_measured").GetInt32() != proof.MaxCountMeasured) return false;
        if (original.GetProperty("decode_steps_with_two_or_more_measured").GetInt32() != proof.StepsWithTwoOrMoreMeasured) return false;
        if (original.GetProperty("decode_steps_with_two_or_more_warmup").GetInt32() != proof.WarmupStepsWithTwoOrMore) return false;
        Dictionary<int, int> recorded = original.GetProperty("measured_count_histogram").EnumerateObject()
            .ToDictionary(p => int.Parse(p.Name, CultureInfo.InvariantCulture), p => p.Value.GetInt32());
        return recorded.Count == proof.MeasuredCountHistogram.Count
            && recorded.All(entry => proof.MeasuredCountHistogram.TryGetValue(entry.Key, out int steps) && steps == entry.Value);
    }
}
