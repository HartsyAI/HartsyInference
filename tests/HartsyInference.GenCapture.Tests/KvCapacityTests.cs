using System.Text;
using System.Text.Json;
using HartsyInference.GenCapture;
using HartsyInference.ModelAssets.Gguf;
using Xunit;

namespace HartsyInference.GenCapture.Tests;

/// <summary>CPU-only parser and capacity tests. No GPU, no weights and no engine run is involved.</summary>
public sealed class KvCapacityTests : IDisposable
{
    private const long FourGiB = 4L * 1024 * 1024 * 1024;
    private readonly string _root = Path.Combine(Path.GetTempPath(), "gencapture-tests-" + Guid.NewGuid().ToString("N"));

    public KvCapacityTests() => Directory.CreateDirectory(_root);

    public void Dispose()
    {
        if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true);
    }

    [Fact]
    public void ObservedZeroGeometryEvents_ReportUnavailable_AndWriteNoPageFigures()
    {
        string events = WriteFile("zero-geometry.json", """
            [
              {"Site":"scheduler.paged","Count":1},
              {"Site":"paged.keys","Layer":0,"KvHeads":0,"QkHeadDim":0,"Count":1},
              {"Site":"paged.keys","Layer":1,"KvHeads":0,"QkHeadDim":0,"Count":1},
              {"Site":"scheduler.decode.batch","Count":2}
            ]
            """);

        AttentionDecodeProof proof = AttentionDecodeProof.Read(events, measuredStart: 0);
        Assert.Equal(2, proof.PagedLayers);
        Assert.Equal(0, proof.PagedKvHeads);

        KvCapacity capacity = KvCapacity.Unavailable(KvCapacity.ObservedEventsReason(proof), FourGiB, 16);
        JsonElement json = Render(capacity);

        Assert.False(capacity.Available);
        Assert.Equal("unavailable", json.GetProperty("status").GetString());
        Assert.Contains("KvHeads=0", json.GetProperty("reason").GetString());
        foreach (string forbidden in new[] { "pool_pages", "pages_per_request", "pages_two_requests", "spare_pages_after_two_requests", "geometry" })
            Assert.False(json.TryGetProperty(forbidden, out _), forbidden);
    }

    [Fact]
    public void ValidGeometry_ReportsPoolPagesRequestsAndSpare()
    {
        KvCapacity capacity = KvCapacity.Compute(new KvGeometry(36, 8, 128, "qwen3", "sha"), FourGiB, 16, promptTokens: 1024, generatedTokens: 128);
        JsonElement json = Render(capacity);

        Assert.True(capacity.Available);
        Assert.Equal(910, capacity.PoolPages);
        Assert.Equal(73, capacity.PagesPerRequest);
        Assert.Equal(146, capacity.PagesTwoRequests);
        Assert.Equal(764, capacity.SparePagesAfterTwoRequests);
        Assert.Equal("available", json.GetProperty("status").GetString());
        Assert.Equal(910, json.GetProperty("pool_pages").GetInt32());
        Assert.Equal(764, json.GetProperty("spare_pages_after_two_requests").GetInt32());
        Assert.Equal("gguf-header", json.GetProperty("source").GetProperty("kind").GetString());
    }

    [Fact]
    public void UnknownLayerCount_IsUnavailable_NotZeroCapacity()
    {
        KvCapacity capacity = KvCapacity.Compute(new KvGeometry(0, 8, 128, "qwen3", "sha"), FourGiB, 16, promptTokens: 1024, generatedTokens: 128);
        JsonElement json = Render(capacity);

        Assert.False(capacity.Available);
        Assert.Equal("KV geometry is not positive", json.GetProperty("reason").GetString());
        Assert.False(json.TryGetProperty("pool_pages", out _));
        Assert.False(json.TryGetProperty("spare_pages_after_two_requests", out _));
    }

    [Fact]
    public void GgufHeader_YieldsKvGeometry_WhenTheHashMatches()
    {
        string model = WriteGguf("qwen3.gguf",
            ("general.architecture", "qwen3"),
            ("qwen3.block_count", 36u),
            ("qwen3.attention.head_count", 32u),
            ("qwen3.attention.head_count_kv", 8u),
            ("qwen3.attention.key_length", 128u),
            ("qwen3.embedding_length", 2560u));

        (KvGeometry? geometry, string? reason) = KvCapacity.ReadGgufGeometry(model, KvCapacity.Sha256Hex(model));

        Assert.Null(reason);
        Assert.NotNull(geometry);
        Assert.Equal(36, geometry!.Layers);
        Assert.Equal(8, geometry.KvHeads);
        Assert.Equal(128, geometry.HeadDim);
        Assert.Equal("qwen3", geometry.Architecture);
    }

    [Fact]
    public void GgufHeader_DerivesHeadDim_FromEmbeddingLength_AndDefaultsKvHeads_WhenAbsent()
    {
        string model = WriteGguf("llama.gguf",
            ("general.architecture", "llama"),
            ("llama.block_count", 4u),
            ("llama.attention.head_count", 8u),
            ("llama.embedding_length", 512u));

        (KvGeometry? geometry, string? reason) = KvCapacity.ReadGgufGeometry(model, KvCapacity.Sha256Hex(model));

        Assert.Null(reason);
        Assert.Equal(64, geometry!.HeadDim);
        Assert.Equal(8, geometry.KvHeads);
    }

    [Fact]
    public void GgufHeader_IsRefused_WhenTheFileDoesNotMatchTheRecordedHash()
    {
        string model = WriteGguf("mismatch.gguf",
            ("general.architecture", "qwen3"),
            ("qwen3.block_count", 36u),
            ("qwen3.attention.head_count_kv", 8u),
            ("qwen3.attention.key_length", 128u));

        (KvGeometry? geometry, string? reason) = KvCapacity.ReadGgufGeometry(model, new string('0', 64));

        Assert.Null(geometry);
        Assert.Contains("SHA-256", reason!);
    }

    [Fact]
    public void GgufHeader_WithoutBlockCount_IsUnavailable()
    {
        string model = WriteGguf("noblocks.gguf",
            ("general.architecture", "qwen3"),
            ("qwen3.attention.head_count_kv", 8u),
            ("qwen3.attention.key_length", 128u));

        (KvGeometry? geometry, string? reason) = KvCapacity.ReadGgufGeometry(model, KvCapacity.Sha256Hex(model));

        Assert.Null(geometry);
        Assert.Contains("block_count", reason!);
    }

    [Fact]
    public void Proof_CountsOnlyTheMeasuredWindow_AndItsHistogram()
    {
        string events = WriteFile("proof.json", """
            [
              {"Site":"scheduler.decode.batch","Count":2},
              {"Site":"scheduler.decode.batch","Count":1},
              {"Site":"scheduler.decode.batch","Count":2},
              {"Site":"scheduler.decode.batch","Count":2}
            ]
            """);

        AttentionDecodeProof proof = AttentionDecodeProof.Read(events, measuredStart: 2);

        Assert.Equal(4, proof.DecodeEvents);
        Assert.Equal(2, proof.MeasuredDecodeEvents);
        Assert.Equal(2, proof.MaxCountMeasured);
        Assert.Equal(2, proof.StepsWithTwoOrMoreMeasured);
        Assert.Equal(1, proof.WarmupStepsWithTwoOrMore);
        Assert.Equal(2, proof.MeasuredCountHistogram[2]);
        Assert.True(proof.Proven);
    }

    [Fact]
    public void BatchProofSection_CarriesNoCapacityFields()
    {
        string events = WriteFile("proof-only.json",
            """[{"Site":"scheduler.decode.batch","Count":2},{"Site":"paged.keys","Layer":0,"KvHeads":0,"QkHeadDim":0,"Count":1}]""");
        AttentionDecodeProof proof = AttentionDecodeProof.Read(events, measuredStart: 0);

        using MemoryStream buffer = new();
        using (Utf8JsonWriter writer = new(buffer))
        {
            writer.WriteStartObject();
            proof.WriteBatchProof(writer, required: true, proven: proof.Proven);
            writer.WriteEndObject();
        }
        using JsonDocument document = JsonDocument.Parse(buffer.ToArray());
        JsonElement batch = document.RootElement.GetProperty("batch_proof");

        Assert.True(batch.GetProperty("proven").GetBoolean());
        foreach (string forbidden in new[] { "kv_pool_pages_expected", "pages_spare_after_two_requests_expected",
                     "paged_kv_heads_observed", "paged_head_dim_observed", "kv_page_size", "kv_capacity" })
            Assert.False(batch.TryGetProperty(forbidden, out _), forbidden);
    }

    [Fact]
    public void FreshOutput_RefusesAnExistingFile_AndKeepsItsContent()
    {
        string path = Path.Combine(_root, "derived.json");
        using (FileStream first = KvCapacity.CreateFreshOutput(path))
        {
            first.Write(Encoding.UTF8.GetBytes("original"));
        }

        Assert.ThrowsAny<IOException>(() => KvCapacity.CreateFreshOutput(path));
        Assert.Equal("original", File.ReadAllText(path));
    }

    private string WriteFile(string name, string content)
    {
        string path = Path.Combine(_root, name);
        File.WriteAllText(path, content);
        return path;
    }

    private string WriteGguf(string name, params (string Key, object Value)[] metadata)
    {
        string path = Path.Combine(_root, name);
        using GgufWriter writer = new(path);
        foreach ((string key, object value) in metadata) writer.SetMetadata(key, value);
        writer.Flush();
        return path;
    }

    private static JsonElement Render(KvCapacity capacity)
    {
        using MemoryStream buffer = new();
        using (Utf8JsonWriter writer = new(buffer)) capacity.WriteJson(writer);
        using JsonDocument document = JsonDocument.Parse(buffer.ToArray());
        return document.RootElement.Clone();
    }
}
