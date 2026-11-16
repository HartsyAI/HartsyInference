using System.Collections.Immutable;
using System.Text;
using System.Text.Json;
using HartsyInference.Core.Tensors;
using HartsyInference.GenCapture;
using HartsyInference.LLM.Transformer;
using HartsyInference.ModelAssets.Gguf;
using Xunit;

namespace HartsyInference.GenCapture.Tests;

/// <summary>CPU-only parser and capacity tests. No GPU and no engine run is involved; the GGUF headers are written here.</summary>
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
        KvCapacity capacity = KvCapacity.Compute(KvGeometry.Uniform(36, 8, 128, "qwen3", "sha"), FourGiB, 16, promptTokens: 1024, generatedTokens: 128);
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
        Assert.Equal(128, json.GetProperty("geometry").GetProperty("head_dim").GetInt32());
    }

    [Fact]
    public void UnknownLayerCount_IsUnavailable_NotZeroCapacity()
    {
        KvCapacity capacity = KvCapacity.Compute(KvGeometry.Uniform(0, 8, 128, "qwen3", "sha"), FourGiB, 16, promptTokens: 1024, generatedTokens: 128);
        JsonElement json = Render(capacity);

        Assert.False(capacity.Available);
        Assert.Equal("KV geometry is not positive", json.GetProperty("reason").GetString());
        Assert.False(json.TryGetProperty("pool_pages", out _));
        Assert.False(json.TryGetProperty("spare_pages_after_two_requests", out _));
    }

    [Fact]
    public void GgufHeader_YieldsKvGeometry_ThroughTheEngineReader_WhenTheHashMatches()
    {
        string model = WriteGguf("qwen3.gguf", Qwen3Metadata());

        (KvGeometry? geometry, string? reason) = KvCapacity.ReadGgufGeometry(model, KvCapacity.Sha256Hex(model));

        Assert.Null(reason);
        Assert.NotNull(geometry);
        Assert.Equal(36, geometry!.Layers);
        Assert.Equal(8, geometry.KvHeads);
        Assert.True(geometry.UniformHeadDim);
        Assert.All(geometry.HeadDimPerLayer, dim => Assert.Equal(128, dim));
        Assert.Equal("qwen3", geometry.Architecture);
    }

    [Fact]
    public void GgufHeader_DerivesHeadDim_FromEmbeddingLength_AndDefaultsKvHeads_WhenAbsent()
    {
        string model = WriteGguf("qwen3-derived.gguf", Qwen3Metadata(heads: 8, kvHeads: null, keyLength: null, embedding: 512));

        (KvGeometry? geometry, string? reason) = KvCapacity.ReadGgufGeometry(model, KvCapacity.Sha256Hex(model));

        Assert.Null(reason);
        Assert.Equal(64, geometry!.HeadDimPerLayer[0]);
        Assert.Equal(8, geometry.KvHeads);
    }

    [Fact]
    public void GgufHeader_ExcludesNextTokenPredictionLayers_LikeTheEngine()
    {
        string model = WriteGguf("qwen3-nextn.gguf", Qwen3Metadata(blockCount: 37, nextn: 1));

        (KvGeometry? geometry, string? reason) = KvCapacity.ReadGgufGeometry(model, KvCapacity.Sha256Hex(model));

        Assert.Null(reason);
        Assert.Equal(36, geometry!.Layers);
    }

    [Fact]
    public void GgufHeader_IsRefused_WhenTheFileDoesNotMatchTheRecordedHash()
    {
        string model = WriteGguf("mismatch.gguf", Qwen3Metadata());

        (KvGeometry? geometry, string? reason) = KvCapacity.ReadGgufGeometry(model, new string('0', 64));

        Assert.Null(geometry);
        Assert.Contains("SHA-256", reason!);
    }

    [Fact]
    public void GgufHeader_WithoutBlockCount_IsUnavailable()
    {
        string model = WriteGguf("noblocks.gguf", Qwen3Metadata(blockCount: null));

        (KvGeometry? geometry, string? reason) = KvCapacity.ReadGgufGeometry(model, KvCapacity.Sha256Hex(model));

        Assert.Null(geometry);
        Assert.Contains("block_count", reason!);
    }

    [Fact]
    public void GgufHeader_ForAnotherArchitecture_IsUnavailableBeforeAnyReaderRuns()
    {
        string model = WriteGguf("llama.gguf",
            ("general.architecture", "llama"),
            ("llama.block_count", 4u),
            ("llama.attention.head_count", 8u),
            ("llama.embedding_length", 512u),
            ("llama.feed_forward_length", 2048u));

        (KvGeometry? geometry, string? reason) = KvCapacity.ReadGgufGeometry(model, KvCapacity.Sha256Hex(model));

        Assert.Null(geometry);
        Assert.Contains("no verified KV capacity reader", reason!);
        Assert.Contains("'llama'", reason!);
    }

    [Fact]
    public void GgufHeader_ReaderErrors_BecomeUnavailableReasons_NotExceptions()
    {
        string model = WriteGguf("qwen3-no-ffn.gguf", Qwen3Metadata(feedForward: null));

        (KvGeometry? geometry, string? reason) = KvCapacity.ReadGgufGeometry(model, KvCapacity.Sha256Hex(model));

        Assert.Null(geometry);
        Assert.Contains("engine GGUF reader refused the header", reason!);
        Assert.Contains("feed_forward_length", reason!);
    }

    [Fact]
    public void GgufHeader_NotAGgufFile_IsReportedNotThrown()
    {
        string path = Path.Combine(_root, "garbage.gguf");
        File.WriteAllBytes(path, Encoding.ASCII.GetBytes("this is not a GGUF file"));

        (KvGeometry? geometry, string? reason) = KvCapacity.ReadGgufGeometry(path, KvCapacity.Sha256Hex(path));

        Assert.Null(geometry);
        Assert.Contains("engine GGUF reader refused the header", reason!);
    }

    [Fact]
    public void EngineConfig_MultiHeadLatentAttention_IsUnavailable()
    {
        TransformerConfig config = TransformerConfig.Qwen3_0_6B with
        {
            Mla = new MlaConfig { KvLoraRank = 512, QkNopeHeadDim = 128, QkRopeHeadDim = 64, VHeadDim = 128 },
        };

        (KvGeometry? geometry, string? reason) = KvCapacity.FromEngineConfig(config, "qwen3", "sha");

        Assert.Null(geometry);
        Assert.Contains("latent KV cache", reason!);
    }

    [Fact]
    public void EngineConfig_CrossAttentionLayers_AreUnavailable()
    {
        TransformerConfig config = TransformerConfig.Qwen3_0_6B with { CrossAttnLayers = ImmutableHashSet.Create(3) };

        (KvGeometry? geometry, string? reason) = KvCapacity.FromEngineConfig(config, "qwen3", "sha");

        Assert.Null(geometry);
        Assert.Contains("cross-attention", reason!);
    }

    [Fact]
    public void EngineConfig_PerLayerHeadDims_AreSizedAsTheEnginesPoolSizesThem()
    {
        TransformerConfig config = TransformerConfig.Qwen3_0_6B with
        {
            NumLayers = 4,
            NumKvHeads = 2,
            HeadDim = 256,
            HeadDimSwa = 128,
            GlobalLayers = new[] { true, false, false, true },
        };

        (KvGeometry? geometry, string? reason) = KvCapacity.FromEngineConfig(config, "qwen3", "sha");

        Assert.Null(reason);
        Assert.Equal(new[] { 256, 128, 128, 256 }, geometry!.HeadDimPerLayer);
        Assert.False(geometry.UniformHeadDim);

        KvCapacity capacity = KvCapacity.Compute(geometry, FourGiB, 16, promptTokens: 1024, generatedTokens: 128);
        int engine = PagedKvPool.PageCountForBudget(2, new[] { 256, 128, 128, 256 }, 16, FourGiB);
        int uniformFirstLayer = PagedKvPool.PageCountForBudget(2, new[] { 256, 256, 256, 256 }, 16, FourGiB);
        Assert.Equal(engine, capacity.PoolPages);
        Assert.NotEqual(uniformFirstLayer, capacity.PoolPages);

        JsonElement json = Render(capacity);
        JsonElement shape = json.GetProperty("geometry");
        Assert.Equal(4, shape.GetProperty("head_dim_per_layer").GetArrayLength());
        Assert.False(shape.TryGetProperty("head_dim", out _));
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

    /// <summary>A GGUF header with the metadata and one embedding tensor. The embedding tensor is what the engine's config
    /// reader checks for, and the file is read through the same loader the engine uses.</summary>
    private string WriteGguf(string name, params (string Key, object Value)[] metadata) =>
        WriteGguf(name, (IEnumerable<(string Key, object Value)>)metadata);

    private string WriteGguf(string name, IEnumerable<(string Key, object Value)> metadata)
    {
        string path = Path.Combine(_root, name);
        using GgufWriter writer = new(path);
        foreach ((string key, object value) in metadata) writer.SetMetadata(key, value);
        writer.AddTensor("token_embd.weight", new TensorShape(2, 2), DType.F32, () => new Tensor(new TensorShape(2, 2), DType.F32));
        writer.Flush();
        return path;
    }

    /// <summary>The metadata of a qwen3 header with the keys the engine's reader requires. A null argument leaves that key out.</summary>
    private static List<(string Key, object Value)> Qwen3Metadata(uint? blockCount = 36, uint heads = 32, uint? kvHeads = 8,
        uint? keyLength = 128, uint embedding = 2560, uint? feedForward = 9728, uint? nextn = null)
    {
        List<(string Key, object Value)> metadata = [("general.architecture", "qwen3")];
        if (blockCount is { } blocks) metadata.Add(("qwen3.block_count", blocks));
        metadata.Add(("qwen3.attention.head_count", heads));
        metadata.Add(("qwen3.embedding_length", embedding));
        if (feedForward is { } ffn) metadata.Add(("qwen3.feed_forward_length", ffn));
        if (kvHeads is { } kv) metadata.Add(("qwen3.attention.head_count_kv", kv));
        if (keyLength is { } key) metadata.Add(("qwen3.attention.key_length", key));
        if (nextn is { } predicted) metadata.Add(("qwen3.nextn_predict_layers", predicted));
        return metadata;
    }

    private static JsonElement Render(KvCapacity capacity)
    {
        using MemoryStream buffer = new();
        using (Utf8JsonWriter writer = new(buffer)) capacity.WriteJson(writer);
        using JsonDocument document = JsonDocument.Parse(buffer.ToArray());
        return document.RootElement.Clone();
    }
}
