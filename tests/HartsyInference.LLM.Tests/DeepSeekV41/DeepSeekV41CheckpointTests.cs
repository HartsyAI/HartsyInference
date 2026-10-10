using System.Text.Json.Nodes;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.DeepSeekV41;
using HartsyInference.ModelAssets.Quant;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.LLM.Tests.DeepSeekV41;

/// <summary>Opening a DeepSeek-V4.1 checkpoint from headers alone: mapping, binding, borrowed views, Engram and the draft scan.</summary>
public sealed class DeepSeekV41CheckpointTests : IDisposable
{
    /// <summary>Env var pointing at a directory of full-length sparse shard files carrying the 48 real official headers.</summary>
    private const string ReplicaEnvVar = "HARTSY_DSV41_HEADER_REPLICA";

    private readonly ITestOutputHelper _output;
    private readonly string _directory = Directory.CreateTempSubdirectory("dsv41-checkpoint-").FullName;

    public DeepSeekV41CheckpointTests(ITestOutputHelper output) => _output = output;

    public void Dispose() => Directory.Delete(_directory, recursive: true);

    [Fact]
    public void Open_Official_BindsFp4ExpertsAndKeepsEngramPreadOnly()
    {
        TinyDeepSeekV41Checkpoint.Write(_directory);

        using DeepSeekV41Checkpoint checkpoint = DeepSeekV41Checkpoint.Open(_directory);

        QuantWeightInfo? expert = checkpoint.GetQuant("layers.2.ffn.experts.3.w1.weight");
        Assert.NotNull(expert);
        Assert.Null(checkpoint.GetQuant("layers.2.attn.wq_a.weight"));
        Assert.True(checkpoint.Bindings.Bindings.ContainsKey("layers.1.engram.embed.weight"));
        // The Engram scale lives in a pread-only shard, so a recipe (which maps the scale) is refused, not silently mapped.
        Assert.Throws<InvalidOperationException>(() => checkpoint.GetQuant("layers.1.engram.embed.weight"));
    }

    [Fact]
    public void EngramTable_IsPreadOnlyAndNeverMapped()
    {
        TinyDeepSeekV41Checkpoint.Write(_directory);

        using DeepSeekV41Checkpoint checkpoint = DeepSeekV41Checkpoint.Open(_directory);
        DeepSeekV41EngramTable table = checkpoint.EngramTable(TinyDeepSeekV41Checkpoint.EngramLayer);

        Assert.Equal(TinyDeepSeekV41Checkpoint.EngramRows, table.Rows);
        Assert.NotNull(table.Scale);
        Assert.Equal(TinyDeepSeekV41Checkpoint.EngramRows, table.Scale!.Shape[0]);
        Assert.ThrowsAny<Exception>(() => checkpoint.GetWeight("layers.1.engram.embed.weight"));
        Assert.Throws<ArgumentException>(() => checkpoint.EngramTable(0));
    }

    [Fact]
    public void ExpertBank_HandsOutLocationsForEachExpertProjection()
    {
        TinyDeepSeekV41Checkpoint.Write(_directory);

        using DeepSeekV41Checkpoint checkpoint = DeepSeekV41Checkpoint.Open(_directory);
        DeepSeekV41ExpertBank bank = checkpoint.ExpertBank(2);
        DeepSeekV41ExpertBank draft = checkpoint.ExpertBank(TinyDeepSeekV41Checkpoint.BackboneLayers);

        Assert.Equal(TinyDeepSeekV41Checkpoint.RoutedExperts, bank.ExpertCount);
        Assert.Equal(TinyDeepSeekV41Checkpoint.DraftExperts, draft.ExpertCount);
        Assert.Equal("layers.2.ffn.experts.1.w3.weight", bank.WeightKey(1, DeepSeekV41ExpertProjection.W3));
        Assert.Equal("mtp.0.ffn.experts.2.w2.weight", draft.WeightKey(2, DeepSeekV41ExpertProjection.W2));
        Assert.Equal(64, bank.Location(3, DeepSeekV41ExpertProjection.W1).ByteLength);
        Assert.Throws<ArgumentOutOfRangeException>(() => checkpoint.ExpertBank(TinyDeepSeekV41Checkpoint.BackboneLayers + 1));
    }

    [Fact]
    public void Open_WithoutDraft_ReportsAbsentAndRefusesTheDraftBank()
    {
        TinyDeepSeekV41Checkpoint.Write(_directory, includeDraft: false);

        using DeepSeekV41Checkpoint checkpoint = DeepSeekV41Checkpoint.Open(_directory);

        Assert.Equal(DeepSeekV41DraftStatus.Absent, checkpoint.Draft.Status);
        Assert.Equal(0, checkpoint.Weights.BytesByClass[DeepSeekV41WeightClass.Draft]);
        HartsyInferenceException error = Assert.Throws<HartsyInferenceException>(
            () => checkpoint.ExpertBank(TinyDeepSeekV41Checkpoint.BackboneLayers));
        Assert.Contains("no draft", error.Message);
    }

    [Fact]
    public void Open_MlxWithGappedDraft_RefusesTheDraftNamingTheMissingExperts()
    {
        // Expert 0 whole, expert 1 weights only (like MLX's expert 87), expert 2 absent (like 88-99).
        TinyDeepSeekV41Checkpoint.Write(_directory, QuantFlavor.Mlx, fullDraftExperts: [0], partialDraftExpert: 1);

        using DeepSeekV41Checkpoint checkpoint = DeepSeekV41Checkpoint.Open(_directory);

        Assert.Equal(QuantFlavor.Mlx, checkpoint.Flavor);
        Assert.Equal(DeepSeekV41DraftStatus.Incomplete, checkpoint.Draft.Status);
        Assert.Equal([1, 2], checkpoint.Draft.MissingExpertsByLayer[0]);
        Assert.Equal("mtp.0 lacks 2 of 3 routed experts (1-2)", checkpoint.Draft.DescribeMissing());
        HartsyInferenceException error = Assert.Throws<HartsyInferenceException>(() => checkpoint.RequireDraft());
        Assert.Contains("DSpark speculative decoding is refused", error.Message);
        Assert.Contains("mtp.0 lacks 2 of 3 routed experts (1-2)", error.Message);
        Assert.Throws<HartsyInferenceException>(() => checkpoint.ExpertBank(TinyDeepSeekV41Checkpoint.BackboneLayers));
        Assert.False(checkpoint.HasWeight("mtp.0.ffn.experts.1.w1.weight"));
        Assert.True(checkpoint.HasWeight("mtp.0.ffn.experts.0.w1.weight"));
        Assert.NotNull(checkpoint.GetQuant("layers.0.ffn.experts.0.w1.weight"));
    }

    [Fact]
    public void Open_RefusesEngramTablesSharingAShardWithOtherWeights()
    {
        TinyDeepSeekV41Checkpoint.Write(_directory);
        string indexPath = Path.Combine(_directory, "model.safetensors.index.json");
        JsonNode index = JsonNode.Parse(File.ReadAllText(indexPath))!;
        foreach (KeyValuePair<string, JsonNode?> entry in index["weight_map"]!.AsObject().ToArray())
            index["weight_map"]![entry.Key] = "model-00001-of-00003.safetensors";
        File.WriteAllText(indexPath, index.ToJsonString());

        HartsyInferenceException error = Assert.Throws<HartsyInferenceException>(() => DeepSeekV41Checkpoint.Open(_directory));

        Assert.Contains("Engram tables share shards", error.Message);
    }

    [Fact]
    public void Open_RejectsAnUnshardedOrForeignDirectory()
    {
        File.WriteAllText(Path.Combine(_directory, "config.json"), "{\"model_type\":\"llama\"}");
        File.WriteAllBytes(Path.Combine(_directory, "model.safetensors"), new byte[16]);

        HartsyInferenceException error = Assert.Throws<HartsyInferenceException>(() => DeepSeekV41Checkpoint.Open(_directory));

        Assert.Contains("model_type 'llama'", error.Message);
    }

    [Fact]
    public void RealOfficialShardHeaders_OpenAndSumToTheIndexTotal()
    {
        string? replica = Environment.GetEnvironmentVariable(ReplicaEnvVar);
        if (string.IsNullOrEmpty(replica) || !RealWeightGate.Require(_output.WriteLine, replica))
        {
            _output.WriteLine($"SKIPPED: set {ReplicaEnvVar} to a directory of sparse full-length shards with the real headers.");
            return;
        }

        using DeepSeekV41Checkpoint checkpoint = DeepSeekV41Checkpoint.Open(replica);

        Assert.Equal(DeepSeekV41HeaderTemplates.OfficialTensorCount, checkpoint.Weights.TotalCount);
        Assert.Equal(DeepSeekV41HeaderTemplates.OfficialTotalSize, checkpoint.Weights.TotalBytes);
        Assert.Equal(DeepSeekV41DraftStatus.Complete, checkpoint.Draft.Status);
        Assert.Equal(384006168L, checkpoint.EngramTable(1).Rows);
        Assert.Equal(384016682L, checkpoint.EngramTable(14).Rows);
        Assert.Equal(384, checkpoint.ExpertBank(39).ExpertCount);
        Assert.Equal(128, checkpoint.ExpertBank(42).ExpertCount);
    }
}
