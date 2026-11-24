using System.IO.Compression;
using System.Text.Json;
using HartsyInference.Core.Exceptions;
using HartsyInference.LLM.Hy4Flash;

namespace HartsyInference.LLM.Tests.Hy4Flash;

/// <summary>Config and key-map checks against the real safetensors header inventories of the three Hy4 Flash checkpoints (headers only, no tensor data).</summary>
public sealed class Hy4FlashWeightMapTests
{
    private static string Fixture(string name) => Path.Combine(AppContext.BaseDirectory, "Fixtures", "Hy4Flash", name);

    private static Hy4FlashConfig Config(string variant) => Hy4FlashConfig.Load(Fixture($"{variant}.config.json"));

    private static List<Hy4FlashTensorInfo> Tensors(string variant)
    {
        using FileStream file = File.OpenRead(Fixture($"{variant}.tensors.json.gz"));
        using GZipStream gzip = new(file, CompressionMode.Decompress);
        using JsonDocument document = JsonDocument.Parse(gzip);
        List<Hy4FlashTensorInfo> list = [];
        foreach (JsonElement row in document.RootElement.GetProperty("tensors").EnumerateArray())
        {
            long[] shape = [.. row[2].EnumerateArray().Select(static e => e.GetInt64())];
            list.Add(new Hy4FlashTensorInfo(row[0].GetString()!, row[1].GetString()!, shape, row[3].GetInt64()));
        }
        return list;
    }

    [Theory]
    [InlineData("instruct")]
    [InlineData("fp8")]
    [InlineData("base")]
    public void RealInventory_MapsEveryKeyAndShape(string variant)
    {
        Hy4FlashWeightCoverage coverage = Hy4FlashWeightMap.Check(Config(variant), Tensors(variant));
        coverage.RequireComplete();
        Assert.Equal(Tensors(variant).Count, coverage.Total);
    }

    [Fact]
    public void InstructConfig_ReadsTheShippedLegacyNames()
    {
        Hy4FlashConfig c = Config("instruct");
        Assert.Equal(42, c.NumHiddenLayers);
        Assert.Equal(12, c.LayerKinds.Count(static k => k == Hy4FlashLayerKind.Indexed));
        int[] indexed = [.. Enumerable.Range(0, 42).Where(i => c.LayerKinds[i] == Hy4FlashLayerKind.Indexed)];
        Assert.Equal([0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41], indexed);
        Assert.Equal(32, c.HeadsFor(0));
        Assert.Equal(64, c.HeadsFor(2));
        Assert.False(c.SparseFfn[0]);
        Assert.All(Enumerable.Range(1, 41), i => Assert.True(c.SparseFfn[i]));
        Assert.Equal(1536, c.MoeHiddenSize);
        Assert.Equal(4, c.IndexKpool);
        Assert.Equal(2048, c.IndexTopk);
        Assert.Equal([12, 10, 10], c.MropeSection);
        Assert.Equal(10_000_000.0, c.IndexedRopeTheta);
        Assert.Equal(10_000.0, c.SlidingRopeTheta);
        Assert.Equal([2, 12, 22, 32], c.EngramLayerIds);
        Assert.Equal([1, 11, 21, 31], Enumerable.Range(0, 42).Where(c.HasEngram));
        Assert.False(c.IsFp8);
        Assert.False(c.EngramFp8Embedding);
        Assert.Equal(5, c.DflyNumHiddenLayers);
        Assert.Equal(120120, c.ImageTokenId);
        Assert.Equal(16385, c.Vision.PositionRows);
    }

    [Fact]
    public void Fp8Config_CarriesItsQuantizationInsideTheTextConfig()
    {
        Hy4FlashConfig c = Config("fp8");
        Assert.True(c.IsFp8);
        Assert.Equal("e4m3", c.QuantFormat);
        Assert.Equal([128, 128], c.QuantBlock);
    }

    [Fact]
    public void BaseConfig_DiffersFromInstructWhereTheFilesDo()
    {
        Hy4FlashConfig b = Config("base");
        Assert.True(b.EngramFp8Embedding);
        Assert.Equal(120020, b.EosTokenId);
        Assert.Equal(0, b.DflyNumHiddenLayers);
        Assert.Equal(120025, Config("instruct").EosTokenId);
    }

    [Fact]
    public void EngramHashModuli_SumToTheTableRows()
    {
        Hy4FlashConfig c = Config("instruct");
        Assert.Equal(164_994_380L, c.EngramTableRows(2));
        foreach (int id in c.EngramLayerIds)
        {
            long[] primes = [.. c.EngramHeadVocabSizes[id].SelectMany(static o => o)];
            Assert.Equal(16, primes.Length);
            Hy4FlashTensorInfo table = Tensors("instruct").Single(t => t.Name == $"model.layers.{id - 1}.engram_layer.multi_head_embedding.embedding.weight");
            Assert.Equal(primes.Sum(), table.Shape[0]);
            Assert.Equal(96, table.Shape[1]);
        }
    }

    [Fact]
    public void ByteClasses_MatchTheHeaderSizesQuotedInTheResearchDoc()
    {
        IReadOnlyDictionary<Hy4FlashWeightClass, long> bf16 = Hy4FlashWeightMap.Summarize(Tensors("instruct"));
        IReadOnlyDictionary<Hy4FlashWeightClass, long> fp8 = Hy4FlashWeightMap.Summarize(Tensors("fp8"));
        Assert.Equal(126.72, bf16[Hy4FlashWeightClass.EngramTable] / 1e9, 2);
        Assert.Equal(126.72, fp8[Hy4FlashWeightClass.EngramTable] / 1e9, 2);
        Assert.Equal(297.16, bf16[Hy4FlashWeightClass.RoutedExperts] / 1e9, 2);
        Assert.Equal(148.60, fp8[Hy4FlashWeightClass.RoutedExperts] / 1e9, 2);
    }

    [Fact]
    public void Drafter_IsSkippedNotMappedAndTheBaseHasNone()
    {
        Assert.Equal(87, Hy4FlashWeightMap.Check(Config("instruct"), Tensors("instruct")).SkippedDrafter);
        Assert.Equal(0, Hy4FlashWeightMap.Check(Config("base"), Tensors("base")).SkippedDrafter);
    }

    [Fact]
    public void Fp8Checkpoint_EveryFp8WeightHasAnExactScaleGrid()
    {
        List<Hy4FlashTensorInfo> tensors = Tensors("fp8");
        Hy4FlashWeightCoverage coverage = Hy4FlashWeightMap.Check(Config("fp8"), tensors);
        coverage.RequireComplete();
        Assert.Equal(tensors.Count(static t => t.DType == "F8_E4M3" && !t.Name.StartsWith("model.dfly_layers.", StringComparison.Ordinal)),
            coverage.ScaleCompanions);
        Hy4FlashTensorInfo gateUpScale = tensors.Single(t => t.Name == "model.layers.3.mlp.experts.gate_up_proj_scale_inv");
        Assert.Equal([512L, 24, 12], gateUpScale.Shape);
    }

    [Fact]
    public void MissingKey_ExtraKeyAndWrongShape_AreEachReported()
    {
        List<Hy4FlashTensorInfo> tensors = Tensors("instruct");
        Hy4FlashTensorInfo gate = tensors.Single(static t => t.Name == "model.layers.3.mlp.gate.weight");
        tensors.Remove(tensors.Single(static t => t.Name == "model.layers.3.self_attn.o_proj.weight"));
        tensors.Add(new Hy4FlashTensorInfo("model.layers.3.mlp.bogus", "BF16", [1], 2));
        tensors[tensors.IndexOf(gate)] = gate with { Shape = [gate.Shape[0] + 1, gate.Shape[1]] };
        Hy4FlashWeightCoverage coverage = Hy4FlashWeightMap.Check(Config("instruct"), tensors);
        Assert.Contains("model.layers.3.self_attn.o_proj.weight", coverage.Missing);
        Assert.Contains("model.layers.3.mlp.bogus", coverage.Unmapped);
        Assert.Single(coverage.ShapeMismatches);
        Assert.Throws<InvalidDataException>(coverage.RequireComplete);
    }

    [Fact]
    public void Fp8WeightWithoutItsScaleGrid_IsReported()
    {
        List<Hy4FlashTensorInfo> tensors = Tensors("fp8");
        tensors.RemoveAll(static t => t.Name == "model.layers.3.mlp.experts.down_proj_scale_inv");
        Hy4FlashWeightCoverage coverage = Hy4FlashWeightMap.Check(Config("fp8"), tensors);
        Assert.Contains(coverage.ScaleMismatches, static m => m.Contains("model.layers.3.mlp.experts.down_proj", StringComparison.Ordinal));
    }

    [Fact]
    public void ScaleGridBesideABf16Weight_IsReported()
    {
        List<Hy4FlashTensorInfo> tensors = Tensors("instruct");
        tensors.Add(new Hy4FlashTensorInfo("model.layers.3.self_attn.o_proj.weight_scale_inv", "BF16", [24, 128], 6144));
        Assert.Single(Hy4FlashWeightMap.Check(Config("instruct"), tensors).ScaleMismatches);
    }

    [Fact]
    public void Parse_RefusesAnotherModelType()
    {
        string json = System.Text.RegularExpressions.Regex.Replace(File.ReadAllText(Fixture("instruct.config.json")),
            @"""model_type""\s*:\s*""hy_v4_flash"",(\s*)""processor_class""", @"""model_type"": ""other"",$1""processor_class""");
        Assert.Contains("expected 'hy_v4_flash'", Assert.Throws<HartsyInferenceException>(() => Hy4FlashConfig.Parse(json)).Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Validate_RefusesTheReferencesOwnRules()
    {
        Hy4FlashConfig c = Config("instruct");
        Assert.Throws<HartsyInferenceException>(() => (c with { IndexTopk = 2050 }).Validate());
        Assert.Throws<HartsyInferenceException>(() => (c with { MropeSection = [12, 10, 9] }).Validate());
        Assert.Throws<HartsyInferenceException>(() => (c with { EngramPadId = 120818 }).Validate());
        Assert.Throws<HartsyInferenceException>(() => (c with { QuantMethod = "fp8", QuantFormat = "e4m3", QuantBlock = [64, 64] }).Validate());
    }
}
