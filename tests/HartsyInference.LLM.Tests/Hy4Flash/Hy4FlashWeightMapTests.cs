using System.IO.Compression;
using System.Text.Json;
using System.Text.Json.Nodes;
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
    public void TinyConfigOfTheReferenceDump_ParsesAndYieldsEveryKindOfLayer()
    {
        Hy4FlashConfig c = Hy4FlashConfig.Load(Fixture("tiny_config.json"));
        Assert.Equal([Hy4FlashLayerKind.Indexed, Hy4FlashLayerKind.Indexed, Hy4FlashLayerKind.Sliding], c.LayerKinds);
        Assert.Equal([false, true, true], c.SparseFfn);
        Assert.Equal([1], Enumerable.Range(0, 3).Where(c.HasEngram));
        Assert.Equal(4, c.IndexKpool);
        Assert.Equal(16, c.IndexTopk);
        Assert.Equal(8, c.HeadsFor(2));
        Assert.Equal(4, c.EngramHashHeads);
        IReadOnlyList<Hy4FlashExpectedTensor> keys = Hy4FlashWeightMap.Expected(c);
        Assert.Contains(keys, static k => k.Key == "model.layers.1.engram_layer.multi_head_embedding.embedding.weight" && k.Shape[0] == 101 + 103 + 107 + 109);
        Assert.Contains(keys, static k => k.Key == "model.layers.2.self_attn.o_proj.weight" && k.Shape[0] == 48 && k.Shape[1] == 64);
        Assert.DoesNotContain(keys, static k => k.Key.Contains("layers.2.self_attn.compressed_indexer", StringComparison.Ordinal));
    }

    [Fact]
    public void Validate_RefusesSlidingHeadWidthsThatDifferFromTheIndexedOnes()
    {
        Hy4FlashConfig c = Config("instruct");
        Assert.Throws<HartsyInferenceException>(() => (c with { SlidingVHeadDim = 128 }).Validate());
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

    [Fact]
    public void WrongStorageDtype_IsReported_ForTheF32AndI32Tensors()
    {
        List<Hy4FlashTensorInfo> tensors = Tensors("instruct");
        int hc = tensors.FindIndex(static t => t.Name == "model.layers.3.hc_mlp_layer.hc_pre.hc_fn");
        tensors[hc] = tensors[hc] with { DType = "BF16" };
        int table = tensors.FindIndex(static t => t.Name == "model.layers.1.engram_layer.lookup_table");
        tensors[table] = tensors[table] with { DType = "F32" };
        Hy4FlashWeightCoverage coverage = Hy4FlashWeightMap.Check(Config("instruct"), tensors);
        Assert.Equal(2, coverage.DtypeMismatches.Count);
        Assert.Throws<InvalidDataException>(coverage.RequireComplete);
    }

    [Fact]
    public void ScaleGridOfAnotherDtype_IsReported()
    {
        List<Hy4FlashTensorInfo> tensors = Tensors("fp8");
        int scale = tensors.FindIndex(static t => t.Name == "model.layers.3.mlp.experts.gate_up_proj_scale_inv");
        tensors[scale] = tensors[scale] with { DType = "F16" };
        Assert.Single(Hy4FlashWeightMap.Check(Config("fp8"), tensors).ScaleMismatches);
    }

    [Fact]
    public void MalformedConfigValue_IsRefusedWithAHartsyInferenceException()
    {
        JsonNode root = JsonNode.Parse(File.ReadAllText(Fixture("instruct.config.json")))!;
        root["text_config"]!["num_attention_heads"] = "thirty-two";
        Assert.Throws<HartsyInferenceException>(() => Hy4FlashConfig.Parse(root.ToJsonString()));
        JsonNode layers = JsonNode.Parse(File.ReadAllText(Fixture("instruct.config.json")))!;
        layers["text_config"]!["layer_types"]![3] = 5;
        Assert.Throws<HartsyInferenceException>(() => Hy4FlashConfig.Parse(layers.ToJsonString()));
    }

    [Fact]
    public void ExtraValidationRules_AreEnforced()
    {
        Hy4FlashConfig c = Config("instruct");
        Assert.Throws<HartsyInferenceException>(() => (c with { NumExpertsPerTok = c.NRoutedExperts + 1 }).Validate());
        Assert.Throws<HartsyInferenceException>(() => (c with { TopkGroup = c.NGroup + 1, NGroup = c.NGroup }).Validate());
        Assert.Throws<HartsyInferenceException>(() => (c with { NRoutedExperts = 510, NGroup = 4 }).Validate());
        Assert.Throws<HartsyInferenceException>(() => (c with { TieWordEmbeddings = true }).Validate());
        Assert.Throws<HartsyInferenceException>(() => (c with { Vision = c.Vision with { MaxImageSize = 2050 } }).Validate());
    }
}
