using System.Text.Json;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.DeepSeekV41;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.LLM.Tests.DeepSeekV41;

/// <summary>The committed upstream vision-tower fixture (<c>vision_tower.json</c>, written by <c>dump_vision_fixtures.py</c>) and the weights, grids and comparisons the vision tests share.</summary>
internal static class DeepSeekV41VisionFixture
{
    /// <summary>Per-stage relative L2 bound for float32 against float32; the same bound the real-weight run is held to by project plan, applied here at stage granularity.</summary>
    public const double RelL2Tolerance = 1e-4;

    private static readonly JsonElement _root = JsonDocument.Parse(File.ReadAllText(Path.Combine(
        RepoRoot.Path, "tests", "python-reference", "deepseek_v41", "fixtures", "vision_tower.json"))).RootElement;

    public static JsonElement Root => _root;

    /// <summary>The fixture's language-model width, which the aligner emits.</summary>
    public static int OutputDim => _root.GetProperty("config").GetProperty("dim").GetInt32();

    public static DeepSeekV41VisionConfig Config()
    {
        JsonElement c = _root.GetProperty("config");
        return new DeepSeekV41VisionConfig(c.GetProperty("vision_n_layers").GetInt32(), c.GetProperty("vision_dim").GetInt32(),
            c.GetProperty("vision_n_heads").GetInt32(), c.GetProperty("vision_inter_dim").GetInt32(), c.GetProperty("vision_patch_size").GetInt32(),
            c.GetProperty("vision_downsample_ratio").GetInt32(), c.GetProperty("vision_rope_theta").GetDouble());
    }

    public static float[] Floats(JsonElement e) => e.EnumerateArray().Select(static v => v.GetSingle()).ToArray();

    public static IEnumerable<string> ParamKeys() => _root.GetProperty("params").EnumerateObject().Select(static p => p.Name);

    public static float[] Param(string key) => Floats(_root.GetProperty("params").GetProperty(key));

    public static long[] Shape(string key) =>
        _root.GetProperty("shapes").GetProperty(key).EnumerateArray().Select(static v => v.GetInt64()).ToArray();

    public static Tensor ParamTensor(string key, Func<string, float[]>? source = null) =>
        DeepSeekV41HostMath.Tensor((source ?? Param)(key), Shape(key));

    /// <summary>The tower weights as owned tensors; <paramref name="source"/> replaces the values of a key (same shape) for tests that need other numbers.</summary>
    public static DeepSeekV41VisionWeights TowerWeights(Func<string, float[]>? source = null)
    {
        DeepSeekV41VisionConfig config = Config();
        List<DeepSeekV41VisionBlockWeights> blocks = [];
        for (int i = 0; i < config.NumLayers; i++)
        {
            string b = $"vision.blocks.{i}.";
            blocks.Add(new DeepSeekV41VisionBlockWeights(ParamTensor(b + "norm1.weight", source), ParamTensor(b + "attn.wqkv.weight", source),
                ParamTensor(b + "attn.wqkv.bias", source), ParamTensor(b + "attn.wo.weight", source), ParamTensor(b + "attn.wo.bias", source),
                ParamTensor(b + "norm2.weight", source), ParamTensor(b + "mlp.w1.weight", source), ParamTensor(b + "mlp.w2.weight", source)));
        }
        return new DeepSeekV41VisionWeights(ParamTensor("vision.patch_embed.proj.weight", source),
            ParamTensor("vision.patch_embed.proj.bias", source), blocks, ParamTensor("vision.norm.weight", source));
    }

    public static DeepSeekV41AlignerWeights AlignerWeights(Func<string, float[]>? source = null) =>
        new(ParamTensor("aligner.w1.weight", source), ParamTensor("aligner.w1.bias", source), ParamTensor("aligner.w2.weight", source),
            ParamTensor("aligner.w2.bias", source));

    public static JsonElement TowerCase(int gridHeight, int gridWidth) => Case("towerCases", gridHeight, gridWidth);

    public static JsonElement AlignerCase(int gridHeight, int gridWidth) => Case("alignerCases", gridHeight, gridWidth);

    public static JsonElement RopeCase(int gridHeight, int gridWidth, int ropeDim) =>
        _root.GetProperty("ropeCases").EnumerateArray().Single(c => c.GetProperty("gridHeight").GetInt32() == gridHeight
            && c.GetProperty("gridWidth").GetInt32() == gridWidth && c.GetProperty("ropeDim").GetInt32() == ropeDim);

    /// <summary>Requires <paramref name="actual"/> to match <paramref name="expected"/> within <see cref="RelL2Tolerance"/> overall and, element by element, within the same fraction of the stage's largest value.</summary>
    public static void AssertClose(ITestOutputHelper output, string what, float[] actual, float[] expected)
    {
        Assert.Equal(expected.Length, actual.Length);
        Assert.All(actual, static v => Assert.True(float.IsFinite(v)));
        double rel = DeepSeekV41VisionMetrics.RelL2(actual, expected), abs = DeepSeekV41VisionMetrics.MaxAbsDiff(actual, expected);
        double scale = Math.Max(1.0, DeepSeekV41VisionMetrics.MaxAbs(expected));
        output.WriteLine($"{what}: relL2 {rel:E2} maxAbs {abs:E2} (stage max {scale:F2})");
        Assert.True(rel <= RelL2Tolerance, $"{what}: relL2 {rel:E3} > {RelL2Tolerance:E1}");
        Assert.True(abs <= RelL2Tolerance * scale, $"{what}: maxAbs {abs:E3} > {RelL2Tolerance * scale:E3}");
    }

    private static JsonElement Case(string list, int gridHeight, int gridWidth) =>
        _root.GetProperty(list).EnumerateArray().Single(c =>
            c.GetProperty("gridHeight").GetInt32() == gridHeight && c.GetProperty("gridWidth").GetInt32() == gridWidth);
}
