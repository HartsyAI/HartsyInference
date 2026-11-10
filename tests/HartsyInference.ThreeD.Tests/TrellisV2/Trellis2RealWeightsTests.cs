using System.Text.Json;
using HartsyInference.ThreeD.Models.TrellisV2;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.ThreeD.Tests.TrellisV2;

/// <summary>Header-only checks of the downloaded TRELLIS.2 files against the key maps: no tensor data is read, so these run in seconds.
/// Integration lane; skipped, not passed, when the files are absent. They show the key names, shapes and dtypes match; they are not parity.</summary>
[Trait("Category", "Integration")]
public sealed class Trellis2RealWeightsTests
{
    private readonly ITestOutputHelper _output;

    public Trellis2RealWeightsTests(ITestOutputHelper output) => _output = output;

    [Trellis2RealWeightsFact]
    public void FlowHeaders_MatchTheKeyMapExactly()
    {
        string folder = Trellis2RealWeightsFactAttribute.FindFolder()!;
        foreach (string name in new[]
        {
            Trellis2ModelFiles.StructureFlowFile, Trellis2ModelFiles.ShapeFlow512File, Trellis2ModelFiles.ShapeFlow1024File,
        })
        {
            string path = Path.Combine(folder, name);
            Trellis2FlowConfig config = Trellis2FlowConfig.FromJson(File.ReadAllText(Path.ChangeExtension(path, ".json")));
            Dictionary<string, (string DType, long[] Shape)> header = ReadHeader(path);
            IReadOnlyList<ShapeVaeKeyMap.TensorSpec> specs = Trellis2FlowKeyMap.Specs(config);

            Assert.Equal(specs.Select(s => s.CheckpointKey).Order(StringComparer.Ordinal), header.Keys.Order(StringComparer.Ordinal));
            foreach (ShapeVaeKeyMap.TensorSpec spec in specs)
            {
                Assert.Equal(spec.Shape, header[spec.CheckpointKey].Shape);
                Assert.Equal("BF16", header[spec.CheckpointKey].DType);
            }

            _output.WriteLine($"{name}: {specs.Count} tensors match ({(config.Dense ? "dense" : "sparse")}, resolution {config.Resolution}).");
        }
    }

    [Trellis2RealWeightsFact]
    public void StructureDecoderHeader_HoldsTheKeysTheV1DecoderReads()
    {
        // SparseStructureDecoder.LoadWeights reads: input_layer, two middle res-blocks, eight blocks (res-blocks except the upsamples at
        // 2 and 5, which hold one conv) and out_layer.0 / out_layer.2.
        string path = Path.Combine(Trellis2RealWeightsFactAttribute.FindFolder()!, Trellis2ModelFiles.StructureDecoderFile);
        List<string> expected =
        [
            "input_layer.weight", "input_layer.bias", "out_layer.0.weight", "out_layer.0.bias", "out_layer.2.weight", "out_layer.2.bias",
        ];
        foreach (string block in new[] { "middle_block.0", "middle_block.1", "blocks.0", "blocks.1", "blocks.3", "blocks.4", "blocks.6", "blocks.7" })
        {
            foreach (string part in new[] { "norm1", "conv1", "norm2", "conv2" })
                expected.AddRange([$"{block}.{part}.weight", $"{block}.{part}.bias"]);
        }

        foreach (string block in new[] { "blocks.2", "blocks.5" })
            expected.AddRange([$"{block}.conv.weight", $"{block}.conv.bias"]);

        Dictionary<string, (string DType, long[] Shape)> header = ReadHeader(path);

        Assert.Equal(expected.Order(StringComparer.Ordinal), header.Keys.Order(StringComparer.Ordinal));
        Assert.Equal([512L, 8, 3, 3, 3], header["input_layer.weight"].Shape);
        Assert.Equal([1L, 32, 3, 3, 3], header["out_layer.2.weight"].Shape);
    }

    [Trellis2RealWeightsFact]
    public void ShapeDecoderHeader_MatchesTheShapeVaeKeyMap()
    {
        string folder = Trellis2RealWeightsFactAttribute.FindFolder()!;
        string path = Trellis2ModelFiles.Resolve(folder).ShapeDecoder;
        Dictionary<string, (string DType, long[] Shape)> header = ReadHeader(path);

        IReadOnlyList<ShapeVaeKeyMap.TensorSpec> specs = ShapeVaeKeyMap.Decoder(ShapeVaeConfig.NextDcF16C32.Decoder);

        Assert.Equal(specs.Select(s => s.CheckpointKey).Order(StringComparer.Ordinal), header.Keys.Order(StringComparer.Ordinal));
        foreach (ShapeVaeKeyMap.TensorSpec spec in specs)
            Assert.Equal(spec.Shape, header[spec.CheckpointKey].Shape);
    }

    [Trellis2RealWeightsFact]
    public void ModelFiles_ResolveEveryFileAndPipelineJsonParses()
    {
        Trellis2ModelFiles files = Trellis2ModelFiles.Resolve(Trellis2RealWeightsFactAttribute.FindFolder()!);

        Trellis2PipelineConfig config = Trellis2PipelineConfig.Load(files.Pipeline);

        Assert.True(File.Exists(files.Dino));
        Assert.True(File.Exists(files.ShapeDecoder));
        Assert.Equal(12, config.SparseStructureSampler.Steps);
        Assert.Equal(0.781296f, config.ShapeLatentMean[0]);
        _output.WriteLine($"dino={files.Dino}\nshape decoder={files.ShapeDecoder}");
    }

    // Reads only the JSON header: 8 little-endian bytes of length, then that many bytes of JSON.
    private static Dictionary<string, (string DType, long[] Shape)> ReadHeader(string path)
    {
        using FileStream stream = File.OpenRead(path);
        byte[] length = new byte[8];
        stream.ReadExactly(length);
        byte[] json = new byte[BitConverter.ToInt64(length)];
        stream.ReadExactly(json);
        using JsonDocument doc = JsonDocument.Parse(json);
        Dictionary<string, (string, long[])> header = [];
        foreach (JsonProperty property in doc.RootElement.EnumerateObject())
        {
            if (property.Name == "__metadata__")
                continue;
            header[property.Name] = (
                property.Value.GetProperty("dtype").GetString()!,
                property.Value.GetProperty("shape").EnumerateArray().Select(e => e.GetInt64()).ToArray());
        }

        return header;
    }
}
