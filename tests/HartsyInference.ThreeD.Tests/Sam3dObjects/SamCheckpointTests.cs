using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ThreeD.Models.Sam3dObjects;
using Xunit;

namespace HartsyInference.ThreeD.Tests.Sam3dObjects;

public sealed class SamCheckpointTests
{
    private static readonly string[] Prefixed =
    [
        "state_dict._base_models.generator.blocks.0.weight",
        "state_dict._base_models.generator.final.bias",
        "state_dict._base_models.condition_embedder.idx_emb",
        "state_dict.epoch_counter",
    ];

    [Fact]
    public void Select_GeneratorStripsEnvelopeAndPrefixAndDropsEmbedderKeys()
    {
        Dictionary<string, string> map = SamCheckpointKeys.Select(Prefixed, SamCheckpointSection.Generator);

        Assert.Equal(2, map.Count);
        Assert.Equal("state_dict._base_models.generator.blocks.0.weight", map["blocks.0.weight"]);
        Assert.False(map.ContainsKey("idx_emb"));
    }

    [Fact]
    public void Select_ConditionEmbedderTakesOnlyItsPrefix()
    {
        Dictionary<string, string> map = SamCheckpointKeys.Select(Prefixed, SamCheckpointSection.ConditionEmbedder);

        Assert.Equal(["idx_emb"], map.Keys);
    }

    [Fact]
    public void Select_PrefixedKeysWithoutTheEnvelopeWorkToo()
    {
        Dictionary<string, string> map = SamCheckpointKeys.Select(
            ["_base_models.generator.a.weight", "_base_models.condition_embedder.b.weight"], SamCheckpointSection.Generator);

        Assert.Equal(["a.weight"], map.Keys);
    }

    [Fact]
    public void Select_AnAlreadyStrippedConversionYieldsEveryKeyAsGeneratorAndNothingAsEmbedder()
    {
        string[] stripped = ["blocks.0.weight", "final.bias"];

        Assert.Equal(2, SamCheckpointKeys.Select(stripped, SamCheckpointSection.Generator).Count);
        Assert.Empty(SamCheckpointKeys.Select(stripped, SamCheckpointSection.ConditionEmbedder));
    }

    [Fact]
    public void Select_RootKeepsEveryKeyAndStripsOnlyTheEnvelope()
    {
        Dictionary<string, string> map = SamCheckpointKeys.Select(["state_dict.conv.weight", "conv.bias"], SamCheckpointSection.Root);

        Assert.Equal("state_dict.conv.weight", map["conv.weight"]);
        Assert.Equal("conv.bias", map["conv.bias"]);
    }

    [Fact]
    public void Load_ReadsASafetensorsConversionAndSelectsSections()
    {
        string path = Path.Combine(Path.GetTempPath(), $"sam3d-test-{Guid.NewGuid():N}.safetensors");
        Dictionary<string, Tensor> tensors = [];
        try
        {
            foreach (string key in new[] { "_base_models.generator.w", "_base_models.condition_embedder.idx_emb" })
            {
                Tensor t = new(new TensorShape(2), DType.F32);
                t.AsSpan<float>()[0] = key.Length;
                tensors[key] = t;
            }

            SafeTensorsWriter.Save(path, tensors);

            using SamCheckpoint checkpoint = SamCheckpoint.Load(path);
            Dictionary<string, Tensor> generator = checkpoint.Section(SamCheckpointSection.Generator);
            Dictionary<string, Tensor> embedder = checkpoint.Section(SamCheckpointSection.ConditionEmbedder);

            Assert.True(checkpoint.WasSafeTensors);
            Assert.Equal(["w"], generator.Keys);
            Assert.Equal(["idx_emb"], embedder.Keys);
            Assert.Equal("_base_models.generator.w".Length, generator["w"].AsReadOnlySpan<float>()[0]);
        }
        finally
        {
            foreach (Tensor t in tensors.Values)
                t.Dispose();
            File.Delete(path);
        }
    }

    [Fact]
    public void Load_MissingFileAndEmptySectionFailWithClearErrors()
    {
        Assert.Throws<FileNotFoundException>(() => SamCheckpoint.Load(Path.Combine(Path.GetTempPath(), "sam3d-missing.ckpt")));

        string path = Path.Combine(Path.GetTempPath(), $"sam3d-test-{Guid.NewGuid():N}.safetensors");
        Tensor t = new(new TensorShape(1), DType.F32);
        try
        {
            SafeTensorsWriter.Save(path, new Dictionary<string, Tensor> { ["plain.weight"] = t });
            using SamCheckpoint checkpoint = SamCheckpoint.Load(path);

            Assert.Throws<InvalidOperationException>(() => checkpoint.Section(SamCheckpointSection.ConditionEmbedder));
        }
        finally
        {
            t.Dispose();
            File.Delete(path);
        }
    }
}
