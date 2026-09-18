using System.Text;
using System.Text.Json;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Lora;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>Pins that a LyCORIS or DoRA file loads for every family the ordinary mappers already know, with no
/// per-format code. Before the shared <see cref="LoraRoleSuffix"/> table no detector or mapper mentioned
/// <c>hada_w1_a</c>, <c>lokr_w1</c> or <c>dora_scale</c> at all, so such a file was rejected outright — as an
/// undetectable format when it carried no other marker, and as the wrong family's LoRA when it did.</summary>
public sealed class LoraLycorisDetectionTests : IDisposable
{
    private readonly string _tempDir = Path.Combine(Path.GetTempPath(), $"lora-lycoris-{Guid.NewGuid():N}");

    public LoraLycorisDetectionTests() => Directory.CreateDirectory(_tempDir);

    [Fact]
    public void KohyaSdxlLoHa_DetectsAsKohyaSdxlAndParsesBothComponents()
    {
        const string UnetRoot = "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q";
        const string TextRoot = "lora_te2_text_model_encoder_layers_0_self_attn_q_proj";
        Dictionary<string, (DType dtype, long[] shape, float[] data)> tensors = new(StringComparer.Ordinal);
        AddLoHa(tensors, UnetRoot, outDim: 4, inDim: 4, rank: 2);
        AddLoHa(tensors, TextRoot, outDim: 4, inDim: 4, rank: 2);
        string path = CreateSafeTensorsFile(_tempDir, "kohya_sdxl_loha", tensors);

        using LoraFile file = LoraFile.Load(path);

        Assert.Equal(LoraFormat.KohyaSdxl, file.Format);
        Assert.Equal(2, file.Layers.Count);
        Assert.All(file.Layers, layer => Assert.Equal(LoraVariant.LoHa, layer.Variant));
        LoraLayer unet = Assert.Single(file.Layers, layer => layer.Target == LoraTarget.UNet);
        Assert.Equal("down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight", unet.TargetKey);
        LoHaDelta delta = Assert.IsType<LoHaDelta>(unet.Delta);
        Assert.Equal(2, delta.Rank);
        Assert.Equal(4, delta.OutFeatures);
        Assert.Equal(4, delta.InFeatures);
        Assert.Contains(file.Layers, layer => layer.Target == LoraTarget.ClipG);
    }

    [Fact]
    public void BareRootLoKr_DetectsAsBareDitAndParsesTheKroneckerFactors()
    {
        Dictionary<string, (DType dtype, long[] shape, float[] data)> tensors = new(StringComparer.Ordinal)
        {
            ["blocks.0.attn.to_q.lokr_w1"] = (DType.F32, [2, 2], [1, 2, 3, 4]),
            ["blocks.0.attn.to_q.lokr_w2"] = (DType.F32, [3, 3], [1, 0, 0, 0, 1, 0, 0, 0, 1]),
            ["blocks.0.attn.to_q.alpha"] = (DType.F32, [1], [8]),
        };
        string path = CreateSafeTensorsFile(_tempDir, "bare_lokr", tensors);

        using LoraFile file = LoraFile.Load(path);

        Assert.Equal(LoraFormat.DiffusersBareDit, file.Format);
        LoraLayer layer = Assert.Single(file.Layers);
        Assert.Equal("blocks.0.attn.to_q.weight", layer.TargetKey);
        Assert.Equal(LoraVariant.LoKr, layer.Variant);
        LoKrDelta delta = Assert.IsType<LoKrDelta>(layer.Delta);
        Assert.Equal(6, delta.OutFeatures);
        Assert.Equal(6, delta.InFeatures);
    }

    [Fact]
    public void DiffusersDoraScale_RidesTheStandardPairAndKeepsItsMagnitudeVector()
    {
        Dictionary<string, (DType dtype, long[] shape, float[] data)> tensors = new(StringComparer.Ordinal)
        {
            ["transformer.blocks.0.attn.to_q.lora_A.weight"] = (DType.F32, [2, 4], new float[8]),
            ["transformer.blocks.0.attn.to_q.lora_B.weight"] = (DType.F32, [4, 2], new float[8]),
            ["transformer.blocks.0.attn.to_q.dora_scale"] = (DType.F32, [4, 1], [1, 2, 3, 4]),
        };
        string path = CreateSafeTensorsFile(_tempDir, "diffusers_dora", tensors);

        using LoraFile file = LoraFile.Load(path);

        Assert.Equal(LoraFormat.DiffusersFlux, file.Format);
        LoraLayer layer = Assert.Single(file.Layers);
        Assert.Equal(LoraVariant.DoRA, layer.Variant);
        StandardLoraDelta delta = Assert.IsType<StandardLoraDelta>(layer.Delta);
        Assert.NotNull(delta.DoraScale);
        Assert.Equal(4, delta.DoraScale!.Shape.ElementCount);
    }

    [Fact]
    public void IncompleteLoHaGroup_IsSkippedRatherThanBuiltFromMissingFactors()
    {
        const string Root = "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q";
        Dictionary<string, (DType dtype, long[] shape, float[] data)> tensors = new(StringComparer.Ordinal);
        AddLoHa(tensors, Root, outDim: 4, inDim: 4, rank: 2);
        tensors.Remove($"{Root}.hada_w2_b");
        string path = CreateSafeTensorsFile(_tempDir, "partial_loha", tensors);

        using LoraFile file = LoraFile.Load(path);

        Assert.Equal(LoraFormat.KohyaSd15, file.Format);
        Assert.Empty(file.Layers);
    }

    [Fact]
    public void LycorisOnlyFile_IsNoLongerAnUndetectableFormat()
    {
        Dictionary<string, (DType dtype, long[] shape, float[] data)> tensors = new(StringComparer.Ordinal);
        AddLoHa(tensors, "transformer.transformer_blocks.0.attn.to_q", outDim: 4, inDim: 4, rank: 2);
        string path = CreateSafeTensorsFile(_tempDir, "lycoris_only", tensors);

        using LoraFile file = LoraFile.Load(path);

        Assert.Equal(LoraFormat.DiffusersFlux, file.Format);
        Assert.Equal(LoraVariant.LoHa, Assert.Single(file.Layers).Variant);
    }

    public void Dispose()
    {
        if (Directory.Exists(_tempDir))
        {
            Directory.Delete(_tempDir, recursive: true);
        }
    }

    private static void AddLoHa(Dictionary<string, (DType dtype, long[] shape, float[] data)> tensors,
        string root, int outDim, int inDim, int rank)
    {
        tensors[$"{root}.hada_w1_a"] = (DType.F32, [outDim, rank], Ramp(outDim * rank, 1.0f));
        tensors[$"{root}.hada_w1_b"] = (DType.F32, [rank, inDim], Ramp(rank * inDim, -1.0f));
        tensors[$"{root}.hada_w2_a"] = (DType.F32, [outDim, rank], Ramp(outDim * rank, 2.0f));
        tensors[$"{root}.hada_w2_b"] = (DType.F32, [rank, inDim], Ramp(rank * inDim, 0.5f));
        tensors[$"{root}.alpha"] = (DType.F32, [1], [rank]);
    }

    private static float[] Ramp(int count, float start)
    {
        float[] values = new float[count];
        for (int i = 0; i < count; i++)
        {
            values[i] = start + i;
        }
        return values;
    }

    private static string CreateSafeTensorsFile(string dir, string name,
        Dictionary<string, (DType dtype, long[] shape, float[] data)> tensors)
    {
        using MemoryStream dataStream = new();
        Dictionary<string, (long start, long end)> offsets = [];
        foreach (KeyValuePair<string, (DType dtype, long[] shape, float[] data)> entry in tensors)
        {
            long start = dataStream.Position;
            foreach (float value in entry.Value.data)
            {
                dataStream.Write(BitConverter.GetBytes(value), 0, sizeof(float));
            }
            offsets[entry.Key] = (start, dataStream.Position);
        }
        byte[] dataBlob = dataStream.ToArray();

        Dictionary<string, object> headerDict = [];
        foreach (KeyValuePair<string, (DType dtype, long[] shape, float[] data)> entry in tensors)
        {
            (long start, long end) = offsets[entry.Key];
            headerDict[entry.Key] = new Dictionary<string, object>
            {
                ["dtype"] = entry.Value.dtype.Name,
                ["shape"] = entry.Value.shape,
                ["data_offsets"] = new long[] { start, end },
            };
        }

        byte[] headerBytes = Encoding.UTF8.GetBytes(JsonSerializer.Serialize(headerDict));
        string filePath = Path.Combine(dir, $"{name}.safetensors");
        using FileStream stream = new(filePath, FileMode.Create, FileAccess.Write);
        using BinaryWriter writer = new(stream);
        writer.Write((long)headerBytes.Length);
        writer.Write(headerBytes);
        writer.Write(dataBlob);
        return filePath;
    }
}
