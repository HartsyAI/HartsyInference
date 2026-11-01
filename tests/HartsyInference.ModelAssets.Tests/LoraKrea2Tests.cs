using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Lora;
using HartsyInference.ModelAssets.Lora.Mappers;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>Krea 2 LoRA support: detection in the ai-toolkit / ComfyUI spelling (<c>diffusion_model.blocks.N.attn.wq</c>, the
/// text-fusion streams), the Wan files that must keep routing to Wan, and a parse of a real safetensors file onto the
/// transformer's diffusers weight names. A Krea file was refused with "matched 0 weights" because the Wan rename table has
/// no <c>attn.wq</c>, so the regression case is a file with only Krea attention names.</summary>
public sealed class LoraKrea2Tests : IDisposable
{
    private readonly string _tempDir = Directory.CreateTempSubdirectory("lora_krea2_").FullName;

    [Theory]
    [InlineData("diffusion_model.blocks.0.attn.wq.lora_A.weight")]
    [InlineData("diffusion_model.blocks.27.attn.gate.lora_B.weight")]
    [InlineData("diffusion_model.blocks.3.attn.wo.lora_down.weight")]
    [InlineData("model.diffusion_model.blocks.5.attn.wv.lora_up.weight")]
    [InlineData("diffusion_model.txtfusion.layerwise_blocks.0.attn.wk.lora_A.weight")]
    [InlineData("diffusion_model.txtfusion.refiner_blocks.1.attn.wo.lora_down.weight")]
    [InlineData("blocks.2.attn.wq.lora_A.weight")]
    public void Detect_ClaimsEveryKrea2AttentionSpelling(string key)
    {
        Assert.Equal(LoraFormat.Krea2, LoraFormatDetector.Detect(Descriptors(key)));
    }

    [Theory]
    // Wan's own attention and FFN names, in both dotted spellings, stay Wan.
    [InlineData("diffusion_model.blocks.0.self_attn.q.lora_A.weight", LoraFormat.DiffusersWan)]
    [InlineData("diffusion_model.blocks.0.ffn.0.lora_A.weight", LoraFormat.DiffusersWan)]
    // A Krea FFN name alone is not distinctive enough to claim the file.
    [InlineData("diffusion_model.blocks.0.mlp.gate.lora_A.weight", LoraFormat.DiffusersWan)]
    public void Detect_LeavesWanFilesOnWan(string key, LoraFormat expected)
    {
        Assert.Equal(expected, LoraFormatDetector.Detect(Descriptors(key)));
    }

    [Fact]
    public void Load_ComfyFile_MapsKreaModulesOntoTheTransformerWeightNames()
    {
        const int rank = 2, dim = 8;
        Dictionary<string, (DType dtype, long[] shape, float[] data)> tensors = new()
        {
            ["diffusion_model.blocks.0.attn.wq.lora_A.weight"] = (DType.F32, [rank, dim], Fill(rank * dim, 0.5f)),
            ["diffusion_model.blocks.0.attn.wq.lora_B.weight"] = (DType.F32, [dim, rank], Fill(dim * rank, 0.25f)),
            ["diffusion_model.blocks.0.attn.gate.lora_A.weight"] = (DType.F32, [rank, dim], Fill(rank * dim, 0.1f)),
            ["diffusion_model.blocks.0.attn.gate.lora_B.weight"] = (DType.F32, [dim, rank], Fill(dim * rank, 0.1f)),
            ["diffusion_model.blocks.0.mlp.gate.lora_A.weight"] = (DType.F32, [rank, dim], Fill(rank * dim, 0.1f)),
            ["diffusion_model.blocks.0.mlp.gate.lora_B.weight"] = (DType.F32, [dim, rank], Fill(dim * rank, 0.1f)),
            ["diffusion_model.txtfusion.layerwise_blocks.0.attn.wk.lora_A.weight"] = (DType.F32, [rank, dim], Fill(rank * dim, 0.1f)),
            ["diffusion_model.txtfusion.layerwise_blocks.0.attn.wk.lora_B.weight"] = (DType.F32, [dim, rank], Fill(dim * rank, 0.1f)),
        };
        string path = CreateSafeTensorsFile("krea2_comfy", tensors);

        using LoraFile file = LoraFile.Load(path);

        Assert.Equal(LoraFormat.Krea2, file.Format);
        Assert.Equal(4, file.Layers.Count);
        LoraLayer query = file.Layers.Single(l => l.TargetKey == "transformer_blocks.0.attn.to_q.weight");
        Assert.Equal(LoraTarget.Transformer, query.Target);
        Assert.Equal(rank, query.Rank);
        Assert.Single(file.Layers, l => l.TargetKey == "transformer_blocks.0.attn.to_gate.weight");
        Assert.Single(file.Layers, l => l.TargetKey == "transformer_blocks.0.ff.gate.weight");
        Assert.Single(file.Layers, l => l.TargetKey == "text_fusion.layerwise_blocks.0.attn.to_k.weight");
    }

    [Fact]
    public void Load_FileWithNoCompletePair_IsRefusedNotLoadedEmpty()
    {
        // The detector claims this file by its one Krea key; with no up matrix there is no layer, and an empty LoRA would change nothing.
        Dictionary<string, (DType dtype, long[] shape, float[] data)> tensors = new()
        {
            ["diffusion_model.blocks.0.attn.wq.lora_A.weight"] = (DType.F32, [2, 8], Fill(16, 0.5f)),
        };
        string path = CreateSafeTensorsFile("krea2_half_pair", tensors);

        NotSupportedException ex = Assert.Throws<NotSupportedException>(() => LoraFile.Load(path));
        Assert.Contains("no complete block-module layers", ex.Message);
    }

    [Fact]
    public void Load_FullWeightDiffBesideKreaLayers_IsRefusedByName()
    {
        Dictionary<string, (DType dtype, long[] shape, float[] data)> tensors = new()
        {
            ["diffusion_model.blocks.0.attn.wq.lora_A.weight"] = (DType.F32, [2, 8], Fill(16, 0.5f)),
            ["diffusion_model.blocks.0.attn.wq.lora_B.weight"] = (DType.F32, [8, 2], Fill(16, 0.5f)),
            ["diffusion_model.blocks.0.attn.wo.diff"] = (DType.F32, [8, 8], Fill(64, 0.5f)),
        };
        string path = CreateSafeTensorsFile("krea2_diff", tensors);

        NotSupportedException ex = Assert.Throws<NotSupportedException>(() => LoraFile.Load(path));
        Assert.Contains("full-weight diff", ex.Message);
    }

    [Theory]
    [InlineData("diffusion_model.blocks.0.attn.wq.lora_down.weight", true)]
    [InlineData("diffusion_model.blocks.0.attn.qknorm.lora_down.weight", false)]
    [InlineData("diffusion_model.blocks.0.self_attn.q_proj.lora_down.weight", false)]
    public void IsDistinctive_OnlyClaimsKreaAttentionModules(string key, bool expected)
    {
        Assert.Equal(expected, Krea2LoraMapper.IsDistinctive(key));
    }

    private static Dictionary<string, SafeTensorDescriptor> Descriptors(params string[] keys)
    {
        Dictionary<string, SafeTensorDescriptor> descriptors = [];
        foreach (string key in keys)
        {
            descriptors[key] = new SafeTensorDescriptor
            {
                Name = key,
                DType = DType.F16,
                Shape = new TensorShape(2, 2),
                DataOffset = 0,
                ByteLength = 8,
            };
        }
        return descriptors;
    }

    private static float[] Fill(int count, float value)
    {
        float[] data = new float[count];
        Array.Fill(data, value);
        return data;
    }

    private string CreateSafeTensorsFile(string name, Dictionary<string, (DType dtype, long[] shape, float[] data)> tensors)
    {
        using MemoryStream dataStream = new();
        Dictionary<string, (long start, long end)> offsets = [];
        foreach (KeyValuePair<string, (DType dtype, long[] shape, float[] data)> kvp in tensors)
        {
            long start = dataStream.Position;
            foreach (float val in kvp.Value.data)
            {
                byte[] bytes = BitConverter.GetBytes(val);
                dataStream.Write(bytes, 0, bytes.Length);
            }
            offsets[kvp.Key] = (start, dataStream.Position);
        }
        byte[] dataBlob = dataStream.ToArray();

        Dictionary<string, object> headerDict = [];
        foreach (KeyValuePair<string, (DType dtype, long[] shape, float[] data)> kvp in tensors)
        {
            (long start, long end) = offsets[kvp.Key];
            headerDict[kvp.Key] = new Dictionary<string, object>
            {
                ["dtype"] = kvp.Value.dtype.Name,
                ["shape"] = kvp.Value.shape,
                ["data_offsets"] = new long[] { start, end },
            };
        }
        byte[] headerBytes = System.Text.Json.JsonSerializer.SerializeToUtf8Bytes(headerDict);
        using MemoryStream file = new();
        file.Write(BitConverter.GetBytes((long)headerBytes.Length));
        file.Write(headerBytes);
        file.Write(dataBlob);

        string path = Path.Combine(_tempDir, name + ".safetensors");
        File.WriteAllBytes(path, file.ToArray());
        return path;
    }

    public void Dispose()
    {
        if (Directory.Exists(_tempDir))
        {
            Directory.Delete(_tempDir, true);
        }
    }
}
