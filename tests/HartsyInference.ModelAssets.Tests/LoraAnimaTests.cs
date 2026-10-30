using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Lora;
using HartsyInference.ModelAssets.Lora.Mappers;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>Anima LoRA support: detection in each spelling (ComfyUI <c>diffusion_model.</c>, the checkpoint's <c>net.</c>
/// root, bare <c>blocks.</c>, kohya <c>lora_unet_</c>), the Wan files that must keep routing to Wan, and a parse of a real
/// safetensors file onto the checkpoint's own DiT weight names. The Anima adapter's modules are the regression case: its
/// <c>self_attn.output_proj</c> and <c>mlp.layer1</c> keys were silently mangled by the Wan rename table.</summary>
public sealed class LoraAnimaTests : IDisposable
{
    private readonly string _tempDir = Directory.CreateTempSubdirectory("lora_anima_").FullName;

    [Theory]
    [InlineData("diffusion_model.blocks.0.self_attn.q_proj.lora_down.weight")]
    [InlineData("diffusion_model.blocks.27.cross_attn.output_proj.lora_up.weight")]
    [InlineData("diffusion_model.blocks.3.mlp.layer1.lora_A.weight")]
    [InlineData("diffusion_model.blocks.9.mlp.layer2.lora_B.weight")]
    [InlineData("diffusion_model.blocks.4.adaln_modulation_cross_attn.1.lora_down.weight")]
    [InlineData("model.diffusion_model.blocks.0.self_attn.v_proj.lora_up.weight")]
    [InlineData("net.blocks.5.cross_attn.k_proj.lora_down.weight")]
    [InlineData("blocks.2.self_attn.output_proj.lora_down.weight")]
    public void Detect_ClaimsEveryAnimaDottedSpelling(string key)
    {
        Assert.Equal(LoraFormat.Anima, LoraFormatDetector.Detect(Descriptors(key)));
    }

    [Theory]
    [InlineData("lora_unet_blocks_0_self_attn_q_proj.lora_down.weight")]
    [InlineData("lora_unet_blocks_12_mlp_layer1.lora_up.weight")]
    [InlineData("lora_unet_blocks_3_adaln_modulation_self_attn_2.lora_down.weight")]
    public void Detect_RoutesKohyaAnimaSpellingToTheGenericKohyaMapper(string key)
    {
        // Resolved against the transformer's own weight names when applied, so no per-family rename is needed.
        Assert.Equal(LoraFormat.KohyaGeneric, LoraFormatDetector.Detect(Descriptors(key)));
    }

    [Theory]
    // Wan's own attention and FFN names, in both dotted spellings, stay Wan.
    [InlineData("diffusion_model.blocks.0.self_attn.q.lora_A.weight", LoraFormat.DiffusersWan)]
    [InlineData("diffusion_model.blocks.0.ffn.0.lora_A.weight", LoraFormat.DiffusersWan)]
    [InlineData("lora_unet_blocks_0_self_attn_q.lora_down.weight", LoraFormat.KohyaWan)]
    // A module Anima does not own is not claimed by the Anima arm.
    [InlineData("diffusion_model.blocks.0.self_attn.out.lora_down.weight", LoraFormat.DiffusersWan)]
    public void Detect_LeavesWanFilesOnWan(string key, LoraFormat expected)
    {
        Assert.Equal(expected, LoraFormatDetector.Detect(Descriptors(key)));
    }

    [Fact]
    public void Load_ComfyFile_MapsAnimaModulesOntoTheCheckpointWeightNames()
    {
        const int rank = 2, dim = 8;
        Dictionary<string, (DType dtype, long[] shape, float[] data)> tensors = new()
        {
            ["diffusion_model.blocks.0.self_attn.q_proj.lora_down.weight"] = (DType.F32, [rank, dim], Fill(rank * dim, 0.5f)),
            ["diffusion_model.blocks.0.self_attn.q_proj.lora_up.weight"] = (DType.F32, [dim, rank], Fill(dim * rank, 0.25f)),
            ["diffusion_model.blocks.0.self_attn.q_proj.alpha"] = (DType.F32, [1], [1.0f]),
            ["diffusion_model.blocks.0.self_attn.output_proj.lora_down.weight"] = (DType.F32, [rank, dim], Fill(rank * dim, 0.1f)),
            ["diffusion_model.blocks.0.self_attn.output_proj.lora_up.weight"] = (DType.F32, [dim, rank], Fill(dim * rank, 0.1f)),
            ["diffusion_model.blocks.0.mlp.layer1.lora_down.weight"] = (DType.F32, [rank, dim], Fill(rank * dim, 0.1f)),
            ["diffusion_model.blocks.0.mlp.layer1.lora_up.weight"] = (DType.F32, [dim, rank], Fill(dim * rank, 0.1f)),
        };
        string path = CreateSafeTensorsFile("anima_comfy", tensors);

        using LoraFile file = LoraFile.Load(path);

        Assert.Equal(LoraFormat.Anima, file.Format);
        Assert.Equal(3, file.Layers.Count);
        LoraLayer query = file.Layers.Single(l => l.TargetKey == "blocks.0.self_attn.q_proj.weight");
        Assert.Equal(LoraTarget.Transformer, query.Target);
        Assert.Equal(rank, query.Rank);
        Assert.Equal(1.0f, query.Alpha);
        Assert.Single(file.Layers, l => l.TargetKey == "blocks.0.self_attn.output_proj.weight");
        Assert.Single(file.Layers, l => l.TargetKey == "blocks.0.mlp.layer1.weight");
    }

    [Fact]
    public void Load_FullWeightDiffBesideAnimaLayers_IsRefusedByName()
    {
        // A diff is no LoRA suffix, so the file is claimed by its low-rank pair; the diff is what must be refused.
        Dictionary<string, (DType dtype, long[] shape, float[] data)> tensors = new()
        {
            ["diffusion_model.blocks.0.self_attn.q_proj.lora_down.weight"] = (DType.F32, [2, 8], Fill(16, 0.5f)),
            ["diffusion_model.blocks.0.self_attn.q_proj.lora_up.weight"] = (DType.F32, [8, 2], Fill(16, 0.5f)),
            ["diffusion_model.blocks.0.self_attn.output_proj.diff"] = (DType.F32, [8, 8], Fill(64, 0.5f)),
        };
        string path = CreateSafeTensorsFile("anima_diff", tensors);

        NotSupportedException ex = Assert.Throws<NotSupportedException>(() => LoraFile.Load(path));
        Assert.Contains("full-weight diff", ex.Message);
    }

    [Theory]
    [InlineData("blocks.0.self_attn.q_proj.lora_down.weight", true)]
    [InlineData("diffusion_model.blocks.0.self_attn.q.lora_down.weight", false)]
    [InlineData("diffusion_model.blocks.0.self_attn.q_proj.weight", false)]
    public void IsDistinctive_OnlyClaimsAnimaBlockModules(string key, bool expected)
    {
        Assert.Equal(expected, AnimaLoraMapper.IsDistinctive(key));
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
