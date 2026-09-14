using System.Text;
using System.Text.Json;
using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Lora;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>Pins the LDM/CompVis block-name arm of the kohya SD path: sd-scripts emits
/// <c>lora_unet_input_blocks_*</c> / <c>output_blocks_*</c> / <c>middle_block_*</c> for most SD1.5 and SDXL LoRAs,
/// while the loaded UNet dict is diffusers-named. Expected keys are the checkpoint converter's own output for the
/// matching weight, so detection and the LDM-to-diffusers mapping are pinned against one source of truth.</summary>
public sealed unsafe class LoraKohyaLdmBlockTests : IDisposable
{
    private readonly string _tempDir = Path.Combine(Path.GetTempPath(), $"lora-ldm-{Guid.NewGuid():N}");

    public LoraKohyaLdmBlockTests() => Directory.CreateDirectory(_tempDir);

    private static SafeTensorDescriptor Desc(string name) => new()
    {
        Name = name,
        DType = DType.F16,
        Shape = new TensorShape(16, 2048),
        DataOffset = 0,
        ByteLength = 16 * 2048 * 2,
    };

    private static Dictionary<string, SafeTensorDescriptor> Descriptors(params string[] keys)
    {
        Dictionary<string, SafeTensorDescriptor> descriptors = [];
        foreach (string key in keys)
        {
            descriptors[key] = Desc(key);
        }
        return descriptors;
    }

    /// <summary>LDM block names plus a CLIP-G encoder are SDXL, exactly as the diffusers spellings are.</summary>
    [Theory]
    [InlineData("lora_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_q")]
    [InlineData("lora_unet_output_blocks_5_1_transformer_blocks_1_attn2_to_k")]
    [InlineData("lora_unet_middle_block_1_proj_in")]
    public void LdmBlocks_WithTe2_IsKohyaSdxl(string root)
    {
        Dictionary<string, SafeTensorDescriptor> descriptors = Descriptors(
            $"{root}.lora_down.weight",
            $"{root}.lora_up.weight",
            "lora_te2_text_model_encoder_layers_0_self_attn_q_proj.lora_down.weight");
        Assert.Equal(LoraFormat.KohyaSdxl, LoraFormatDetector.Detect(descriptors));
    }

    /// <summary>Without a CLIP-G encoder the same block names are SD1.5.</summary>
    [Fact]
    public void LdmBlocks_WithoutTe2_IsKohyaSd15()
    {
        Dictionary<string, SafeTensorDescriptor> descriptors = Descriptors(
            "lora_unet_input_blocks_1_1_transformer_blocks_0_attn1_to_q.lora_down.weight",
            "lora_unet_input_blocks_1_1_transformer_blocks_0_attn1_to_q.lora_up.weight");
        Assert.Equal(LoraFormat.KohyaSd15, LoraFormatDetector.Detect(descriptors));
    }

    /// <summary>Each LDM root merges onto the diffusers key the SDXL checkpoint converter produces for it.</summary>
    [Theory]
    [InlineData("lora_unet_input_blocks_7_1_transformer_blocks_3_attn2_to_out_0",
        "down_blocks.2.attentions.0.transformer_blocks.3.attn2.to_out.0.weight")]
    [InlineData("lora_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_q",
        "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q.weight")]
    [InlineData("lora_unet_input_blocks_4_1_proj_in", "down_blocks.1.attentions.0.proj_in.weight")]
    [InlineData("lora_unet_middle_block_1_proj_in", "mid_block.attentions.0.proj_in.weight")]
    [InlineData("lora_unet_middle_block_1_transformer_blocks_0_attn2_to_v",
        "mid_block.attentions.0.transformer_blocks.0.attn2.to_v.weight")]
    [InlineData("lora_unet_output_blocks_0_1_transformer_blocks_9_ff_net_0_proj",
        "up_blocks.0.attentions.0.transformer_blocks.9.ff.net.0.proj.weight")]
    [InlineData("lora_unet_output_blocks_5_1_transformer_blocks_1_attn2_to_k",
        "up_blocks.1.attentions.2.transformer_blocks.1.attn2.to_k.weight")]
    [InlineData("lora_unet_output_blocks_2_1_transformer_blocks_0_ff_net_2",
        "up_blocks.0.attentions.2.transformer_blocks.0.ff.net.2.weight")]
    public void LdmRoot_MergesOntoDiffusersUNetKey(string loraRoot, string canonicalKey)
        => AssertMergesOnto(loraRoot, canonicalKey, LoraTarget.UNet,
            "lora_te2_text_model_encoder_layers_0_self_attn_q_proj");

    /// <summary>SD1.5's block tables differ from SDXL's — 12 input blocks over 4 levels, with attention at level 0
    /// where SDXL has none — so the same root maps elsewhere. A UNet-only companion keeps the file out of the SDXL arm.
    /// A wrong map here is the silent case: <c>lora_te_</c> still merges, so the zero-match refusal never fires.</summary>
    [Theory]
    [InlineData("lora_unet_input_blocks_1_1_transformer_blocks_0_attn1_to_q",
        "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight")]
    [InlineData("lora_unet_input_blocks_8_1_transformer_blocks_0_attn2_to_v",
        "down_blocks.2.attentions.1.transformer_blocks.0.attn2.to_v.weight")]
    [InlineData("lora_unet_output_blocks_3_1_transformer_blocks_0_ff_net_0_proj",
        "up_blocks.1.attentions.0.transformer_blocks.0.ff.net.0.proj.weight")]
    [InlineData("lora_unet_output_blocks_11_1_proj_out", "up_blocks.3.attentions.2.proj_out.weight")]
    public void Sd15LdmRoot_MergesOntoDiffusersUNetKey(string loraRoot, string canonicalKey)
        => AssertMergesOnto(loraRoot, canonicalKey, LoraTarget.UNet, "lora_unet_middle_block_1_transformer_blocks_0_attn1_to_k");

    /// <summary>LoCon/conv LoRAs reach the resnets, whose LDM sub-keys are compound names the underscore→dot pass
    /// would otherwise split (<c>in_layers</c> → <c>in.layers</c>) into a key that matches nothing.</summary>
    [Theory]
    [InlineData("lora_unet_input_blocks_1_0_in_layers_2", "down_blocks.0.resnets.0.conv1.weight")]
    [InlineData("lora_unet_input_blocks_1_0_out_layers_3", "down_blocks.0.resnets.0.conv2.weight")]
    [InlineData("lora_unet_input_blocks_4_0_emb_layers_1", "down_blocks.1.resnets.0.time_emb_proj.weight")]
    [InlineData("lora_unet_input_blocks_4_0_skip_connection", "down_blocks.1.resnets.0.conv_shortcut.weight")]
    [InlineData("lora_unet_output_blocks_2_0_in_layers_2", "up_blocks.0.resnets.2.conv1.weight")]
    public void LoConResnetRoot_MergesOntoDiffusersUNetKey(string loraRoot, string canonicalKey)
        => AssertMergesOnto(loraRoot, canonicalKey, LoraTarget.UNet,
            "lora_te2_text_model_encoder_layers_0_self_attn_q_proj");

    /// <summary>The CLIP-G half of the same file routes unchanged — LDM naming is a UNet-only concern.</summary>
    [Fact]
    public void Te2Root_MergesOntoClipGKey()
        => AssertMergesOnto("lora_te2_text_model_encoder_layers_0_self_attn_q_proj",
            "text_model.encoder.layers.0.self_attn.q_proj.weight", LoraTarget.ClipG,
            "lora_unet_middle_block_1_proj_in");

    /// <summary>Builds a two-layer LoRA and asserts the root under test merges onto exactly the expected key.
    /// The companion root is what decides detection — a <c>lora_te2_</c> root makes the file SDXL, a UNet-only one
    /// leaves it SD1.5 — and never contributes to the merge count, since only the key under test is in the dict.</summary>
    private void AssertMergesOnto(string loraRoot, string canonicalKey, LoraTarget target, string companionRoot)
    {
        // alpha = rank = 2 and strength 1 make the merged delta exactly down·up = 4 · (0.5 · 0.25) per element.
        Dictionary<string, (DType dtype, long[] shape, float[] data)> tensors = [];
        foreach (string root in new[] { loraRoot, companionRoot })
        {
            tensors[$"{root}.lora_down.weight"] = (DType.F32, [2, 4], Fill(8, 0.5f));
            tensors[$"{root}.lora_up.weight"] = (DType.F32, [4, 2], Fill(8, 0.25f));
            tensors[$"{root}.alpha"] = (DType.F32, [1], [2.0f]);
        }
        string path = CreateSafeTensorsFile(_tempDir, "ldm_merge", tensors);

        Tensor baseW = new Tensor(new TensorShape(4, 4), DType.F32);
        float* bp = (float*)baseW.DataPointer;
        for (int i = 0; i < 16; i++)
        {
            bp[i] = 1.0f;
        }
        Dictionary<string, Tensor> weights = new() { [canonicalKey] = baseW };

        using CpuBackend backend = new();
        using LoraStack stack = new();
        stack.AddFromPath(path, strength: 1.0f);
        int merged = stack.ApplyTo(weights, target, backend);

        Assert.Equal(1, merged);
        float* mp = (float*)weights[canonicalKey].DataPointer;
        for (int i = 0; i < 16; i++)
        {
            Assert.True(MathF.Abs(mp[i] - 1.25f) < 1e-5f, $"merged[{i}] = {mp[i]}, expected 1.25");
        }
        baseW.Dispose();
    }

    private static float[] Fill(int count, float value)
    {
        float[] data = new float[count];
        Array.Fill(data, value);
        return data;
    }

    private static string CreateSafeTensorsFile(string dir, string name,
        Dictionary<string, (DType dtype, long[] shape, float[] data)> tensors)
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

        string headerJson = JsonSerializer.Serialize(headerDict);
        byte[] headerBytes = Encoding.UTF8.GetBytes(headerJson);

        string filePath = Path.Combine(dir, $"{name}.safetensors");
        using FileStream fs = new(filePath, FileMode.Create, FileAccess.Write);
        using BinaryWriter writer = new(fs);
        writer.Write((long)headerBytes.Length);
        writer.Write(headerBytes);
        writer.Write(dataBlob);
        return filePath;
    }

    public void Dispose()
    {
        try
        {
            if (Directory.Exists(_tempDir))
            {
                Directory.Delete(_tempDir, true);
            }
        }
        catch (IOException)
        {
            // best-effort cleanup
        }
    }
}
