using System.Text;
using System.Text.Json;
using SharpInference.Cpu;
using SharpInference.Core.Tensors;
using SharpInference.ModelHandler.Lora;
using SharpInference.ModelHandler.Lora.Mappers;
using SharpInference.ModelHandler.SafeTensors;
using Xunit;

namespace SharpInference.ModelHandler.Tests;

/// <summary>Wan LoRA support: format detection for the three naming families found in the wild (kohya/musubi-tuner
/// underscored, ComfyUI <c>diffusion_model.</c> dotted, diffusers-PEFT <c>transformer.</c> passthrough), the pure
/// body→canonical mapping through the Wan rename table, end-to-end parsing from real safetensors bytes, and a
/// numeric merge through <see cref="LoraStack"/> against a <c>WanVideoTransformer</c>-keyed weight dict.</summary>
public sealed unsafe class LoraWanTests : IDisposable
{
    private readonly string _tempDir;

    public LoraWanTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), $"lora_wan_tests_{Guid.NewGuid():N}");
        Directory.CreateDirectory(_tempDir);
    }

    private static SafeTensorDescriptor Desc(string name) => new()
    {
        Name = name,
        DType = DType.F16,
        Shape = new TensorShape(16, 3072),
        DataOffset = 0,
        ByteLength = 16 * 3072 * 2,
    };

    private static Dictionary<string, SafeTensorDescriptor> Descriptors(params string[] keys)
    {
        Dictionary<string, SafeTensorDescriptor> d = new();
        foreach (string k in keys) d[k] = Desc(k);
        return d;
    }

    [Fact]
    public void MusubiKeys_DetectedAsKohyaWan()
    {
        Dictionary<string, SafeTensorDescriptor> d = Descriptors(
            "lora_unet_blocks_0_self_attn_q.lora_down.weight",
            "lora_unet_blocks_0_self_attn_q.lora_up.weight",
            "lora_unet_blocks_0_self_attn_q.alpha",
            "lora_unet_blocks_29_cross_attn_o.lora_down.weight",
            "lora_unet_blocks_29_cross_attn_o.lora_up.weight",
            "lora_unet_blocks_5_ffn_0.lora_down.weight",
            "lora_unet_blocks_5_ffn_0.lora_up.weight");
        Assert.Equal(LoraFormat.KohyaWan, LoraFormatDetector.Detect(d));
    }

    [Fact]
    public void ComfyDiffusionModelKeys_DetectedAsDiffusersWan()
    {
        // Both suffix conventions appear in Comfy-style Wan repacks.
        Dictionary<string, SafeTensorDescriptor> peft = Descriptors(
            "diffusion_model.blocks.0.self_attn.q.lora_A.weight",
            "diffusion_model.blocks.0.self_attn.q.lora_B.weight");
        Assert.Equal(LoraFormat.DiffusersWan, LoraFormatDetector.Detect(peft));

        Dictionary<string, SafeTensorDescriptor> kohya = Descriptors(
            "diffusion_model.blocks.12.cross_attn.v.lora_down.weight",
            "diffusion_model.blocks.12.cross_attn.v.lora_up.weight",
            "diffusion_model.blocks.12.cross_attn.v.alpha");
        Assert.Equal(LoraFormat.DiffusersWan, LoraFormatDetector.Detect(kohya));
    }

    [Fact]
    public void DiffusersPeftWanKeys_RouteThroughExistingPassthrough()
    {
        // `transformer.blocks.{i}.attn1.to_q` is already the canonical WanVideoTransformer key — the
        // architecture-agnostic diffusers passthrough handles it without a Wan-specific mapper.
        Dictionary<string, SafeTensorDescriptor> d = Descriptors(
            "transformer.blocks.0.attn1.to_q.lora_A.weight",
            "transformer.blocks.0.attn1.to_q.lora_B.weight",
            "transformer.blocks.7.ffn.net.0.proj.lora_A.weight",
            "transformer.blocks.7.ffn.net.0.proj.lora_B.weight");
        Assert.Equal(LoraFormat.DiffusersFlux, LoraFormatDetector.Detect(d));
    }

    [Fact]
    public void ExistingFormats_NoDetectionRegression()
    {
        Dictionary<string, SafeTensorDescriptor> kohyaFlux = Descriptors(
            "lora_unet_double_blocks_0_img_attn_qkv.lora_down.weight",
            "lora_unet_double_blocks_0_img_attn_qkv.lora_up.weight");
        Assert.Equal(LoraFormat.KohyaFlux, LoraFormatDetector.Detect(kohyaFlux));

        Dictionary<string, SafeTensorDescriptor> sd15 = Descriptors(
            "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight",
            "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_up.weight");
        Assert.Equal(LoraFormat.KohyaSd15, LoraFormatDetector.Detect(sd15));
    }

    [Theory]
    // Original Wan naming → the diffusers-style keys WanVideoTransformer.LoadWeights expects.
    [InlineData("blocks.0.self_attn.q", "blocks.0.attn1.to_q.weight")]
    [InlineData("blocks.0.self_attn.o", "blocks.0.attn1.to_out.0.weight")]
    [InlineData("blocks.29.cross_attn.k", "blocks.29.attn2.to_k.weight")]
    [InlineData("blocks.3.cross_attn.k_img", "blocks.3.attn2.add_k_proj.weight")]
    [InlineData("blocks.3.cross_attn.v_img", "blocks.3.attn2.add_v_proj.weight")]
    [InlineData("blocks.5.ffn.0", "blocks.5.ffn.net.0.proj.weight")]
    [InlineData("blocks.5.ffn.2", "blocks.5.ffn.net.2.weight")]
    [InlineData("time_embedding.0", "condition_embedder.time_embedder.linear_1.weight")]
    [InlineData("text_embedding.2", "condition_embedder.text_embedder.linear_2.weight")]
    [InlineData("time_projection.1", "condition_embedder.time_proj.weight")]
    [InlineData("head.head", "proj_out.weight")]
    // Already-diffusers bodies pass through.
    [InlineData("blocks.0.attn1.to_q", "blocks.0.attn1.to_q.weight")]
    [InlineData("blocks.7.ffn.net.0.proj", "blocks.7.ffn.net.0.proj.weight")]
    public void MapBodyToCanonical_HandlesBothNamings(string body, string expected)
    {
        Assert.Equal(expected, WanLoraMapper.MapBodyToCanonical(body));
    }

    [Theory]
    [InlineData("blocks_0_self_attn_q", "blocks.0.self_attn.q")]
    [InlineData("blocks_29_cross_attn_o", "blocks.29.cross_attn.o")]
    [InlineData("blocks_3_cross_attn_k_img", "blocks.3.cross_attn.k_img")]
    [InlineData("blocks_3_cross_attn_norm_k_img", "blocks.3.cross_attn.norm_k_img")]
    [InlineData("blocks_5_ffn_0", "blocks.5.ffn.0")]
    [InlineData("time_projection_1", "time_projection.1")]
    [InlineData("text_embedding_0", "text_embedding.0")]
    public void LoraKeyTransformer_DemanglesWanModulePaths(string input, string expected)
    {
        Assert.Equal(expected, LoraKeyTransformer.UnderscoreToDot(input));
    }

    [Fact]
    public void Load_MusubiFile_ParsesLayersWithCanonicalKeysAndAlpha()
    {
        const int rank = 2, inDim = 8, outDim = 8;
        Dictionary<string, (DType dtype, long[] shape, float[] data)> tensors = new()
        {
            ["lora_unet_blocks_0_self_attn_q.lora_down.weight"] = (DType.F32, [rank, inDim], Fill(rank * inDim, 0.5f)),
            ["lora_unet_blocks_0_self_attn_q.lora_up.weight"] = (DType.F32, [outDim, rank], Fill(outDim * rank, 0.25f)),
            ["lora_unet_blocks_0_self_attn_q.alpha"] = (DType.F32, [1], [1.0f]),
            ["lora_unet_blocks_1_ffn_0.lora_down.weight"] = (DType.F32, [rank, inDim], Fill(rank * inDim, 0.1f)),
            ["lora_unet_blocks_1_ffn_0.lora_up.weight"] = (DType.F32, [outDim, rank], Fill(outDim * rank, 0.1f)),
        };
        string path = CreateSafeTensorsFile(_tempDir, "musubi_wan", tensors);

        using LoraFile file = LoraFile.Load(path);
        Assert.Equal(LoraFormat.KohyaWan, file.Format);
        Assert.Equal(2, file.Layers.Count);

        LoraLayer attn = file.Layers.Single(l => l.TargetKey == "blocks.0.attn1.to_q.weight");
        Assert.Equal(LoraTarget.Transformer, attn.Target);
        Assert.Equal(rank, attn.Rank);
        Assert.Equal(1.0f, attn.Alpha);

        LoraLayer ffn = file.Layers.Single(l => l.TargetKey == "blocks.1.ffn.net.0.proj.weight");
        Assert.Equal(rank, ffn.Alpha);   // no explicit alpha → defaults to rank (scale 1)
    }

    [Fact]
    public void Load_ComfyFile_SkipsDiffEntriesAndParsesPeftSuffixes()
    {
        Dictionary<string, (DType dtype, long[] shape, float[] data)> tensors = new()
        {
            ["diffusion_model.blocks.0.cross_attn.v.lora_A.weight"] = (DType.F32, [2, 8], Fill(16, 0.5f)),
            ["diffusion_model.blocks.0.cross_attn.v.lora_B.weight"] = (DType.F32, [8, 2], Fill(16, 0.5f)),
            ["diffusion_model.blocks.0.norm3.diff"] = (DType.F32, [8], Fill(8, 0.5f)),       // full-weight diff → skipped
            ["diffusion_model.blocks.0.norm3.diff_b"] = (DType.F32, [8], Fill(8, 0.5f)),
        };
        string path = CreateSafeTensorsFile(_tempDir, "comfy_wan", tensors);

        using LoraFile file = LoraFile.Load(path);
        Assert.Equal(LoraFormat.DiffusersWan, file.Format);
        LoraLayer layer = Assert.Single(file.Layers);
        Assert.Equal("blocks.0.attn2.to_v.weight", layer.TargetKey);
    }

    [Fact]
    public void LoraStack_MergesWanDelta_Numerically()
    {
        // W' = W + strength · (alpha/rank) · (B @ A). With A = ones·0.5 [2,4], B = ones·0.25 [4,2],
        // alpha = 1, strength = 2: delta = 2 · (1/2) · (0.25·0.5·2) = 0.25 per element.
        Dictionary<string, (DType dtype, long[] shape, float[] data)> tensors = new()
        {
            ["lora_unet_blocks_0_self_attn_q.lora_down.weight"] = (DType.F32, [2, 4], Fill(8, 0.5f)),
            ["lora_unet_blocks_0_self_attn_q.lora_up.weight"] = (DType.F32, [4, 2], Fill(8, 0.25f)),
            ["lora_unet_blocks_0_self_attn_q.alpha"] = (DType.F32, [1], [1.0f]),
        };
        string path = CreateSafeTensorsFile(_tempDir, "merge_wan", tensors);

        Tensor baseW = new Tensor(new TensorShape(4, 4), DType.F32);
        float* bp = (float*)baseW.DataPointer;
        for (int i = 0; i < 16; i++) bp[i] = 1.0f;
        Dictionary<string, Tensor> weights = new() { ["blocks.0.attn1.to_q.weight"] = baseW };

        using CpuBackend backend = new();
        using LoraStack stack = new();
        stack.AddFromPath(path, strength: 2.0f);
        int merged = stack.ApplyTo(weights, LoraTarget.Transformer, backend);

        Assert.Equal(1, merged);
        float* mp = (float*)weights["blocks.0.attn1.to_q.weight"].DataPointer;
        for (int i = 0; i < 16; i++)
            Assert.True(MathF.Abs(mp[i] - 1.25f) < 1e-5f, $"merged[{i}] = {mp[i]}, expected 1.25");
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
            long end = dataStream.Position;
            offsets[kvp.Key] = (start, end);
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
        catch
        {
            // best-effort cleanup
        }
    }
}
