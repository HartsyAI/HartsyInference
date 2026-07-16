using System.Text;
using System.Text.Json;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Adapters;
using HartsyInference.Diffusion.Models.Denoisers;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

public sealed class AdapterLoaderTests : IDisposable
{
    private readonly string _tempDir;

    public AdapterLoaderTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), $"sharpinf-adapter-tests-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_tempDir);
    }

    [Fact]
    public void ControlNetLoader_DetectsSd15FromKeySignature()
    {
        string path = CreateSafeTensorsFile("sd15-canny.safetensors", new()
        {
            ["input_blocks.0.0.weight"] = (DType.F32, [320, 4, 3, 3], new float[320 * 4 * 3 * 3]),
            ["input_hint_block.0.weight"] = (DType.F32, [16, 3, 3, 3], new float[16 * 3 * 3 * 3]),
            ["controlnet_down_blocks.0.weight"] = (DType.F32, [320, 320, 1, 1], new float[320 * 320]),
        });

        using ControlNetFile file = ControlNetLoader.Load(path);
        Assert.Equal(ControlNetBaseModel.Sd15, file.BaseModel);
        Assert.Equal(ControlNetMode.Canny, file.Mode);
        Assert.Equal(768, file.Config.CrossAttentionDim);
    }

    [Fact]
    public void ControlNetLoader_DetectsSdxlFromCrossAttnDim()
    {
        string path = CreateSafeTensorsFile("controlnet-sdxl-depth.safetensors", new()
        {
            ["down_blocks.0.attentions.0.proj_in.weight"] = (DType.F32, [640, 2048], new float[640 * 2048]),
            ["controlnet_down_blocks.0.weight"] = (DType.F32, [320, 320, 1, 1], new float[320 * 320]),
        });

        using ControlNetFile file = ControlNetLoader.Load(path);
        Assert.Equal(ControlNetBaseModel.Sdxl, file.BaseModel);
        Assert.Equal(ControlNetMode.Depth, file.Mode);
        Assert.Equal(2048, file.Config.CrossAttentionDim);
    }

    [Fact]
    public void ControlNetLoader_DetectsFluxFromTransformerBlocks()
    {
        string path = CreateSafeTensorsFile("flux-controlnet-canny.safetensors", new()
        {
            ["transformer_blocks.0.attn.to_q.weight"] = (DType.F32, [3072, 3072], new float[3072 * 3072]),
            ["controlnet_blocks.0.weight"] = (DType.F32, [3072, 3072], new float[3072 * 3072]),
        });

        using ControlNetFile file = ControlNetLoader.Load(path);
        Assert.Equal(ControlNetBaseModel.Flux, file.BaseModel);
        Assert.Equal(ControlNetMode.Canny, file.Mode);
        Assert.Equal(4096, file.Config.CrossAttentionDim);
    }

    [Fact]
    public void ControlNetLoader_ModeOverrideTakesPrecedence()
    {
        string path = CreateSafeTensorsFile("ambiguous-name.safetensors", new()
        {
            ["input_blocks.0.0.weight"] = (DType.F32, [320, 4, 3, 3], new float[320 * 4 * 3 * 3]),
        });

        using ControlNetFile file = ControlNetLoader.Load(path, modeOverride: ControlNetMode.OpenPose);
        Assert.Equal(ControlNetMode.OpenPose, file.Mode);
    }

    [Fact]
    public void ControlNetLoader_UnrecognizedFile_Throws()
    {
        string path = CreateSafeTensorsFile("garbage.safetensors", new()
        {
            ["some.random.key"] = (DType.F32, [4, 4], new float[16]),
        });

        Assert.Throws<HartsyInferenceException>(() => ControlNetLoader.Load(path));
    }

    [Fact]
    public void IpAdapterLoader_DetectsSd15Standard()
    {
        string path = CreateSafeTensorsFile("ip-adapter_sd15.safetensors", new()
        {
            ["image_proj.weight"] = (DType.F32, [768 * 4, 1024], new float[768 * 4 * 1024]),
            ["ip_adapter.0.to_k_ip.weight"] = (DType.F32, [768, 768], new float[768 * 768]),
            ["ip_adapter.0.to_v_ip.weight"] = (DType.F32, [768, 768], new float[768 * 768]),
        });

        using IpAdapterFile file = IpAdapterLoader.Load(path);
        Assert.Equal(IpAdapterBaseModel.Sd15, file.BaseModel);
        Assert.False(file.Config.IsPlus);
        Assert.False(file.Config.IsFaceId);
        Assert.Equal(4, file.Config.NumImageTokens);
        Assert.Equal(768, file.Config.CrossAttentionDim);
    }

    [Fact]
    public void IpAdapterLoader_DetectsPlus_From_NameAnd_NormKey()
    {
        string path = CreateSafeTensorsFile("ip-adapter-plus_sdxl.safetensors", new()
        {
            ["image_proj.norm.weight"] = (DType.F32, [768], new float[768]),
            ["image_proj.proj_in.weight"] = (DType.F32, [768, 1024], new float[768 * 1024]),
            ["ip_adapter.0.to_k_ip.weight"] = (DType.F32, [2048, 2048], new float[2048 * 2048]),
        });

        using IpAdapterFile file = IpAdapterLoader.Load(path);
        Assert.Equal(IpAdapterBaseModel.Sdxl, file.BaseModel);
        Assert.True(file.Config.IsPlus);
        Assert.Equal(16, file.Config.NumImageTokens);
        Assert.Equal(2048, file.Config.CrossAttentionDim);
    }

    [Fact]
    public void IpAdapterLoader_DetectsFaceId_From_FilenameKeyword()
    {
        string path = CreateSafeTensorsFile("ip-adapter-faceid_sd15.safetensors", new()
        {
            ["image_proj.weight"] = (DType.F32, [768 * 4, 512], new float[768 * 4 * 512]),
            ["ip_adapter.0.to_k_ip.weight"] = (DType.F32, [768, 768], new float[768 * 768]),
        });

        using IpAdapterFile file = IpAdapterLoader.Load(path);
        Assert.True(file.Config.IsFaceId);
        Assert.Contains("ArcFace", file.Config.ClipImageModel);
    }

    [Fact]
    public void IpAdapterLoader_StandardWithNormKeys_NotDetectedAsPlus()
    {
        // Real ip-adapter_sd15.safetensors carries image_proj.norm.{weight,bias} (the standard
        // projection's post-Linear LayerNorm) — the norm key must NOT trip Plus detection.
        string path = CreateSafeTensorsFile("ip-adapter_sd15_std.safetensors", new()
        {
            ["image_proj.proj.weight"] = (DType.F32, [768 * 4, 1024], new float[768 * 4 * 1024]),
            ["image_proj.proj.bias"] = (DType.F32, [768 * 4], new float[768 * 4]),
            ["image_proj.norm.weight"] = (DType.F32, [768], new float[768]),
            ["image_proj.norm.bias"] = (DType.F32, [768], new float[768]),
            ["ip_adapter.1.to_k_ip.weight"] = (DType.F32, [320, 768], new float[320 * 768]),
            ["ip_adapter.1.to_v_ip.weight"] = (DType.F32, [320, 768], new float[320 * 768]),
        });

        using IpAdapterFile file = IpAdapterLoader.Load(path);
        Assert.Equal(IpAdapterBaseModel.Sd15, file.BaseModel);
        Assert.False(file.Config.IsPlus);
        Assert.Equal(4, file.Config.NumImageTokens);
        Assert.Equal(768, file.Config.CrossAttentionDim);
        Assert.Equal(1024, file.Config.ImageEmbeddingDim);
    }

    [Fact]
    public void IpAdapter_Sd15Standard_LoadsAndProjectsFromSyntheticCheckpoint()
    {
        // Tiny SD1.5 standard checkpoint in the real key layout (2 cross-attn layers instead of 16).
        string path = CreateSafeTensorsFile("ip-adapter_sd15_tiny.safetensors", new()
        {
            ["image_proj.proj.weight"] = (DType.F32, [768 * 4, 1024], new float[768 * 4 * 1024]),
            ["image_proj.proj.bias"] = (DType.F32, [768 * 4], new float[768 * 4]),
            ["image_proj.norm.weight"] = (DType.F32, [768], OnesArray(768)),
            ["image_proj.norm.bias"] = (DType.F32, [768], new float[768]),
            ["ip_adapter.1.to_k_ip.weight"] = (DType.F32, [320, 768], new float[320 * 768]),
            ["ip_adapter.1.to_v_ip.weight"] = (DType.F32, [320, 768], new float[320 * 768]),
            ["ip_adapter.3.to_k_ip.weight"] = (DType.F32, [640, 768], new float[640 * 768]),
            ["ip_adapter.3.to_v_ip.weight"] = (DType.F32, [640, 768], new float[640 * 768]),
        });

        using IpAdapterFile file = IpAdapterLoader.Load(path);
        using IpAdapter adapter = new IpAdapter(file.Config);
        adapter.LoadWeights(file.Weights);

        Assert.Equal(2, adapter.CrossAttentionLayerCount);
        Assert.Equal(4, adapter.NumImageTokens);
        Assert.Equal(new TensorShape(320, 768), adapter.GetToKIpWeight(0).Shape);
        Assert.Equal(new TensorShape(640, 768), adapter.GetToVIpWeight(1).Shape);

        using CpuBackend backend = new CpuBackend();
        using Tensor clipEmbed = new Tensor(new TensorShape(1, 1024), DType.F32);
        clipEmbed.AsSpan<float>().Clear();
        using Tensor tokens = adapter.ProjectImage(backend, clipEmbed);
        Assert.Equal(new TensorShape(1, 4, 768), tokens.Shape);
        foreach (float v in tokens.AsSpan<float>())
        {
            Assert.Equal(0f, v);
        }
    }

    [Fact]
    public void IpAdapter_Sd15Plus_ConstructsResampler()
    {
        using IpAdapter adapter = new IpAdapter(IpAdapterConfig.Sd15Plus);
        Assert.Equal(16, adapter.NumImageTokens);
        Assert.IsType<IpAdapterPlusResampler>(adapter.ImageProjection);
    }

    [Fact]
    public void ControlNetConditioning_DefaultWindow_IsAlwaysActive()
    {
        using ControlNet controlNet = new ControlNet(ControlNetConfig.Sd15(ControlNetMode.Canny), UNetConfig.Sd15);
        using Tensor condImage = new Tensor(new TensorShape(1, 3, 8, 8), DType.F32);
        ControlNetConditioning conditioning = new() { Adapter = controlNet, ConditionImage = condImage };
        for (int i = 0; i < 20; i++)
        {
            Assert.True(conditioning.IsActiveAtStep(i, 20));
        }
    }

    [Fact]
    public void ControlNetConditioning_StartEndWindow_GatesSteps()
    {
        using ControlNet controlNet = new ControlNet(ControlNetConfig.Sd15(ControlNetMode.Canny), UNetConfig.Sd15);
        using Tensor condImage = new Tensor(new TensorShape(1, 3, 8, 8), DType.F32);
        ControlNetConditioning conditioning = new()
        {
            Adapter = controlNet,
            ConditionImage = condImage,
            StartFraction = 0.25f,
            EndFraction = 0.75f,
        };
        // fraction = step / totalSteps with 20 steps → active iff 0.25 <= i/20 <= 0.75 → i in [5, 15].
        Assert.False(conditioning.IsActiveAtStep(4, 20));
        Assert.True(conditioning.IsActiveAtStep(5, 20));
        Assert.True(conditioning.IsActiveAtStep(15, 20));
        Assert.False(conditioning.IsActiveAtStep(16, 20));
    }

    [Fact]
    public void ControlNetConditioning_FilterActive_HandlesAllNoneAndPartial()
    {
        using ControlNet controlNet = new ControlNet(ControlNetConfig.Sd15(ControlNetMode.Canny), UNetConfig.Sd15);
        using Tensor condImage = new Tensor(new TensorShape(1, 3, 8, 8), DType.F32);
        ControlNetConditioning early = new()
        {
            Adapter = controlNet,
            ConditionImage = condImage,
            EndFraction = 0.4f,
        };
        ControlNetConditioning late = new()
        {
            Adapter = controlNet,
            ConditionImage = condImage,
            StartFraction = 0.6f,
        };
        List<ControlNetConditioning> stack = [early, late];

        Assert.Null(ControlNetConditioning.FilterActive(null, 0, 20));
        Assert.Null(ControlNetConditioning.FilterActive([], 0, 20));

        // Step 0 (fraction 0): only the early adapter.
        IReadOnlyList<ControlNetConditioning>? atStart = ControlNetConditioning.FilterActive(stack, 0, 20);
        Assert.NotNull(atStart);
        Assert.Same(early, Assert.Single(atStart));

        // Step 10 (fraction 0.5): neither window covers it.
        Assert.Null(ControlNetConditioning.FilterActive(stack, 10, 20));

        // Step 19 (fraction 0.95): only the late adapter.
        IReadOnlyList<ControlNetConditioning>? atEnd = ControlNetConditioning.FilterActive(stack, 19, 20);
        Assert.NotNull(atEnd);
        Assert.Same(late, Assert.Single(atEnd));

        // All active → the original list instance comes back (no per-step allocation).
        List<ControlNetConditioning> alwaysOn = [new() { Adapter = controlNet, ConditionImage = condImage }];
        Assert.Same(alwaysOn, ControlNetConditioning.FilterActive(alwaysOn, 10, 20));
    }

    [Fact]
    public void ControlNet_Sd15AndSdxl_ResidualCounts_MatchBaseUNetSkipLayout()
    {
        // SD1.5: 1 conv_in + 4 blocks × 2 resnet skips + 3 downsample skips = 12.
        using ControlNet sd15 = new ControlNet(ControlNetConfig.Sd15(ControlNetMode.Canny), UNetConfig.Sd15);
        Assert.Equal(12, sd15.DownResidualCount);

        // SDXL: 1 conv_in + 3 blocks × 2 resnet skips + 2 downsample skips = 9.
        using ControlNet sdxl = new ControlNet(ControlNetConfig.Sdxl(ControlNetMode.Depth), UNetConfig.SdxlBase);
        Assert.Equal(9, sdxl.DownResidualCount);
    }

    [Fact]
    public void UNet_CrossAttentionLayerCounts_MatchIpAdapterCheckpointLayout()
    {
        // ip-adapter_sd15 ships 16 K/V pairs; SDXL adapters ship 70. The UNet's enumeration
        // must land on the same counts or UNet.Forward rejects the flat K/V lists.
        Assert.Equal(16, new UNet(UNetConfig.Sd15).CrossAttentionLayerCount);
        Assert.Equal(70, new UNet(UNetConfig.SdxlBase).CrossAttentionLayerCount);
    }

    private static float[] OnesArray(int count)
    {
        float[] result = new float[count];
        Array.Fill(result, 1f);
        return result;
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
                dataStream.Write(BitConverter.GetBytes(val), 0, 4);
            }
            long end = dataStream.Position;
            offsets[kvp.Key] = (start, end);
        }
        byte[] dataBlob = dataStream.ToArray();

        Dictionary<string, object> header = [];
        foreach (KeyValuePair<string, (DType dtype, long[] shape, float[] data)> kvp in tensors)
        {
            (long start, long end) = offsets[kvp.Key];
            header[kvp.Key] = new Dictionary<string, object>
            {
                ["dtype"] = kvp.Value.dtype.Name,
                ["shape"] = kvp.Value.shape,
                ["data_offsets"] = new long[] { start, end },
            };
        }

        byte[] headerBytes = Encoding.UTF8.GetBytes(JsonSerializer.Serialize(header));
        string filePath = Path.Combine(_tempDir, name);
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
            if (Directory.Exists(_tempDir)) Directory.Delete(_tempDir, true);
        }
        catch { }
    }
}
