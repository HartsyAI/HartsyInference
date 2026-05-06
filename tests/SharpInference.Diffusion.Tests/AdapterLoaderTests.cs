using System.Text;
using System.Text.Json;
using SharpInference.Core.Exceptions;
using SharpInference.Core.Tensors;
using SharpInference.Diffusion.Adapters;
using Xunit;

namespace SharpInference.Diffusion.Tests;

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

        Assert.Throws<SharpInferenceException>(() => ControlNetLoader.Load(path));
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
