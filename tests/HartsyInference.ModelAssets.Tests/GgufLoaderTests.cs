using System.Text;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Gguf;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

public sealed unsafe class GgufLoaderTests : IDisposable
{
    private readonly string _tempDir;

    public GgufLoaderTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), $"gguf_tests_{Guid.NewGuid():N}");
        Directory.CreateDirectory(_tempDir);
    }

    [Fact]
    public void Load_ValidFile_ParsesVersion()
    {
        Dictionary<string, object> metadata = new();
        Dictionary<string, (uint ggufType, long[] shape, byte[] data)> tensors = new();
        string filePath = CreateGgufFile(_tempDir, "version_check", metadata, tensors);

        using GgufLoader loader = new();
        loader.Load(filePath);

        Assert.Equal(3, loader.Version);
    }

    [Fact]
    public void Load_ParsesMetadata()
    {
        Dictionary<string, object> metadata = new()
        {
            ["general.architecture"] = "llama",
        };
        Dictionary<string, (uint ggufType, long[] shape, byte[] data)> tensors = new();
        string filePath = CreateGgufFile(_tempDir, "metadata_check", metadata, tensors);

        using GgufLoader loader = new();
        loader.Load(filePath);

        Assert.Equal("llama", loader.Metadata.GetString("general.architecture"));
    }

    [Fact]
    public void Load_ParsesTensorInfo()
    {
        float[] values = [1f, 2f, 3f, 4f];
        byte[] tensorData = new byte[values.Length * sizeof(float)];
        Buffer.BlockCopy(values, 0, tensorData, 0, tensorData.Length);

        Dictionary<string, object> metadata = new();
        Dictionary<string, (uint ggufType, long[] shape, byte[] data)> tensors = new()
        {
            ["weight"] = (0, [4], tensorData),
        };
        string filePath = CreateGgufFile(_tempDir, "tensor_info", metadata, tensors);

        using GgufLoader loader = new();
        loader.Load(filePath);

        Assert.True(loader.Descriptors.ContainsKey("weight"));
        GgufTensorDescriptor desc = loader.Descriptors["weight"];
        Assert.Equal(DType.F32, desc.DType);
        Assert.Equal(1, desc.Shape.Rank);
        Assert.Equal(4, desc.Shape[0]);
    }

    [Fact]
    public void Load_InvalidMagic_Throws()
    {
        string filePath = Path.Combine(_tempDir, "bad_magic.gguf");
        using (FileStream fs = new(filePath, FileMode.Create, FileAccess.Write))
        using (BinaryWriter writer = new(fs))
        {
            writer.Write((uint)0xDEADBEEF); // wrong magic
            writer.Write((uint)3);            // version
            writer.Write((ulong)0);           // tensor count
            writer.Write((ulong)0);           // metadata kv count
        }

        using GgufLoader loader = new();

        Assert.Throws<HartsyInferenceException>(() => loader.Load(filePath));
    }

    [Fact]
    public void Load_UnsupportedVersion_Throws()
    {
        string filePath = Path.Combine(_tempDir, "bad_version.gguf");
        using (FileStream fs = new(filePath, FileMode.Create, FileAccess.Write))
        using (BinaryWriter writer = new(fs))
        {
            writer.Write((uint)0x46554747); // "GGUF" magic
            writer.Write((uint)99);          // unsupported version
            writer.Write((ulong)0);          // tensor count
            writer.Write((ulong)0);          // metadata kv count
        }

        using GgufLoader loader = new();

        Assert.Throws<UnsupportedModelException>(() => loader.Load(filePath));
    }

    [Fact]
    public void GetTensor_ReturnsCorrectData()
    {
        float[] expected = [1.0f, 2.0f, 3.0f, 4.0f];
        byte[] tensorData = new byte[expected.Length * sizeof(float)];
        Buffer.BlockCopy(expected, 0, tensorData, 0, tensorData.Length);

        Dictionary<string, object> metadata = new();
        Dictionary<string, (uint ggufType, long[] shape, byte[] data)> tensors = new()
        {
            ["data"] = (0, [4], tensorData),
        };
        string filePath = CreateGgufFile(_tempDir, "data_check", metadata, tensors);

        using GgufLoader loader = new();
        loader.Load(filePath);

        Tensor tensor = loader.GetTensor("data");
        ReadOnlySpan<float> actual = tensor.AsReadOnlySpan<float>();

        Assert.Equal(4, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], actual[i]);
        }
    }

    [Fact]
    public void Dispose_PreventsAccess()
    {
        float[] values = [1f, 2f];
        byte[] tensorData = new byte[values.Length * sizeof(float)];
        Buffer.BlockCopy(values, 0, tensorData, 0, tensorData.Length);

        Dictionary<string, object> metadata = new();
        Dictionary<string, (uint ggufType, long[] shape, byte[] data)> tensors = new()
        {
            ["t"] = (0, [2], tensorData),
        };
        string filePath = CreateGgufFile(_tempDir, "dispose_check", metadata, tensors);

        GgufLoader loader = new();
        loader.Load(filePath);
        loader.Dispose();

        Assert.Throws<ObjectDisposedException>(() => loader.GetTensor("t"));
    }

    private static string CreateGgufFile(
        string dir,
        string name,
        Dictionary<string, object> metadata,
        Dictionary<string, (uint ggufType, long[] shape, byte[] data)> tensors)
    {
        string filePath = Path.Combine(dir, $"{name}.gguf");
        using FileStream fs = new(filePath, FileMode.Create, FileAccess.Write);
        using BinaryWriter writer = new(fs);

        // Magic
        writer.Write((uint)0x46554747);

        // Version 3
        writer.Write((uint)3);

        // Tensor count
        writer.Write((ulong)tensors.Count);

        // Metadata KV count
        writer.Write((ulong)metadata.Count);

        // Write metadata KVs
        foreach (KeyValuePair<string, object> kvp in metadata)
        {
            WriteGgufString(writer, kvp.Key);

            if (kvp.Value is string strVal)
            {
                writer.Write((uint)8); // STRING type
                WriteGgufString(writer, strVal);
            }
            else if (kvp.Value is uint uintVal)
            {
                writer.Write((uint)4); // UINT32 type
                writer.Write(uintVal);
            }
            else
            {
                throw new NotSupportedException($"Unsupported metadata value type: {kvp.Value.GetType()}");
            }
        }

        // Write tensor infos
        // Track relative offsets within the data section
        long relativeOffset = 0;
        List<(string name, byte[] data)> tensorDataList = new();

        foreach (KeyValuePair<string, (uint ggufType, long[] shape, byte[] data)> kvp in tensors)
        {
            // Tensor name
            WriteGgufString(writer, kvp.Key);

            // Number of dimensions (uint32)
            writer.Write((uint)kvp.Value.shape.Length);

            // Dimension sizes (uint64 each)
            foreach (long dim in kvp.Value.shape)
            {
                writer.Write((ulong)dim);
            }

            // Type (uint32)
            writer.Write(kvp.Value.ggufType);

            // Relative offset within data section (uint64)
            writer.Write((ulong)relativeOffset);

            relativeOffset += kvp.Value.data.Length;
            tensorDataList.Add((kvp.Key, kvp.Value.data));
        }

        // Alignment padding to 32 bytes
        long currentPos = fs.Position;
        long alignment = 32;
        long aligned = (currentPos + alignment - 1) & ~(alignment - 1);
        long paddingNeeded = aligned - currentPos;
        if (paddingNeeded > 0)
        {
            byte[] padding = new byte[paddingNeeded];
            writer.Write(padding);
        }

        // Write tensor data
        foreach ((string tensorName, byte[] data) in tensorDataList)
        {
            writer.Write(data);
        }

        return filePath;
    }

    private static void WriteGgufString(BinaryWriter writer, string value)
    {
        byte[] utf8Bytes = Encoding.UTF8.GetBytes(value);
        writer.Write((ulong)utf8Bytes.Length);
        writer.Write(utf8Bytes);
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
            // Best-effort cleanup
        }
    }
}
