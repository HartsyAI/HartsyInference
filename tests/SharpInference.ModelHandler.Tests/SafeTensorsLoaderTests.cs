using System.Text;
using System.Text.Json;
using SharpInference.Core.Exceptions;
using SharpInference.Core.Tensors;
using SharpInference.ModelHandler.SafeTensors;
using Xunit;

namespace SharpInference.ModelHandler.Tests;

public sealed unsafe class SafeTensorsLoaderTests : IDisposable
{
    private readonly string _tempDir;

    public SafeTensorsLoaderTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), $"safetensors_tests_{Guid.NewGuid():N}");
        Directory.CreateDirectory(_tempDir);
    }

    [Fact]
    public void Load_ValidFile_ParsesHeader()
    {
        Dictionary<string, (DType dtype, long[] shape, float[] data)> tensors = new()
        {
            ["weight"] = (DType.F32, [2, 3], [1f, 2f, 3f, 4f, 5f, 6f]),
        };
        string filePath = CreateSafeTensorsFile(_tempDir, "valid_single", tensors);

        using SafeTensorsLoader loader = new();
        loader.Load(filePath);

        Assert.True(loader.Descriptors.ContainsKey("weight"));
        SafeTensorDescriptor desc = loader.Descriptors["weight"];
        Assert.Equal(DType.F32, desc.DType);
        Assert.Equal(2, desc.Shape.Rank);
        Assert.Equal(2, desc.Shape[0]);
        Assert.Equal(3, desc.Shape[1]);
    }

    [Fact]
    public void Load_MultipleTensors_ParsesAll()
    {
        Dictionary<string, (DType dtype, long[] shape, float[] data)> tensors = new()
        {
            ["a"] = (DType.F32, [4], [1f, 2f, 3f, 4f]),
            ["b"] = (DType.F32, [2, 2], [5f, 6f, 7f, 8f]),
        };
        string filePath = CreateSafeTensorsFile(_tempDir, "valid_multi", tensors);

        using SafeTensorsLoader loader = new();
        loader.Load(filePath);

        Assert.True(loader.Descriptors.ContainsKey("a"));
        Assert.True(loader.Descriptors.ContainsKey("b"));
        Assert.Equal(2, loader.Descriptors.Count);

        SafeTensorDescriptor descA = loader.Descriptors["a"];
        Assert.Equal(1, descA.Shape.Rank);
        Assert.Equal(4, descA.Shape[0]);

        SafeTensorDescriptor descB = loader.Descriptors["b"];
        Assert.Equal(2, descB.Shape.Rank);
        Assert.Equal(2, descB.Shape[0]);
        Assert.Equal(2, descB.Shape[1]);
    }

    [Fact]
    public void GetTensor_ReturnsCorrectData()
    {
        float[] expected = [1.0f, 2.0f, 3.0f, 4.0f];
        Dictionary<string, (DType dtype, long[] shape, float[] data)> tensors = new()
        {
            ["values"] = (DType.F32, [4], expected),
        };
        string filePath = CreateSafeTensorsFile(_tempDir, "data_check", tensors);

        using SafeTensorsLoader loader = new();
        loader.Load(filePath);

        Tensor tensor = loader.GetTensor("values");
        ReadOnlySpan<float> actual = tensor.AsReadOnlySpan<float>();

        Assert.Equal(4, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], actual[i]);
        }
    }

    [Fact]
    public void GetTensor_UnknownName_Throws()
    {
        Dictionary<string, (DType dtype, long[] shape, float[] data)> tensors = new()
        {
            ["exists"] = (DType.F32, [2], [1f, 2f]),
        };
        string filePath = CreateSafeTensorsFile(_tempDir, "unknown_name", tensors);

        using SafeTensorsLoader loader = new();
        loader.Load(filePath);

        Assert.Throws<KeyNotFoundException>(() => loader.GetTensor("nonexistent"));
    }

    [Fact]
    public void Load_TooSmallFile_Throws()
    {
        string filePath = Path.Combine(_tempDir, "too_small.safetensors");
        File.WriteAllBytes(filePath, new byte[4]);

        using SafeTensorsLoader loader = new();

        Assert.Throws<SharpInferenceException>(() => loader.Load(filePath));
    }

    [Fact]
    public void Load_InvalidHeaderLength_Throws()
    {
        string filePath = Path.Combine(_tempDir, "bad_header_len.safetensors");
        byte[] data = new byte[16];
        // Write a header length that exceeds the file size
        BitConverter.TryWriteBytes(data.AsSpan(0, 8), (long)999999);
        File.WriteAllBytes(filePath, data);

        using SafeTensorsLoader loader = new();

        Assert.Throws<SharpInferenceException>(() => loader.Load(filePath));
    }

    [Fact]
    public void Dispose_PreventsAccess()
    {
        Dictionary<string, (DType dtype, long[] shape, float[] data)> tensors = new()
        {
            ["t"] = (DType.F32, [2], [1f, 2f]),
        };
        string filePath = CreateSafeTensorsFile(_tempDir, "dispose_check", tensors);

        SafeTensorsLoader loader = new();
        loader.Load(filePath);
        loader.Dispose();

        Assert.Throws<ObjectDisposedException>(() => loader.GetTensor("t"));
    }

    private static string CreateSafeTensorsFile(
        string dir,
        string name,
        Dictionary<string, (DType dtype, long[] shape, float[] data)> tensors)
    {
        // Build the data blob first so we know offsets
        using MemoryStream dataStream = new();
        Dictionary<string, (long start, long end)> offsets = new();

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

        // Build JSON header
        Dictionary<string, object> headerDict = new();
        foreach (KeyValuePair<string, (DType dtype, long[] shape, float[] data)> kvp in tensors)
        {
            string dtypeStr = kvp.Value.dtype.Name;

            (long start, long end) = offsets[kvp.Key];
            Dictionary<string, object> tensorEntry = new()
            {
                ["dtype"] = dtypeStr,
                ["shape"] = kvp.Value.shape,
                ["data_offsets"] = new long[] { start, end },
            };
            headerDict[kvp.Key] = tensorEntry;
        }

        string headerJson = JsonSerializer.Serialize(headerDict);
        byte[] headerBytes = Encoding.UTF8.GetBytes(headerJson);
        long headerLength = headerBytes.Length;

        // Write file: 8-byte header length + header JSON + data blob
        string filePath = Path.Combine(dir, $"{name}.safetensors");
        using FileStream fs = new(filePath, FileMode.Create, FileAccess.Write);
        using BinaryWriter writer = new(fs);

        writer.Write(headerLength);
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
            // Best-effort cleanup
        }
    }
}
