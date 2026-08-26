using System.Security.Cryptography;
using System.Text;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

public sealed class SafeTensorsMetadataTests
{
    private static string WriteFile(string headerJson, byte[] data)
    {
        string path = Path.Combine(Path.GetTempPath(), $"st-meta-{Guid.NewGuid():N}.safetensors");
        byte[] header = Encoding.UTF8.GetBytes(headerJson);
        using FileStream fs = File.Create(path);
        fs.Write(BitConverter.GetBytes((long)header.Length));
        fs.Write(header);
        fs.Write(data);
        return path;
    }

    [Fact]
    public void MetadataRoundTrips()
    {
        // The config value is itself a JSON blob, stored as a string per the safetensors spec — must come back verbatim.
        string header = "{\"__metadata__\":{\"model_version\":\"2.5.0\",\"config\":\"{\\\"transformer\\\":{\\\"ff_bias\\\":false}}\"}," +
            "\"w\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]}}";
        string path = WriteFile(header, new byte[8]);
        try
        {
            using SafeTensorsLoader loader = new SafeTensorsLoader();
            loader.Load(path);
            Assert.NotNull(loader.Metadata);
            Assert.Equal("2.5.0", loader.Metadata!["model_version"]);
            Assert.Equal("{\"transformer\":{\"ff_bias\":false}}", loader.Metadata["config"]);
            Assert.Single(loader.Descriptors);
            Assert.True(loader.Descriptors.ContainsKey("w"));
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void MetadataNullWhenAbsent()
    {
        string header = "{\"w\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]}}";
        string path = WriteFile(header, new byte[8]);
        try
        {
            using SafeTensorsLoader loader = new SafeTensorsLoader();
            loader.Load(path);
            Assert.Null(loader.Metadata);
            Assert.Single(loader.Descriptors);
        }
        finally
        {
            File.Delete(path);
        }
    }

    private static unsafe Tensor MakeTensor(float[] values)
    {
        Tensor tensor = new Tensor(new TensorShape(values.Length), DType.F32);
        values.AsSpan().CopyTo(new Span<float>((void*)tensor.DataPointer, values.Length));
        return tensor;
    }

    [Fact]
    public void WriterMetadataRoundTripsThroughLoader()
    {
        string path = Path.Combine(Path.GetTempPath(), $"st-write-{Guid.NewGuid():N}.safetensors");
        using Tensor w = MakeTensor([1f, 2f, 3f, 4f]);
        Dictionary<string, string> metadata = new(StringComparer.Ordinal)
        {
            ["modelspec.sai_model_spec"] = "1.0.1",
            ["modelspec.architecture"] = "dia_tts",
            ["modelspec.title"] = "Dia 1.6B",
            ["hartsy.component"] = "main",
        };
        try
        {
            SafeTensorsWriter.Save(path, new Dictionary<string, Tensor> { ["w"] = w }, metadata);
            using SafeTensorsLoader loader = new SafeTensorsLoader();
            loader.Load(path);
            Assert.NotNull(loader.Metadata);
            Assert.Equal("1.0.1", loader.Metadata!["modelspec.sai_model_spec"]);
            Assert.Equal("dia_tts", loader.Metadata["modelspec.architecture"]);
            Assert.Equal("Dia 1.6B", loader.Metadata["modelspec.title"]);
            Assert.Equal("main", loader.Metadata["hartsy.component"]);
            Assert.Single(loader.Descriptors);
            Assert.Equal(4, loader.GetTensor("w").Shape[0]);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void WriterWithoutMetadataEmitsNoMetadataKey()
    {
        string path = Path.Combine(Path.GetTempPath(), $"st-write-{Guid.NewGuid():N}.safetensors");
        using Tensor w = MakeTensor([1f, 2f]);
        try
        {
            SafeTensorsWriter.Save(path, new Dictionary<string, Tensor> { ["w"] = w });
            using SafeTensorsLoader loader = new SafeTensorsLoader();
            loader.Load(path);
            Assert.Null(loader.Metadata);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void WriterRejectsEmptyMetadataKey()
    {
        string path = Path.Combine(Path.GetTempPath(), $"st-write-{Guid.NewGuid():N}.safetensors");
        using Tensor w = MakeTensor([1f]);
        try
        {
            Assert.Throws<HartsyInference.Core.Exceptions.HartsyInferenceException>(() =>
                SafeTensorsWriter.Save(path, new Dictionary<string, Tensor> { ["w"] = w },
                    new Dictionary<string, string> { [""] = "value" }));
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void TensorDataHashCoversPayloadOnlyAndIgnoresMetadata()
    {
        string bare = Path.Combine(Path.GetTempPath(), $"st-hash-a-{Guid.NewGuid():N}.safetensors");
        string stamped = Path.Combine(Path.GetTempPath(), $"st-hash-b-{Guid.NewGuid():N}.safetensors");
        using Tensor w = MakeTensor([1f, 2f, 3f, 4f]);
        Dictionary<string, Tensor> tensors = new(StringComparer.Ordinal) { ["w"] = w };
        try
        {
            string expected = SafeTensorsWriter.ComputeTensorDataSha256(tensors);
            SafeTensorsWriter.Save(bare, tensors);
            SafeTensorsWriter.Save(stamped, tensors, new Dictionary<string, string> { ["modelspec.title"] = "T" });

            // Same payload under different headers must hash the same — this is what makes the value stable across restamps.
            Assert.Equal(expected, PayloadHash(bare));
            Assert.Equal(expected, PayloadHash(stamped));
            Assert.NotEqual(new FileInfo(bare).Length, new FileInfo(stamped).Length);
        }
        finally
        {
            File.Delete(bare);
            File.Delete(stamped);
        }
    }

    private static string PayloadHash(string path)
    {
        using FileStream fs = File.OpenRead(path);
        Span<byte> lengthBytes = stackalloc byte[8];
        fs.ReadExactly(lengthBytes);
        long headerLength = BitConverter.ToInt64(lengthBytes);
        fs.Seek(8 + headerLength, SeekOrigin.Begin);
        return Convert.ToHexString(SHA256.HashData(fs)).ToLowerInvariant();
    }
}
