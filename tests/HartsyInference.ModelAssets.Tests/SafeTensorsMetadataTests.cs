using System.Text;
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
}
