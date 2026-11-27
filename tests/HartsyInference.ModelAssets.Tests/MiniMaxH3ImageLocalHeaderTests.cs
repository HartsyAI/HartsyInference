using System.Buffers.Binary;
using System.Text.Json;
using HartsyInference.ModelAssets.Registry;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>Detection on the local safetensors header of each published MiniMax-H3-Image transformer build. Reads only the
/// header. A build that is not on this machine is reported as skipped (not run), never as passed.</summary>
[Trait("Category", "Integration")]
public sealed class MiniMaxH3ImageLocalHeaderTests
{
    [LocalMiniMaxH3ImageBuildFact("diffusion_models/MiniMaxH3Image/minimax_h3_image_pruned_int8_convrot.safetensors")]
    public void Int8ConvRotBuildHeaderDetectsAsTheImageArchitecture()
    {
        AssertDetectsAsImage("minimax_h3_image_pruned_int8_convrot.safetensors");
    }

    [LocalMiniMaxH3ImageBuildFact("diffusion_models/MiniMaxH3Image/minimax_h3_image_pruned_bf16.safetensors")]
    public void Bf16BuildHeaderDetectsAsTheImageArchitecture()
    {
        AssertDetectsAsImage("minimax_h3_image_pruned_bf16.safetensors");
    }

    [LocalMiniMaxH3ImageBuildFact("diffusion_models/MiniMaxH3Image/minimax_h3_image_pruned_w4a8.safetensors")]
    public void W4a8BuildHeaderDetectsAsTheImageArchitecture()
    {
        AssertDetectsAsImage("minimax_h3_image_pruned_w4a8.safetensors");
    }

    private static void AssertDetectsAsImage(string fileName)
    {
        string path = Path.Combine(ModelsDir(), "diffusion_models", "MiniMaxH3Image", fileName);
        Assert.Equal(ModelArchitecture.MiniMaxH3Image, ModelArchitectureDetector.Detect(HeaderKeys(path)));
    }

    private static List<string> HeaderKeys(string path)
    {
        using FileStream stream = File.OpenRead(path);
        byte[] lengthPrefix = new byte[8];
        stream.ReadExactly(lengthPrefix);
        long headerLength = BinaryPrimitives.ReadInt64LittleEndian(lengthPrefix);
        byte[] headerJson = new byte[headerLength];
        stream.ReadExactly(headerJson);

        using JsonDocument header = JsonDocument.Parse(headerJson);
        List<string> keys = [];
        foreach (JsonProperty property in header.RootElement.EnumerateObject())
        {
            if (property.Name != "__metadata__")
            {
                keys.Add(property.Name);
            }
        }
        return keys;
    }

    private static string ModelsDir() =>
        Environment.GetEnvironmentVariable("HARTSYINFERENCE_MODELS_DIR") ?? string.Empty;
}

/// <summary>Runs only when <c>HARTSYINFERENCE_MODELS_DIR</c> holds the named file, and says so when it does not.</summary>
[AttributeUsage(AttributeTargets.Method, AllowMultiple = false, Inherited = false)]
public sealed class LocalMiniMaxH3ImageBuildFactAttribute : FactAttribute
{
    public LocalMiniMaxH3ImageBuildFactAttribute(string relativePath)
    {
        string? root = Environment.GetEnvironmentVariable("HARTSYINFERENCE_MODELS_DIR");
        if (string.IsNullOrWhiteSpace(root) || !File.Exists(Path.Combine(root, relativePath)))
        {
            Skip = $"SKIPPED: {relativePath} is not on this machine; this build was not run.";
        }
    }
}
