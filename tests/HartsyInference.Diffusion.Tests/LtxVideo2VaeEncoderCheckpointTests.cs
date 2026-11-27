using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Config-derived encoder keys and shapes against the real SANA-WM VAE header. Reads the header only.</summary>
[Trait("Category", "Integration")]
public sealed class LtxVideo2VaeEncoderCheckpointTests
{
    private const string VaeVariable = "HARTSY_SANA_WM_VAE_SAFETENSORS";

    [RealWeightsFact(VaeVariable)]
    public void CheckpointEncoderKeysAndShapesMatchConfigDerivation()
    {
        SafeTensorHeader header = SafeTensorHeaderReader.Read(Environment.GetEnvironmentVariable(VaeVariable)!);
        IReadOnlyDictionary<string, TensorShape> expected = LtxVideo2VaeEncoder.ExpectedWeightShapes(new LtxVideo2VaeEncoderConfig());

        Dictionary<string, TensorShape> actual = new(StringComparer.Ordinal);
        foreach (KeyValuePair<string, SafeTensorDescriptor> entry in header.Tensors)
        {
            if (entry.Key.StartsWith("encoder.", StringComparison.Ordinal)) actual[entry.Key] = entry.Value.Shape;
        }

        Assert.Equal(92, actual.Count);
        Assert.Equal(expected.Count, actual.Count);
        foreach (KeyValuePair<string, TensorShape> entry in expected)
        {
            Assert.True(actual.TryGetValue(entry.Key, out TensorShape shape), $"checkpoint has no {entry.Key}");
            Assert.Equal(entry.Value, shape);
        }
    }
}
