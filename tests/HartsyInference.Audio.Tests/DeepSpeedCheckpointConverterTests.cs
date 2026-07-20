using System.Collections.Generic;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Tests the pure key-normalization for DeepSpeed checkpoints: a flat dict with a literal <c>module.</c>
/// prefix is stripped to the bare resemble-enhance key layout, and an already-clean dict passes through
/// unchanged.</summary>
public sealed class DeepSpeedCheckpointConverterTests
{
    private static Tensor Scalar() => new(new TensorShape(1), DType.F32);

    [Fact]
    public void StripsModulePrefix_WhenPresent()
    {
        Dictionary<string, Tensor> raw = new()
        {
            ["module.denoiser.net.in_conv.weight"] = Scalar(),
            ["module.lcfm.cfm.net.start.weight"] = Scalar(),
            ["module.vocoder.conv_pre.weight"] = Scalar(),
        };
        Dictionary<string, Tensor> outMap = DeepSpeedCheckpointConverter.StripModulePrefix(raw);
        Assert.Contains("denoiser.net.in_conv.weight", outMap.Keys);
        Assert.Contains("lcfm.cfm.net.start.weight", outMap.Keys);
        Assert.Contains("vocoder.conv_pre.weight", outMap.Keys);
        Assert.DoesNotContain("module.denoiser.net.in_conv.weight", outMap.Keys);
        foreach (Tensor t in raw.Values) t.Dispose();
    }

    [Fact]
    public void PassesThroughCleanKeys_Unchanged()
    {
        Dictionary<string, Tensor> raw = new()
        {
            ["denoiser.net.in_conv.weight"] = Scalar(),
            ["lcfm.ae.decoder.in_conv.weight"] = Scalar(),
        };
        Dictionary<string, Tensor> outMap = DeepSpeedCheckpointConverter.StripModulePrefix(raw);
        Assert.Equal(raw.Count, outMap.Count);
        Assert.Contains("denoiser.net.in_conv.weight", outMap.Keys);
        Assert.Contains("lcfm.ae.decoder.in_conv.weight", outMap.Keys);
        foreach (Tensor t in raw.Values) t.Dispose();
    }
}
