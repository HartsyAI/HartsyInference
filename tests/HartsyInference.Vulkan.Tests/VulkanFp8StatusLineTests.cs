using HartsyInference.Vulkan;
using Xunit;

namespace HartsyInference.Vulkan.Tests;

/// <summary>The once-per-device fp8 line a Vulkan backend logs at construction: the reason when off, how to opt in while the
/// path is off by default, and a warning only when the user forced it on and the device cannot run it.</summary>
public sealed class VulkanFp8StatusLineTests
{
    private const string NoExt = "the driver does not offer VK_EXT_shader_float8 (NVIDIA 595 or newer does)";

    [Fact]
    public void UnsetKnob_WithoutTheDriver_SaysWhyAtInfo()
    {
        (bool warn, string message) = VulkanBackend.Fp8StatusLine(null, false, NoExt, 0, 0, 0);
        Assert.False(warn);
        Assert.Contains(NoExt, message);
        Assert.Contains("widened to F16", message);
    }

    [Fact]
    public void ForcedOn_WithoutTheDriver_Warns()
    {
        (bool warn, string message) = VulkanBackend.Fp8StatusLine(true, false, NoExt, 0, 0, 0);
        Assert.True(warn);
        Assert.Contains("numerics.vkFp8 is on", message);
        Assert.Contains(NoExt, message);
    }

    [Fact]
    public void ForcedOn_Available_NamesTheShape()
    {
        (bool warn, string message) = VulkanBackend.Fp8StatusLine(true, true, null, 16, 16, 32);
        Assert.False(warn);
        Assert.Contains("on (E4M3 16x16x32", message);
    }

    [Fact]
    public void UnsetKnob_Available_StaysOffAndSaysHowToOptIn()
    {
        (bool warn, string message) = VulkanBackend.Fp8StatusLine(null, true, null, 16, 16, 32);
        Assert.False(warn);
        Assert.Contains("off until validated", message);
        Assert.Contains("numerics.vkFp8=true", message);
    }

    [Fact]
    public void ForcedOff_SaysSoEvenWhenAvailable()
    {
        (bool warn, string message) = VulkanBackend.Fp8StatusLine(false, true, null, 16, 16, 32);
        Assert.False(warn);
        Assert.Contains("numerics.vkFp8=false", message);
    }
}
