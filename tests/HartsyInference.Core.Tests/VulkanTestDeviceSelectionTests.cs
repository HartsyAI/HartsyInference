using HartsyInference.Tests.Common;
using Xunit;

namespace HartsyInference.Core.Tests;

/// <summary>The Vulkan test pin. A set <c>HARTSY_TEST_VULKAN_DEVICE</c> names exactly one ordinal, and a value that
/// names none throws instead of falling back to the default picker, which is the RTX 4090 on the dev box. Nothing
/// here touches a GPU.</summary>
public sealed class VulkanTestDeviceSelectionTests
{
    [Theory]
    [InlineData(null, null)]
    [InlineData("", null)]
    [InlineData("   ", null)]
    [InlineData("0", 0)]
    [InlineData("1", 1)]
    [InlineData(" 1 ", 1)]
    [InlineData("12", 12)]
    public void Parse_UnsetOrAnOrdinal(string? raw, int? expected) =>
        Assert.Equal(expected, VulkanTestDeviceSelection.Parse(raw));

    [Theory]
    [InlineData("-1")]
    [InlineData("gpu1")]
    [InlineData("1.0")]
    [InlineData("0x1")]
    [InlineData("1,2")]
    public void Parse_RejectsAnythingThatIsNotAnOrdinal(string raw) =>
        Assert.Throws<FormatException>(() => VulkanTestDeviceSelection.Parse(raw));
}
