using HartsyInference.Vulkan;
using Xunit;

namespace HartsyInference.Vulkan.Tests;

/// <summary>The cm2 prefill mask cap as a pure function. It creates no backend, so it runs on the CPU lane and on a machine
/// with no GPU. The counter-level check (<c>FlashCm2PrefillDispatches</c> unchanged above the cap) is a GPU-lane test in
/// <c>VulkanFlashCm2Tests</c>, because the counter lives on a backend.</summary>
public sealed class VulkanFlashCm2PolicyTests
{
    [Theory]
    [InlineData(1, 1)]
    [InlineData(64, 64)]
    [InlineData(750, 750)]
    [InlineData(4096, 4096)]     // 64 MiB
    [InlineData(8192, 8192)]     // exactly the 256 MiB cap, so it still fits
    [InlineData(1, 32768)]       // a single query row is small whatever the key count
    public void At_Or_Below_The_Cap_The_Cm2_Path_Is_Allowed(int sq, int kvLen) =>
        Assert.True(FlashCm2PrefillPolicy.MaskFits(sq, kvLen));

    [Theory]
    [InlineData(8193, 8192)]     // one query row past 8K by 8K
    [InlineData(8192, 8193)]     // one key past 8K by 8K
    [InlineData(16384, 16384)]   // 1 GiB
    [InlineData(32768, 32768)]   // 4 GiB, the prompt length that filled host memory
    public void Above_The_Cap_The_Cm2_Path_Is_Refused(int sq, int kvLen) =>
        Assert.False(FlashCm2PrefillPolicy.MaskFits(sq, kvLen));

    [Fact]
    public void Mask_Bytes_Are_Four_Per_Query_Key_Pair()
    {
        Assert.Equal(4L * 4096 * 4096, FlashCm2PrefillPolicy.MaskBytes(4096, 4096));
        Assert.Equal(256L << 20, FlashCm2PrefillPolicy.MaskBytes(8192, 8192));
        Assert.Equal(FlashCm2PrefillPolicy.MaxMaskBytes, 256L << 20);
    }

    [Theory]
    [InlineData(0, 4096)]
    [InlineData(4096, 0)]
    [InlineData(-1, 4096)]
    public void Non_Positive_Shapes_Are_Refused(int sq, int kvLen) =>
        Assert.False(FlashCm2PrefillPolicy.MaskFits(sq, kvLen));
}
