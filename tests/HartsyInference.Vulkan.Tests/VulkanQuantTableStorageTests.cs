using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Vulkan.Tests;

/// <summary>CPU-only checks of the device sizes given to byte-packed quantized tables, which shaders read one 32-bit word at
/// a time. Opens no device, so it runs in the CPU lane; it checks the byte counts the allocator is handed, not a kernel.</summary>
public sealed class VulkanQuantTableStorageTests
{
    [Theory]
    [InlineData(0, 0)]
    [InlineData(1, 4)]
    [InlineData(4, 4)]
    [InlineData(5, 8)]
    [InlineData(22, 24)]
    public void VkBufferBytes_RoundsUpToAWholeWord(ulong requested, ulong expected)
        => Assert.Equal(expected, VulkanBufferFactory.VkBufferBytes(requested));

    [Theory]
    [InlineData("Q5_0", 3, 72)]    // 66-byte table, odd block count
    [InlineData("Q5_0", 5, 116)]   // 110 bytes, odd block count
    [InlineData("Q5_0", 4, 92)]    // 88 bytes, even control
    [InlineData("Q6_K", 1, 216)]   // 210 bytes, one super-block
    [InlineData("Q6_K", 3, 636)]   // 630 bytes, odd super-block count
    [InlineData("Q6_K", 4, 844)]   // 840 bytes, even control
    public void PreloadedQuantTable_IsAllocatedWithWordRoundingAndSlack(string dtypeName, int blocks, long expectedBytes)
    {
        DType dtype = DTypeOf(dtypeName);
        long logical = dtype.ComputeByteCount((long)blocks * dtype.BlockElementCount);
        long allocated = (long)VulkanBufferFactory.VkBufferBytes((ulong)VulkanGpuTransferHelper.WeightStorageBytes(logical));

        Assert.Equal(expectedBytes, allocated);
        Assert.Equal(0L, allocated % 4);
        Assert.True(allocated - logical >= 4, "the allocation keeps at least one word of slack past the table");
        // Every byte a shader can index is read through the word that holds it; that word must end inside the allocation.
        for (long b = 0; b < logical; b++)
        {
            long wordEnd = ((b >> 2) + 1) * 4;
            Assert.True(wordEnd <= allocated, $"byte {b} of a {logical}-byte {dtypeName} table is read through a word ending at {wordEnd}, past the {allocated}-byte allocation");
        }
    }

    [Fact]
    public void PreloadedWeight_OfZeroBytes_StaysZero()
        => Assert.Equal(0L, VulkanGpuTransferHelper.WeightStorageBytes(0));

    private static DType DTypeOf(string name) => name switch
    {
        "Q5_0" => DType.Q5_0,
        "Q6_K" => DType.Q6_K,
        _ => throw new ArgumentException($"no table layout for {name}", nameof(name)),
    };
}
