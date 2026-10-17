using System.Runtime.InteropServices;
using System.Security.Cryptography;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Vulkan.Tests;

/// <summary>Device-indexed row gather from an F16 or BF16 embed table (embed_gather_decode_f16 / _bf16) against the host
/// widening of the same rows, compared as bit patterns. Mirrors the CUDA EmbedGatherNarrowTableTests: the ordinary values
/// come from the same generator, so both backends are checked against the same reference. The table also carries
/// subnormal and negative-zero patterns, the values a device that flushes subnormals would get wrong.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class VulkanEmbedGatherNarrowTableTests
{
    private const int Vocab = 16;
    private const int Hidden = 96;

    private static bool VulkanAvailable()
    {
        try { using VulkanInstance i = new(); return i.EnumeratePhysicalDevices().Length > 0; }
        catch { return false; }
    }

    /// <summary>Raw 16-bit table. Ordinary entries are the CUDA test's values (multiples of 0.25 in [-25, 25]); every
    /// eleventh entry is a positive subnormal, every eleventh-plus-one a negative one, and every 22nd a negative zero.</summary>
    private static ushort[] MakeRawTable(bool bf16)
    {
        // Largest subnormal magnitude bits: BF16 has 7 mantissa bits, F16 has 10.
        int subnormalSpan = bf16 ? 0x7F : 0x3FF;
        ushort[] raw = new ushort[Vocab * Hidden];
        for (int i = 0; i < raw.Length; i++)
        {
            float value = ((i * 37) % 201 - 100) * 0.25f;
            raw[i] = bf16
                ? (ushort)((uint)BitConverter.SingleToInt32Bits(value) >> 16)
                : BitConverter.HalfToUInt16Bits((Half)value);
            if (i % 22 == 7) raw[i] = 0x8000;
            else if (i % 11 == 5) raw[i] = (ushort)(1 + (i * 13) % subnormalSpan);
            else if (i % 11 == 6) raw[i] = (ushort)(0x8000 | (1 + (i * 17) % subnormalSpan));
        }
        return raw;
    }

    /// <summary>The host widening of one raw bit pattern: exact for every input, subnormals included.</summary>
    private static int WidenToF32Bits(ushort raw, bool bf16) => bf16
        ? (int)((uint)raw << 16)
        : BitConverter.SingleToInt32Bits((float)BitConverter.UInt16BitsToHalf(raw));

    private static string Sha256Hex(ReadOnlySpan<int> bits) => Convert.ToHexString(SHA256.HashData(MemoryMarshal.AsBytes(bits)));

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void EmbedGatherDecodeStep_NarrowTable_MatchesHostWidening(bool bf16)
    {
        if (!VulkanAvailable()) { Console.Error.WriteLine("SKIPPED: no Vulkan device"); return; }

        using VulkanBackend backend = VulkanTestDevice.Create();
        string variant = bf16 ? "bf16" : "f16";
        Console.Error.WriteLine($"Vulkan device for embed gather {variant}: {backend.Vk.DeviceName}");

        ushort[] raw = MakeRawTable(bf16);
        Tensor table = new(new TensorShape(Vocab, Hidden), bf16 ? DType.BF16 : DType.F16);
        raw.AsSpan().CopyTo(table.AsSpan<ushort>());

        int[] rows = [0, 7, Vocab - 1];
        int[] expected = new int[rows.Length * Hidden];
        int[] gathered = new int[rows.Length * Hidden];
        for (int r = 0; r < rows.Length; r++)
        {
            for (int c = 0; c < Hidden; c++)
            {
                expected[r * Hidden + c] = WidenToF32Bits(raw[rows[r] * Hidden + c], bf16);
            }
        }

        backend.PreloadWeights([table]);
        ulong token = backend.AllocDeviceTokenId();
        try
        {
            for (int r = 0; r < rows.Length; r++)
            {
                backend.WriteDeviceTokenId(token, rows[r]);
                Tensor output = new(new TensorShape(1, 1, Hidden), DType.F32);
                try
                {
                    backend.EmbedGatherDecodeStep(output, table, token);
                    ReadOnlySpan<float> got = output.AsReadOnlySpan<float>();
                    for (int c = 0; c < Hidden; c++)
                    {
                        gathered[r * Hidden + c] = BitConverter.SingleToInt32Bits(got[c]);
                        Assert.Equal(expected[r * Hidden + c], gathered[r * Hidden + c]);
                    }
                }
                finally
                {
                    output.Dispose();
                }
            }
        }
        finally
        {
            backend.FreeDeviceTokenId(token);
            backend.FreeWeights([table]);
            table.Dispose();
        }

        // Same bits as the host reference, so the digest is the one a CUDA run of the same generator must print.
        Console.Error.WriteLine($"EMBED_GATHER_DIGEST vulkan {variant} rows=0,7,{Vocab - 1} sha256={Sha256Hex(gathered)}");
        Assert.Equal(Sha256Hex(expected), Sha256Hex(gathered));
    }

    [Fact]
    public void EmbedGatherDecodeStep_NarrowTableWithOddElementCount_IsRefused()
    {
        if (!VulkanAvailable()) { Console.Error.WriteLine("SKIPPED: no Vulkan device"); return; }

        using VulkanBackend backend = VulkanTestDevice.Create();
        Tensor table = new(new TensorShape(3, 5), DType.F16);
        ulong token = backend.AllocDeviceTokenId();
        Tensor output = new(new TensorShape(1, 1, 5), DType.F32);
        try
        {
            backend.WriteDeviceTokenId(token, 1);
            Assert.Throws<NotSupportedException>(() => backend.EmbedGatherDecodeStep(output, table, token));
        }
        finally
        {
            output.Dispose();
            backend.FreeDeviceTokenId(token);
            table.Dispose();
        }
    }
}
