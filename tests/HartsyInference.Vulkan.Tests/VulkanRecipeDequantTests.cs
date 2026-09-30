using HartsyInference.Core.Tensors;
using HartsyInference.Core.Tensors.Quant;
using HartsyInference.ModelAssets.BlockScale;
using Xunit;
using Xunit.Abstractions;
using static HartsyInference.Vulkan.Tests.Dsv41TestData;

namespace HartsyInference.Vulkan.Tests;

/// <summary>Device dequant of MXFP4-E8M0 and block-FP8-E8M0 recipes to BF16 against the host codecs, bit for bit.
/// Skips without a Vulkan device. NVIDIA is plumbing evidence only.</summary>
[Trait("Category", "GpuIntegration")]
public sealed unsafe class VulkanRecipeDequantTests(ITestOutputHelper log)
{
    // Round to nearest even, canonical quiet NaN: what the kernel and the CUDA kernel store.
    private static ushort ToBf16(float value)
    {
        uint bits = BitConverter.SingleToUInt32Bits(value);
        if (float.IsNaN(value)) return 0x7FC0;
        bits += 0x7FFFu + ((bits >> 16) & 1u);
        return (ushort)(bits >> 16);
    }

    private static (Tensor Packed, QuantRecipe Recipe) Build(QuantEncoding enc, int blockRows, int blockCols, int rows, int cols,
        int scaleColOffset, int seed)
    {
        Random rng = new(seed);
        BlockGeometry geometry = new(blockRows, blockCols);
        (long scaleRows, long scaleCols) = geometry.ScaleShape(rows, cols);
        Tensor scale = new(new TensorShape(scaleRows, scaleCols + scaleColOffset), DType.F8E8M0);
        Span<byte> scaleBytes = scale.AsSpan<byte>();
        for (int i = 0; i < scaleBytes.Length; i++) scaleBytes[i] = (byte)rng.Next(96, 160);
        // Subnormal 2^-127, NaN, and the extremes that overflow or underflow a float product.
        byte[] specials = [0, 255, 1, 254, 127, 0, 255];
        for (int i = 0; i < specials.Length; i++)
            scaleBytes[Math.Min(scaleColOffset + i * 3, scaleBytes.Length - 1)] = specials[i];

        bool fp4 = enc == QuantEncoding.Mxfp4E8M0;
        Tensor packed = new(fp4 ? new TensorShape(rows, cols / 2) : new TensorShape(rows, cols), fp4 ? DType.I8 : DType.F8E4M3);
        rng.NextBytes(packed.AsSpan<byte>());
        Span<byte> data = packed.AsSpan<byte>();
        if (!fp4 && data.Length > 4)
        {
            data[0] = 0x7F; data[1] = 0xFF; data[2] = 0x80; data[3] = 0x00;
        }
        else if (fp4 && data.Length > 2)
        {
            data[0] = 0x88; data[1] = 0x8F;
        }
        QuantRecipe recipe = new()
        {
            Encoding = enc, Geometry = geometry, ScaleDType = DType.F8E8M0,
            LogicalRows = rows, LogicalCols = cols, Scale = scale, ScaleColOffset = scaleColOffset,
        };
        return (packed, recipe);
    }

    private static ushort[] HostReference(Tensor packed, QuantRecipe recipe)
    {
        float[] dense = new float[recipe.LogicalRows * recipe.LogicalCols];
        ReadOnlySpan<byte> bytes = packed.AsReadOnlySpan<byte>();
        if (recipe.Encoding == QuantEncoding.Mxfp4E8M0) Mxfp4E8M0Codec.DequantRows(bytes, recipe, 0, recipe.LogicalRows, dense);
        else Fp8BlockE8M0Codec.DequantRows(bytes, recipe, 0, recipe.LogicalRows, dense);
        ushort[] bf16 = new ushort[dense.Length];
        for (int i = 0; i < dense.Length; i++) bf16[i] = ToBf16(dense[i]);
        return bf16;
    }

    [Theory]
    [InlineData(QuantEncoding.Mxfp4E8M0, 1, 32, 5, 128, 0)]
    [InlineData(QuantEncoding.Mxfp4E8M0, 1, 32, 7, 96, 2)]
    [InlineData(QuantEncoding.Mxfp4E8M0, 1, 32, 3, 64, 0)]
    [InlineData(QuantEncoding.Mxfp4E8M0, 1, 32, 2, 4096, 0)]
    [InlineData(QuantEncoding.Fp8E4M3BlockE8M0, 32, 32, 70, 96, 0)]
    [InlineData(QuantEncoding.Fp8E4M3BlockE8M0, 32, 32, 33, 64, 1)]
    [InlineData(QuantEncoding.Fp8E4M3BlockE8M0, 128, 128, 200, 300, 0)]
    [InlineData(QuantEncoding.Fp8E4M3BlockE8M0, 32, 32, 3, 10, 0)]
    [InlineData(QuantEncoding.Fp8E4M3BlockE8M0, 32, 32, 5, 6, 0)]
    public void DequantToBf16_IsBitIdenticalToHostCodec(QuantEncoding enc, int blockRows, int blockCols, int rows, int cols,
        int scaleColOffset)
    {
        using VulkanBackend? vk = TryCreateBackend(out string? skip);
        if (vk is null) { log.WriteLine($"SKIPPED: {skip}"); return; }
        (Tensor packed, QuantRecipe recipe) = Build(enc, blockRows, blockCols, rows, cols, scaleColOffset, seed: rows * 31 + cols);
        using Tensor packedOwner = packed;
        using Tensor scaleOwner = recipe.Scale!;
        Assert.True(vk.SupportsRecipeDequant(recipe));
        ushort[] expected = HostReference(packed, recipe);

        using Tensor output = new(new TensorShape(rows, cols), DType.BF16);
        vk.DequantRecipeToBf16(packed, recipe, output);
        ushort[] actual = new ushort[expected.Length];
        new ReadOnlySpan<ushort>((ushort*)output.DataPointer, actual.Length).CopyTo(actual);

        int mismatches = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            if (expected[i] != actual[i] && mismatches++ < 5)
            {
                log.WriteLine($"[{i}] host 0x{expected[i]:X4} vulkan 0x{actual[i]:X4}");
            }
        }
        Assert.Equal(0, mismatches);
        log.WriteLine($"{enc} {blockRows}x{blockCols} {rows}x{cols} scaleColOffset={scaleColOffset}: {expected.Length} values identical");
    }

    [Fact]
    public void Unsupported_RecipesAreRefusedWithAReason()
    {
        using VulkanBackend? vk = TryCreateBackend(out string? skip);
        if (vk is null) { log.WriteLine($"SKIPPED: {skip}"); return; }
        (Tensor packed, QuantRecipe recipe) = Build(QuantEncoding.Mxfp4E8M0, 1, 32, 2, 64, 0, seed: 1);
        using Tensor packedOwner = packed;
        using Tensor scaleOwner = recipe.Scale!;

        QuantRecipe nvfp4 = recipe with { Encoding = QuantEncoding.Nvfp4 };
        Assert.False(vk.SupportsRecipeDequant(nvfp4));
        using Tensor output = new(new TensorShape(2, 64), DType.BF16);
        NotSupportedException ex = Assert.Throws<NotSupportedException>(() => vk.DequantRecipeToBf16(packed, nvfp4, output));
        Assert.Contains("Nvfp4", ex.Message);

        Assert.False(vk.SupportsRecipeDequant(recipe with { ScaleLayout = ScaleLayout.Swizzled128 }));
        Assert.False(vk.SupportsRecipeDequant(recipe with { ScaleDType = DType.F32 }));
        // Odd element counts cannot be stored as BF16 word pairs.
        Assert.False(vk.SupportsRecipeDequant(recipe with { Encoding = QuantEncoding.Fp8E4M3BlockE8M0, LogicalRows = 3, LogicalCols = 5 }));
        // Whole-word ScaleColOffset overrun: a scale too small for the recipe.
        Assert.False(vk.SupportsRecipeDequant(recipe with { ScaleColOffset = 50 }));
        // Resident packed weights stay unsupported: there is no block-scaled GEMM to read them.
        using Tensor weight = new(new TensorShape(2, 32), DType.I8);
        Assert.False(((HartsyInference.Core.Backends.IBackend)vk).SupportsResidentQuant(weight));
    }
}
