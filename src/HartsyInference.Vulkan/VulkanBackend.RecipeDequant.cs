using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Core.Tensors.Quant;

namespace HartsyInference.Vulkan;

// DeepSeek-V4.1 checkpoint recipe dequantization to BF16. This is a dequant path only: Vulkan has no GEMM that reads
// block-scaled weights, so SupportsResidentQuant(Tensor) stays false and a recipe weight is widened once, then used as BF16.
public sealed partial class VulkanBackend
{
    /// <summary>True when <see cref="DequantRecipeToBf16"/> can decode <paramref name="recipe"/>: MXFP4-E8M0 or block-FP8-E8M0 with row-major E8M0 scale bytes.</summary>
    public bool SupportsRecipeDequant(QuantRecipe recipe)
    {
        ArgumentNullException.ThrowIfNull(recipe);
        return RecipeDequantProblem(recipe) is null;
    }

    /// <summary>Dequantizes a whole recipe matrix into a BF16 <c>[LogicalRows, LogicalCols]</c> tensor with one round-to-nearest-even store per element.</summary>
    /// <param name="packed">The packed matrix, one byte per element for FP8 and one per two for MXFP4.</param>
    /// <param name="recipe">The recipe describing the bytes and the scale tensor.</param>
    /// <param name="output">A BF16 tensor of the recipe's logical shape with an even element count (results are stored as word pairs).</param>
    public void DequantRecipeToBf16(Tensor packed, QuantRecipe recipe, Tensor output)
    {
        ArgumentNullException.ThrowIfNull(packed);
        ArgumentNullException.ThrowIfNull(recipe);
        ArgumentNullException.ThrowIfNull(output);
        string? problem = RecipeDequantProblem(recipe);
        if (problem is not null)
        {
            throw new NotSupportedException($"Vulkan recipe dequant: {problem}");
        }
        long total = recipe.LogicalRows * recipe.LogicalCols;
        long packedBytes = total / recipe.ElementsPerByte;
        if (packed.ElementCount * packed.DType.SizeInBytes != packedBytes)
        {
            throw new ArgumentException($"Packed weight is {packed.ElementCount * packed.DType.SizeInBytes} bytes; the recipe needs {packedBytes}.", nameof(packed));
        }
        if (output.DType != DType.BF16 || output.ElementCount != total)
        {
            throw new ArgumentException($"Output must be BF16 with {total} elements; got {output.DType} {output.Shape}.", nameof(output));
        }
        if (total == 0)
        {
            return;
        }
        using OpScope _op = EnterOp();
        List<VulkanBuffer> temps = new();
        VulkanBuffer outBuf = _xfer.AllocateDevice((ulong)(total * 2));
        try
        {
            VulkanBuffer packedBuf = PaddedInput(packed, temps);
            VulkanBuffer scaleBuf = PaddedInput(recipe.Scale!, temps);
            VulkanKernel kernel = GetKernel("dequant_recipe_bf16", storageBufferCount: 3, _default1DSpec);
            Span<ulong> bufs = stackalloc ulong[] { outBuf.Handle, packedBuf.Handle, scaleBuf.Handle };
            Span<byte> pcBytes = stackalloc byte[(int)VulkanDescriptorManager.PushConstantRangeBytes];
            long pairs = total / 2;
            for (long done = 0; done < pairs; done += MaxElementsPerDispatch)
            {
                Authoring.PushConstants pc = new(pcBytes);
                pc.U32(recipe.Encoding == QuantEncoding.Mxfp4E8M0 ? 0u : 1u);
                pc.U32((uint)recipe.LogicalCols);
                pc.U32((uint)pairs);
                pc.U32((uint)recipe.Scale!.Shape[1]);
                pc.U32((uint)recipe.ScaleColOffset);
                pc.U32((uint)recipe.Geometry.BlockRows);
                pc.U32((uint)recipe.Geometry.BlockCols);
                pc.U32((uint)done);
                Dispatch(kernel, bufs, pc.Written, GroupCount(Math.Min(pairs - done, MaxElementsPerDispatch), LocalX1D));
            }
            CacheOutput(output, outBuf);
        }
        catch (Exception ex)
        {
            Logs.Error("Vulkan DequantRecipeToBf16 dispatch failed", ex);
            outBuf.Dispose();
            throw;
        }
        finally
        {
            FreeTemps(temps);
        }
    }

    // Why a recipe cannot be decoded here, or null. Mirrors the host codecs' checks plus this kernel's 32-bit indexing.
    private static string? RecipeDequantProblem(QuantRecipe recipe)
    {
        if (recipe.Encoding != QuantEncoding.Mxfp4E8M0 && recipe.Encoding != QuantEncoding.Fp8E4M3BlockE8M0)
        {
            return $"{recipe.Encoding} has no Vulkan dequant kernel.";
        }
        if (recipe.ScaleLayout != ScaleLayout.RowMajorBlocks)
        {
            return $"only {ScaleLayout.RowMajorBlocks} scales are read; recipe is {recipe.ScaleLayout}.";
        }
        if (recipe.ScaleDType != DType.F8E8M0 && recipe.ScaleDType != DType.U8)
        {
            return $"scales must be E8M0 bytes; recipe has {recipe.ScaleDType}.";
        }
        Tensor? scale = recipe.Scale;
        if (scale is null || scale.Shape.Rank != 2 || scale.DType.SizeInBytes != 1)
        {
            return "the recipe needs a rank-2 byte scale tensor.";
        }
        if (recipe.Geometry.BlockRows <= 0 || recipe.Geometry.BlockCols <= 0)
        {
            return $"block geometry {recipe.Geometry} is empty.";
        }
        long rows = recipe.LogicalRows, cols = recipe.LogicalCols;
        long total = rows * cols;
        if (total % 2 != 0 || total >= uint.MaxValue / 2)
        {
            return $"{rows}x{cols} elements must be even and under 2^31.";
        }
        if (recipe.Encoding == QuantEncoding.Mxfp4E8M0 && (cols % 2 != 0 || recipe.Geometry.BlockCols % 2 != 0))
        {
            return "MXFP4 needs an even column count and block width.";
        }
        (long scaleRows, long scaleCols) = recipe.Geometry.ScaleShape(rows, cols);
        if (scale.Shape[0] < scaleRows || scale.Shape[1] < recipe.ScaleColOffset + scaleCols)
        {
            return $"scale {scale.Shape} is smaller than [{scaleRows}, {recipe.ScaleColOffset + scaleCols}].";
        }
        return null;
    }
}
