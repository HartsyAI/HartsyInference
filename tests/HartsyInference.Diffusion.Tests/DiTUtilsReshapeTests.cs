using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The in-place <see cref="DiTUtils.ReshapeToMultiHead(Tensor, Tensor, int, int, int, int)"/> and
/// <see cref="DiTUtils.ReshapeFromMultiHead(Tensor, Tensor, int, int, int, int)"/> are what <c>Lumina2Block</c>,
/// <c>Lumina2ContextRefinerBlock</c> and <c>T5Block</c> call now. Each of those blocks used to carry a private copy; the copies
/// are reproduced here as oracles (the element loop <c>T5Block</c> had, and the <c>Buffer.MemoryCopy</c> form both Lumina blocks had),
/// and the surviving helper must match both bit for bit on odd shapes, including values that are NaN, infinite or subnormal.</summary>
public sealed unsafe class DiTUtilsReshapeTests
{
    private static Tensor Filled(TensorShape shape, int seed)
    {
        Tensor tensor = new(shape, DType.F32);
        Random random = new(seed);
        Span<uint> bits = System.Runtime.InteropServices.MemoryMarshal.Cast<float, uint>(tensor.AsSpan<float>());
        for (int i = 0; i < bits.Length; i++)
        {
            bits[i] = (uint)random.NextInt64(0, 1L << 32);
        }
        return tensor;
    }

    private static void ToMultiHeadElementLoop(Tensor output, Tensor input, int batch, int seqLen, int numHeads, int headDim)
    {
        float* inPtr = (float*)input.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                for (int h = 0; h < numHeads; h++)
                {
                    int inOffset = (b * seqLen + s) * (numHeads * headDim) + h * headDim;
                    int outOffset = ((b * numHeads + h) * seqLen + s) * headDim;
                    for (int d = 0; d < headDim; d++)
                    {
                        outPtr[outOffset + d] = inPtr[inOffset + d];
                    }
                }
            }
        }
    }

    private static void FromMultiHeadElementLoop(Tensor output, Tensor input, int batch, int seqLen, int numHeads, int headDim)
    {
        float* inPtr = (float*)input.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                for (int h = 0; h < numHeads; h++)
                {
                    int inOffset = ((b * numHeads + h) * seqLen + s) * headDim;
                    int outOffset = (b * seqLen + s) * (numHeads * headDim) + h * headDim;
                    for (int d = 0; d < headDim; d++)
                    {
                        outPtr[outOffset + d] = inPtr[inOffset + d];
                    }
                }
            }
        }
    }

    private static void ToMultiHeadMemoryCopy(Tensor output, Tensor input, int batch, int seqLen, int numHeads, int headDim)
    {
        float* inPtr = (float*)input.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                for (int h = 0; h < numHeads; h++)
                {
                    int inOffset = (b * seqLen + s) * (numHeads * headDim) + h * headDim;
                    int outOffset = ((b * numHeads + h) * seqLen + s) * headDim;
                    Buffer.MemoryCopy(inPtr + inOffset, outPtr + outOffset, headDim * sizeof(float), headDim * sizeof(float));
                }
            }
        }
    }

    private static void FromMultiHeadMemoryCopy(Tensor output, Tensor input, int batch, int seqLen, int numHeads, int headDim)
    {
        float* inPtr = (float*)input.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                for (int h = 0; h < numHeads; h++)
                {
                    int inOffset = ((b * numHeads + h) * seqLen + s) * headDim;
                    int outOffset = (b * seqLen + s) * (numHeads * headDim) + h * headDim;
                    Buffer.MemoryCopy(inPtr + inOffset, outPtr + outOffset, headDim * sizeof(float), headDim * sizeof(float));
                }
            }
        }
    }

    private static void AssertSameBits(Tensor expected, Tensor actual, string what)
    {
        ReadOnlySpan<uint> e = System.Runtime.InteropServices.MemoryMarshal.Cast<float, uint>(expected.AsReadOnlySpan<float>());
        ReadOnlySpan<uint> a = System.Runtime.InteropServices.MemoryMarshal.Cast<float, uint>(actual.AsReadOnlySpan<float>());
        Assert.Equal(e.Length, a.Length);
        for (int i = 0; i < e.Length; i++)
        {
            Assert.True(e[i] == a[i], $"{what}: element {i} differs ({e[i]:x8} against {a[i]:x8})");
        }
    }

    [Theory]
    [InlineData(1, 1, 1, 1)]
    [InlineData(3, 5, 3, 7)]
    [InlineData(2, 7, 5, 3)]
    [InlineData(1, 13, 1, 11)]
    [InlineData(4, 1, 6, 5)]
    [InlineData(2, 9, 2, 1)]
    public void ToMultiHeadMatchesBothDeletedCopiesBitForBit(int batch, int seqLen, int heads, int headDim)
    {
        using Tensor input = Filled(new TensorShape(batch, seqLen, heads * headDim), batch * 1000 + seqLen);
        TensorShape shape = new(batch, heads, seqLen, headDim);
        using Tensor expectedLoop = new(shape, DType.F32), expectedCopy = new(shape, DType.F32), actual = new(shape, DType.F32);
        ToMultiHeadElementLoop(expectedLoop, input, batch, seqLen, heads, headDim);
        ToMultiHeadMemoryCopy(expectedCopy, input, batch, seqLen, heads, headDim);
        DiTUtils.ReshapeToMultiHead(actual, input, batch, seqLen, heads, headDim);
        AssertSameBits(expectedLoop, actual, "element loop");
        AssertSameBits(expectedCopy, actual, "MemoryCopy");
    }

    [Theory]
    [InlineData(1, 1, 1, 1)]
    [InlineData(3, 5, 3, 7)]
    [InlineData(2, 7, 5, 3)]
    [InlineData(1, 13, 1, 11)]
    [InlineData(4, 1, 6, 5)]
    [InlineData(2, 9, 2, 1)]
    public void FromMultiHeadMatchesBothDeletedCopiesBitForBit(int batch, int seqLen, int heads, int headDim)
    {
        using Tensor input = Filled(new TensorShape(batch, heads, seqLen, headDim), batch * 2000 + seqLen);
        TensorShape shape = new(batch, seqLen, heads * headDim);
        using Tensor expectedLoop = new(shape, DType.F32), expectedCopy = new(shape, DType.F32), actual = new(shape, DType.F32);
        FromMultiHeadElementLoop(expectedLoop, input, batch, seqLen, heads, headDim);
        FromMultiHeadMemoryCopy(expectedCopy, input, batch, seqLen, heads, headDim);
        DiTUtils.ReshapeFromMultiHead(actual, input, batch, seqLen, heads, headDim);
        AssertSameBits(expectedLoop, actual, "element loop");
        AssertSameBits(expectedCopy, actual, "MemoryCopy");
    }

    [Theory]
    [InlineData(3, 5, 3, 7)]
    [InlineData(2, 7, 5, 3)]
    public void ToThenFromMultiHeadRestoresTheInput(int batch, int seqLen, int heads, int headDim)
    {
        using Tensor input = Filled(new TensorShape(batch, seqLen, heads * headDim), 77);
        using Tensor split = new(new TensorShape(batch, heads, seqLen, headDim), DType.F32), merged = new(input.Shape, DType.F32);
        DiTUtils.ReshapeToMultiHead(split, input, batch, seqLen, heads, headDim);
        DiTUtils.ReshapeFromMultiHead(merged, split, batch, seqLen, heads, headDim);
        AssertSameBits(input, merged, "round trip");
    }
}
