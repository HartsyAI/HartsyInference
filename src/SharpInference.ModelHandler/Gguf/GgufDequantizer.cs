using System.Runtime.InteropServices;
using SharpInference.Core.Tensors;

namespace SharpInference.ModelHandler.Gguf;

/// <summary>Dequantizes GGUF quantized tensors (Q8_0, Q4_K_M) to F16 or F32.</summary>
public static class GgufDequantizer
{
    /// <summary>Dequantizes a quantized tensor to the target dtype (F16 or F32). Returns a new tensor with owned memory containing the dequantized data.</summary>
    public static unsafe Tensor Dequantize(Tensor source, DType targetDtype)
    {
        if (!source.DType.IsQuantized())
            throw new ArgumentException($"Source tensor is not quantized (dtype={source.DType}).", nameof(source));

        if (targetDtype != DType.F32 && targetDtype != DType.F16)
            throw new ArgumentException($"Target dtype must be F32 or F16, got {targetDtype}.", nameof(targetDtype));

        Tensor result = new Tensor(source.Shape, targetDtype);

        if (source.DType == DType.Q8_0)
        {
            if (targetDtype == DType.F32)
                DequantizeQ8_0ToF32((byte*)source.DataPointer, (float*)result.DataPointer, source.Shape.ElementCount);
            else
                DequantizeQ8_0ToF16((byte*)source.DataPointer, (Half*)result.DataPointer, source.Shape.ElementCount);
        }
        else if (source.DType == DType.Q4_K)
        {
            if (targetDtype == DType.F32)
                DequantizeQ4KToF32((byte*)source.DataPointer, (float*)result.DataPointer, source.Shape.ElementCount);
            else
                DequantizeQ4KToF16((byte*)source.DataPointer, (Half*)result.DataPointer, source.Shape.ElementCount);
        }
        else
        {
            throw new SharpInference.Core.Exceptions.SharpInferenceException($"Unsupported quantized type for dequantization: {source.DType}.");
        }

        return result;
    }

    /// <summary>Dequantizes Q8_0 blocks to F32. Each block: 2 bytes FP16 scale + 32 bytes int8 data.</summary>
    private static unsafe void DequantizeQ8_0ToF32(byte* src, float* dst, long elementCount)
    {
        const int BlockSize = 32;
        const int BlockBytes = 34;  // 2 (scale) + 32 (data)

        long numBlocks = (elementCount + BlockSize - 1) / BlockSize;

        for (long b = 0; b < numBlocks; b++)
        {
            byte* block = src + b * BlockBytes;

            // Scale is stored as FP16 (2 bytes)
            Half scaleHalf = *(Half*)block;
            float scale = (float)scaleHalf;

            sbyte* data = (sbyte*)(block + 2);

            long baseIdx = b * BlockSize;
            long count = Math.Min(BlockSize, elementCount - baseIdx);

            for (long i = 0; i < count; i++)
            {
                dst[baseIdx + i] = data[i] * scale;
            }
        }
    }

    /// <summary>Dequantizes Q8_0 blocks to F16.</summary>
    private static unsafe void DequantizeQ8_0ToF16(byte* src, Half* dst, long elementCount)
    {
        const int BlockSize = 32;
        const int BlockBytes = 34;

        long numBlocks = (elementCount + BlockSize - 1) / BlockSize;

        for (long b = 0; b < numBlocks; b++)
        {
            byte* block = src + b * BlockBytes;
            Half scaleHalf = *(Half*)block;
            float scale = (float)scaleHalf;
            sbyte* data = (sbyte*)(block + 2);

            long baseIdx = b * BlockSize;
            long count = Math.Min(BlockSize, elementCount - baseIdx);

            for (long i = 0; i < count; i++)
            {
                dst[baseIdx + i] = (Half)(data[i] * scale);
            }
        }
    }

    /// <summary>Dequantizes Q4_K_M blocks to F32. This is a simplified implementation. Q4_K_M uses super-blocks of 256 elements with nested sub-blocks.</summary>
    private static unsafe void DequantizeQ4KToF32(byte* src, float* dst, long elementCount)
    {
        // Q4_K_M block layout (256 elements per super-block, 144 bytes per block):
        // - 2 bytes: d (FP16 scale)
        // - 2 bytes: dmin (FP16 min)
        // - 12 bytes: scales (6-bit scales for 8 sub-blocks, packed)
        // - 128 bytes: quantized data (4-bit values for 256 elements)
        const int SuperBlockSize = 256;
        const int SuperBlockBytes = 144;
        const int SubBlockSize = 32;

        long numSuperBlocks = (elementCount + SuperBlockSize - 1) / SuperBlockSize;

        for (long sb = 0; sb < numSuperBlocks; sb++)
        {
            byte* block = src + sb * SuperBlockBytes;

            Half dHalf = *(Half*)block;
            float d = (float)dHalf;

            Half dminHalf = *(Half*)(block + 2);
            float dmin = (float)dminHalf;

            byte* scales = block + 4;
            byte* quantData = block + 16;

            long baseIdx = sb * SuperBlockSize;

            for (int subBlock = 0; subBlock < 8; subBlock++)
            {
                // Extract 6-bit scale and min for this sub-block
                float subScale = d * (scales[subBlock] & 0x3F);
                float subMin = dmin * (scales[subBlock + 8 < 12 ? subBlock + 8 : subBlock] >> 4);

                for (int i = 0; i < SubBlockSize; i++)
                {
                    long elemIdx = baseIdx + subBlock * SubBlockSize + i;
                    if (elemIdx >= elementCount) break;

                    int dataIdx = subBlock * SubBlockSize / 2 + i / 2;
                    int nibble = (i % 2 == 0)
                        ? quantData[dataIdx] & 0x0F
                        : (quantData[dataIdx] >> 4) & 0x0F;

                    dst[elemIdx] = nibble * subScale - subMin;
                }
            }
        }
    }

    /// <summary>Dequantizes Q4_K_M blocks to F16.</summary>
    private static unsafe void DequantizeQ4KToF16(byte* src, Half* dst, long elementCount)
    {
        // Dequantize to F32 first, then convert
        // This is simpler and avoids precision loss in intermediate calculations
        nuint byteCount = (nuint)(elementCount * sizeof(float));
        float* tempPtr = (float*)NativeMemory.Alloc(byteCount);
        try
        {
            DequantizeQ4KToF32(src, tempPtr, elementCount);

            for (long i = 0; i < elementCount; i++)
            {
                dst[i] = (Half)tempPtr[i];
            }
        }
        finally
        {
            NativeMemory.Free(tempPtr);
        }
    }
}
