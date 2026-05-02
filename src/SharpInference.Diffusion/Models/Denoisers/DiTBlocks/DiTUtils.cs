using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Shared utility methods for DiT/MMDiT transformers. Consolidates helpers duplicated across FluxTransformer, Sd3Transformer, and future DiT models.</summary>
public static unsafe class DiTUtils
{
    /// <summary>Unparameterized LayerNorm (no learned scale/bias). Normalizes the last dimension to zero mean and unit variance.</summary>
    public static void LayerNormNoAffine(Tensor output, Tensor input, int batch, int seqLen, int dim, float eps = 1e-6f)
    {
        float* inPtr = (float*)input.DataPointer;
        float* outPtr = (float*)output.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                int offset = (b * seqLen + s) * dim;

                float mean = 0f;
                for (int d = 0; d < dim; d++)
                    mean += inPtr[offset + d];
                mean /= dim;

                float variance = 0f;
                for (int d = 0; d < dim; d++)
                {
                    float diff = inPtr[offset + d] - mean;
                    variance += diff * diff;
                }
                variance /= dim;

                float invStd = 1.0f / MathF.Sqrt(variance + eps);
                for (int d = 0; d < dim; d++)
                    outPtr[offset + d] = (inPtr[offset + d] - mean) * invStd;
            }
        }
    }

    /// <summary>Sinusoidal timestep embedding with flip_sin_to_cos=True. Output: [cos_0..cos_halfDim, sin_0..sin_halfDim]. Standard for SD3, Flux, and most DiT models.</summary>
    /// <param name="output">Output tensor [batch, embDim].</param>
    /// <param name="timestep">Scalar timestep value.</param>
    /// <param name="batch">Batch size.</param>
    /// <param name="embDim">Embedding dimension (default 256 for SD3/Flux).</param>
    /// <param name="maxPeriod">Maximum period for frequency computation (default 10000).</param>
    public static void SinusoidalTimestepEmbedding(Tensor output, float timestep, int batch, int embDim = 256, float maxPeriod = 10000.0f)
    {
        float* outPtr = (float*)output.DataPointer;
        int halfDim = embDim / 2;

        for (int b = 0; b < batch; b++)
        {
            int baseOffset = b * embDim;
            for (int i = 0; i < halfDim; i++)
            {
                float freq = MathF.Exp(-MathF.Log(maxPeriod) * i / halfDim);
                float angle = timestep * freq;
                outPtr[baseOffset + i] = MathF.Cos(angle);
                outPtr[baseOffset + halfDim + i] = MathF.Sin(angle);
            }
        }
    }

    /// <summary>Linear projection for 1D vectors: output = input @ weight^T + bias. Input: [B, inDim], Output: [B, outDim].</summary>
    public static void LinearProject1D(Tensor output, Tensor input, Tensor weight, Tensor bias, int batch, int inDim, int outDim)
    {
        float* inPtr = (float*)input.DataPointer;
        float* wPtr = (float*)weight.DataPointer;
        float* bPtr = (float*)bias.DataPointer;
        float* outPtr = (float*)output.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            int inOffset = b * inDim;
            int outOffset = b * outDim;
            for (int o = 0; o < outDim; o++)
            {
                float sum = bPtr[o];
                int wOffset = o * inDim;
                for (int i = 0; i < inDim; i++)
                    sum += inPtr[inOffset + i] * wPtr[wOffset + i];
                outPtr[outOffset + o] = sum;
            }
        }
    }

    /// <summary>Batched linear projection: output = input @ weight^T + bias. Input: [B, S, inDim], Output: [B, S, outDim].</summary>
    public static void LinearProjectBatched(Tensor output, Tensor input, Tensor weight, Tensor bias, int batch, int seqLen, int inDim, int outDim)
    {
        float* inPtr = (float*)input.DataPointer;
        float* wPtr = (float*)weight.DataPointer;
        float* bPtr = (float*)bias.DataPointer;
        float* outPtr = (float*)output.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                int inOffset = (b * seqLen + s) * inDim;
                int outOffset = (b * seqLen + s) * outDim;
                for (int o = 0; o < outDim; o++)
                {
                    float sum = bPtr[o];
                    int wOffset = o * inDim;
                    for (int i = 0; i < inDim; i++)
                        sum += inPtr[inOffset + i] * wPtr[wOffset + i];
                    outPtr[outOffset + o] = sum;
                }
            }
        }
    }

    /// <summary>Concatenates two [B, S1, D] and [B, S2, D] tensors along the sequence dimension → [B, S1+S2, D].</summary>
    public static Tensor ConcatAlongSeqDim(Tensor a, Tensor b)
    {
        int batch = (int)a.Shape[0];
        int seqA = (int)a.Shape[1];
        int seqB = (int)b.Shape[1];
        int dim = (int)a.Shape[2];
        int seqOut = seqA + seqB;

        TensorShape outShape = new TensorShape(batch, seqOut, dim);
        Tensor output = new Tensor(outShape, DType.F32);

        float* aPtr = (float*)a.DataPointer;
        float* bPtr = (float*)b.DataPointer;
        float* outPtr = (float*)output.DataPointer;

        for (int bIdx = 0; bIdx < batch; bIdx++)
        {
            int aSrcOffset = bIdx * seqA * dim;
            int aDstOffset = bIdx * seqOut * dim;
            Buffer.MemoryCopy(aPtr + aSrcOffset, outPtr + aDstOffset, seqA * dim * sizeof(float), seqA * dim * sizeof(float));

            int bSrcOffset = bIdx * seqB * dim;
            int bDstOffset = bIdx * seqOut * dim + seqA * dim;
            Buffer.MemoryCopy(bPtr + bSrcOffset, outPtr + bDstOffset, seqB * dim * sizeof(float), seqB * dim * sizeof(float));
        }

        return output;
    }

    /// <summary>Splits a [B, S1+S2, D] tensor along the sequence dimension into [B, S1, D] and [B, S2, D].</summary>
    public static (Tensor first, Tensor second) SplitAlongSeqDim(Tensor combined, int firstSeqLen)
    {
        int batch = (int)combined.Shape[0];
        int totalSeq = (int)combined.Shape[1];
        int dim = (int)combined.Shape[2];
        int secondSeqLen = totalSeq - firstSeqLen;

        TensorShape firstShape = new TensorShape(batch, firstSeqLen, dim);
        TensorShape secondShape = new TensorShape(batch, secondSeqLen, dim);
        Tensor first = new Tensor(firstShape, DType.F32);
        Tensor second = new Tensor(secondShape, DType.F32);

        float* srcPtr = (float*)combined.DataPointer;
        float* firstPtr = (float*)first.DataPointer;
        float* secondPtr = (float*)second.DataPointer;

        for (int bIdx = 0; bIdx < batch; bIdx++)
        {
            int srcBase = bIdx * totalSeq * dim;
            Buffer.MemoryCopy(srcPtr + srcBase, firstPtr + bIdx * firstSeqLen * dim,
                firstSeqLen * dim * sizeof(float), firstSeqLen * dim * sizeof(float));
            Buffer.MemoryCopy(srcPtr + srcBase + firstSeqLen * dim, secondPtr + bIdx * secondSeqLen * dim,
                secondSeqLen * dim * sizeof(float), secondSeqLen * dim * sizeof(float));
        }

        return (first, second);
    }

    /// <summary>In-place variant: writes [B, numHeads, S, headDim] into a pre-allocated output tensor. Use when caller has already sized the destination (joint-attention concat path).</summary>
    public static void ReshapeToMultiHead(Tensor output, Tensor input, int batch, int seqLen, int numHeads, int headDim)
    {
        float* inPtr = (float*)input.DataPointer;
        float* outPtr = (float*)output.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                for (int h = 0; h < numHeads; h++)
                {
                    int srcOffset = (b * seqLen + s) * numHeads * headDim + h * headDim;
                    int dstOffset = ((b * numHeads + h) * seqLen + s) * headDim;
                    Buffer.MemoryCopy(inPtr + srcOffset, outPtr + dstOffset, headDim * sizeof(float), headDim * sizeof(float));
                }
            }
        }
    }

    /// <summary>In-place variant: writes [B, S, numHeads * headDim] into a pre-allocated output tensor. Inverse of <see cref="ReshapeToMultiHead(Tensor, Tensor, int, int, int, int)"/>.</summary>
    public static void ReshapeFromMultiHead(Tensor output, Tensor input, int batch, int seqLen, int numHeads, int headDim)
    {
        int hiddenSize = numHeads * headDim;
        float* inPtr = (float*)input.DataPointer;
        float* outPtr = (float*)output.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                for (int h = 0; h < numHeads; h++)
                {
                    int srcOffset = ((b * numHeads + h) * seqLen + s) * headDim;
                    int dstOffset = (b * seqLen + s) * hiddenSize + h * headDim;
                    Buffer.MemoryCopy(inPtr + srcOffset, outPtr + dstOffset, headDim * sizeof(float), headDim * sizeof(float));
                }
            }
        }
    }

    /// <summary>Concatenates two [B, H, S, D] tensors along the sequence dimension in head-major (multi-head) layout. Joint-attention path: stacks ctx then img per head.</summary>
    public static void ConcatAlongSeqDimMultiHead(Tensor output, Tensor first, Tensor second,
        int batch, int numHeads, int firstSeqLen, int secondSeqLen, int headDim)
    {
        float* firstPtr = (float*)first.DataPointer;
        float* secondPtr = (float*)second.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        int totalSeqLen = firstSeqLen + secondSeqLen;

        for (int b = 0; b < batch; b++)
        {
            for (int h = 0; h < numHeads; h++)
            {
                int outBase = (b * numHeads + h) * totalSeqLen * headDim;
                int firstBase = (b * numHeads + h) * firstSeqLen * headDim;
                int secondBase = (b * numHeads + h) * secondSeqLen * headDim;

                long firstBytes = (long)firstSeqLen * headDim * sizeof(float);
                Buffer.MemoryCopy(firstPtr + firstBase, outPtr + outBase, firstBytes, firstBytes);

                long secondBytes = (long)secondSeqLen * headDim * sizeof(float);
                Buffer.MemoryCopy(secondPtr + secondBase, outPtr + outBase + firstSeqLen * headDim, secondBytes, secondBytes);
            }
        }
    }

    /// <summary>Splits a [B, H, S1+S2, D] tensor along the sequence dimension in head-major layout. Inverse of <see cref="ConcatAlongSeqDimMultiHead"/>.</summary>
    public static void SplitAlongSeqDimMultiHead(Tensor first, Tensor second, Tensor input,
        int batch, int numHeads, int firstSeqLen, int secondSeqLen, int headDim)
    {
        float* inPtr = (float*)input.DataPointer;
        float* firstPtr = (float*)first.DataPointer;
        float* secondPtr = (float*)second.DataPointer;
        int totalSeqLen = firstSeqLen + secondSeqLen;

        for (int b = 0; b < batch; b++)
        {
            for (int h = 0; h < numHeads; h++)
            {
                int inBase = (b * numHeads + h) * totalSeqLen * headDim;
                int firstBase = (b * numHeads + h) * firstSeqLen * headDim;
                int secondBase = (b * numHeads + h) * secondSeqLen * headDim;

                long firstBytes = (long)firstSeqLen * headDim * sizeof(float);
                Buffer.MemoryCopy(inPtr + inBase, firstPtr + firstBase, firstBytes, firstBytes);

                long secondBytes = (long)secondSeqLen * headDim * sizeof(float);
                Buffer.MemoryCopy(inPtr + inBase + firstSeqLen * headDim, secondPtr + secondBase, secondBytes, secondBytes);
            }
        }
    }

    /// <summary>Reshapes [B, S, D] → [B, numHeads, S, headDim] returning a freshly allocated tensor.</summary>
    public static Tensor ReshapeToMultiHead(Tensor input, int batch, int seqLen, int numHeads, int headDim)
    {
        TensorShape outShape = new TensorShape(batch, numHeads, seqLen, headDim);
        Tensor output = new Tensor(outShape, DType.F32);

        float* inPtr = (float*)input.DataPointer;
        float* outPtr = (float*)output.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                for (int h = 0; h < numHeads; h++)
                {
                    int srcOffset = (b * seqLen + s) * numHeads * headDim + h * headDim;
                    int dstOffset = ((b * numHeads + h) * seqLen + s) * headDim;
                    Buffer.MemoryCopy(inPtr + srcOffset, outPtr + dstOffset, headDim * sizeof(float), headDim * sizeof(float));
                }
            }
        }

        return output;
    }

    /// <summary>Reshapes [B, numHeads, S, headDim] → [B, S, numHeads * headDim] from multi-head attention back to flat hidden dim.</summary>
    public static Tensor ReshapeFromMultiHead(Tensor input, int batch, int seqLen, int numHeads, int headDim)
    {
        int hiddenSize = numHeads * headDim;
        TensorShape outShape = new TensorShape(batch, seqLen, hiddenSize);
        Tensor output = new Tensor(outShape, DType.F32);

        float* inPtr = (float*)input.DataPointer;
        float* outPtr = (float*)output.DataPointer;

        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                for (int h = 0; h < numHeads; h++)
                {
                    int srcOffset = ((b * numHeads + h) * seqLen + s) * headDim;
                    int dstOffset = (b * seqLen + s) * hiddenSize + h * headDim;
                    Buffer.MemoryCopy(inPtr + srcOffset, outPtr + dstOffset, headDim * sizeof(float), headDim * sizeof(float));
                }
            }
        }

        return output;
    }

    /// <summary>Concatenates two [B, S, D1] and [B, S, D2] tensors along the last dimension → [B, S, D1+D2].</summary>
    public static Tensor ConcatAlongLastDim(Tensor a, Tensor b)
    {
        int batch = (int)a.Shape[0];
        int seqLen = (int)a.Shape[1];
        int dimA = (int)a.Shape[2];
        int dimB = (int)b.Shape[2];
        int dimOut = dimA + dimB;

        TensorShape outShape = new TensorShape(batch, seqLen, dimOut);
        Tensor output = new Tensor(outShape, DType.F32);

        float* aPtr = (float*)a.DataPointer;
        float* bPtr = (float*)b.DataPointer;
        float* outPtr = (float*)output.DataPointer;

        for (int bIdx = 0; bIdx < batch; bIdx++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                int aOffset = (bIdx * seqLen + s) * dimA;
                int bOffset = (bIdx * seqLen + s) * dimB;
                int outOffset = (bIdx * seqLen + s) * dimOut;

                Buffer.MemoryCopy(aPtr + aOffset, outPtr + outOffset, dimA * sizeof(float), dimA * sizeof(float));
                Buffer.MemoryCopy(bPtr + bOffset, outPtr + outOffset + dimA, dimB * sizeof(float), dimB * sizeof(float));
            }
        }

        return output;
    }

    /// <summary>Concatenates two pooled tensors [B, D1] and [B, D2] along the last dimension → [B, D1+D2].</summary>
    public static Tensor ConcatPooled(Tensor a, Tensor b)
    {
        int batch = (int)a.Shape[0];
        int dimA = (int)a.Shape[1];
        int dimB = (int)b.Shape[1];
        int dimOut = dimA + dimB;

        TensorShape outShape = new TensorShape(batch, dimOut);
        Tensor output = new Tensor(outShape, DType.F32);

        float* aPtr = (float*)a.DataPointer;
        float* bPtr = (float*)b.DataPointer;
        float* outPtr = (float*)output.DataPointer;

        for (int bIdx = 0; bIdx < batch; bIdx++)
        {
            Buffer.MemoryCopy(aPtr + bIdx * dimA, outPtr + bIdx * dimOut, dimA * sizeof(float), dimA * sizeof(float));
            Buffer.MemoryCopy(bPtr + bIdx * dimB, outPtr + bIdx * dimOut + dimA, dimB * sizeof(float), dimB * sizeof(float));
        }

        return output;
    }

    /// <summary>Pads the last dimension with zeros: [B, S, currentDim] → [B, S, targetDim].</summary>
    public static Tensor PadLastDim(Tensor input, int currentDim, int targetDim)
    {
        if (currentDim == targetDim)
            return input;

        int batch = (int)input.Shape[0];
        int seqLen = (int)input.Shape[1];

        TensorShape outShape = new TensorShape(batch, seqLen, targetDim);
        Tensor output = new Tensor(outShape, DType.F32);

        float* inPtr = (float*)input.DataPointer;
        float* outPtr = (float*)output.DataPointer;

        int totalElements = batch * seqLen * targetDim;
        for (int i = 0; i < totalElements; i++)
            outPtr[i] = 0.0f;

        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                int inOffset = (b * seqLen + s) * currentDim;
                int outOffset = (b * seqLen + s) * targetDim;
                Buffer.MemoryCopy(inPtr + inOffset, outPtr + outOffset, currentDim * sizeof(float), currentDim * sizeof(float));
            }
        }

        return output;
    }
}
