using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

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

    /// <summary>Frozen 3D sin-cos position embedding (DiT-style), generic across video DiTs. Returns <c>[frames*height*width, dim]</c> in <c>(t, h, w)</c> row-major order (<c>idx = t*(H*W) + h*W + w</c>). Channels are split <c>dim ≈ [dim/3, dim/3, rest]</c> across the three axes; each axis uses 1D sin-cos (<c>[sin, cos]</c> halves, base 10000). Matches Lance/Wan/`get_3d_sincos_pos_embed`. Reusable by any model needing a frozen 3D positional grid.</summary>
    public static Tensor Sincos3DPositionEmbedding(int frames, int height, int width, int dim)
    {
        if (dim % 2 != 0)
            throw new ArgumentException($"dim {dim} must be even.", nameof(dim));
        int d = dim / 3;
        if (d % 2 != 0) d -= 1;
        int dimT = d, dimH = d, dimW = dim - 2 * d;

        int count = frames * height * width;
        Tensor output = new Tensor(new TensorShape(count, dim), DType.F32);
        float* outPtr = (float*)output.DataPointer;

        for (int ti = 0; ti < frames; ti++)
        {
            for (int hi = 0; hi < height; hi++)
            {
                for (int wi = 0; wi < width; wi++)
                {
                    long row = ((long)ti * height + hi) * width + wi;
                    long baseOff = row * dim;
                    Write1DSincos(outPtr + baseOff, ti, dimT);
                    Write1DSincos(outPtr + baseOff + dimT, hi, dimH);
                    Write1DSincos(outPtr + baseOff + dimT + dimH, wi, dimW);
                }
            }
        }
        return output;
    }

    /// <summary>Writes a 1D sin-cos embedding for scalar <paramref name="pos"/> into <paramref name="dst"/> (length <paramref name="axisDim"/>): first half sin, second half cos, <c>omega_k = 1/10000^(k/(axisDim/2))</c>.</summary>
    private static void Write1DSincos(float* dst, int pos, int axisDim)
    {
        int half = axisDim / 2;
        for (int k = 0; k < half; k++)
        {
            double omega = 1.0 / Math.Pow(10000.0, (double)k / half);
            double angle = pos * omega;
            dst[k] = (float)Math.Sin(angle);
            dst[half + k] = (float)Math.Cos(angle);
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

    /// <summary>Patchifies <c>[B, C, H, W]</c> → <c>[B, (H/p)·(W/p), p²·C]</c> with channel-inner ordering inside each
    /// patch (<c>(py, px, c)</c>), matching the upstream einops <c>'c (h p1) (w p2) -&gt; (h w) (p1 p2 c)'</c> used by
    /// Lumina/OmniGen2/Boogu patch embedders. Reusable across patch-2 latent DiTs.</summary>
    public static Tensor PatchifyNCHW(Tensor latent, int patch)
    {
        int batch = (int)latent.Shape[0];
        int channels = (int)latent.Shape[1];
        int height = (int)latent.Shape[2];
        int width = (int)latent.Shape[3];
        int hPacked = height / patch;
        int wPacked = width / patch;
        int imgSeqLen = hPacked * wPacked;
        int patchVolume = patch * patch * channels;

        Tensor result = new Tensor(new TensorShape(batch, imgSeqLen, patchVolume), DType.F32);
        float* src = (float*)latent.DataPointer;
        float* dst = (float*)result.DataPointer;
        long chwStride = (long)channels * height * width;
        long hwStride = (long)height * width;

        for (int b = 0; b < batch; b++)
        {
            float* batchSrc = src + b * chwStride;
            float* batchDst = dst + (long)b * imgSeqLen * patchVolume;
            for (int hp = 0; hp < hPacked; hp++)
                for (int wp = 0; wp < wPacked; wp++)
                {
                    float* tokenDst = batchDst + ((long)hp * wPacked + wp) * patchVolume;
                    int outIdx = 0;
                    for (int py = 0; py < patch; py++)
                    {
                        int srcRow = hp * patch + py;
                        for (int px = 0; px < patch; px++)
                        {
                            int srcCol = wp * patch + px;
                            for (int c = 0; c < channels; c++)
                                tokenDst[outIdx++] = batchSrc[c * hwStride + srcRow * width + srcCol];
                        }
                    }
                }
        }
        return result;
    }

    /// <summary>Inverse of <see cref="PatchifyNCHW"/>: <c>[B, (H/p)·(W/p), p²·C]</c> → <c>[B, C, H, W]</c>.</summary>
    public static Tensor UnpatchifyToNCHW(Tensor tokens, int channels, int hPacked, int wPacked, int patch)
    {
        int batch = (int)tokens.Shape[0];
        int height = hPacked * patch;
        int width = wPacked * patch;
        int imgSeqLen = hPacked * wPacked;
        int patchVolume = patch * patch * channels;

        Tensor result = new Tensor(new TensorShape(batch, channels, height, width), DType.F32);
        float* src = (float*)tokens.DataPointer;
        float* dst = (float*)result.DataPointer;
        long chwStride = (long)channels * height * width;
        long hwStride = (long)height * width;

        for (int b = 0; b < batch; b++)
        {
            float* batchSrc = src + (long)b * imgSeqLen * patchVolume;
            float* batchDst = dst + b * chwStride;
            for (int hp = 0; hp < hPacked; hp++)
                for (int wp = 0; wp < wPacked; wp++)
                {
                    float* tokenSrc = batchSrc + ((long)hp * wPacked + wp) * patchVolume;
                    int srcIdx = 0;
                    for (int py = 0; py < patch; py++)
                    {
                        int dstRow = hp * patch + py;
                        for (int px = 0; px < patch; px++)
                        {
                            int dstCol = wp * patch + px;
                            for (int c = 0; c < channels; c++)
                                batchDst[c * hwStride + dstRow * width + dstCol] = tokenSrc[srcIdx++];
                        }
                    }
                }
        }
        return result;
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

    /// <summary>LayerNorm (no affine, eps 1e-6) followed by AdaLN modulation <c>out = x*(1+scale)+shift</c>, entirely on
    /// the backend so the activation stays device-resident. <c>AffineBroadcastLastDim</c> computes <c>x*scale+shift</c>,
    /// so <paramref name="scale"/> is pre-incremented by 1 (<c>AddScalar</c>) to reproduce the <c>(1+scale)</c> factor —
    /// bit-identical to the old host <see cref="AdaLNModulation.ApplyModulation"/>. Mirrors QwenImageBlock.NormModulate.</summary>
    public static Tensor NormModulate(IBackend backend, Tensor x, Tensor shift, Tensor scale, TensorShape shape, float eps = 1e-6f)
    {
        Tensor normed = new Tensor(shape, DType.F32);
        backend.LayerNormNoAffine(normed, x, eps);
        Tensor output = Modulate(backend, normed, shift, scale, shape);
        normed.Dispose();
        return output;
    }

    /// <summary>AdaLN modulation <c>out = x*(1+scale)+shift</c> on an ALREADY-normalized input, entirely on the
    /// backend. Split out of <see cref="NormModulate"/> for blocks that share one LayerNorm across two modulation
    /// paths (SD3.5 dual-attention) or use an affine <c>backend.LayerNorm</c> before modulating. Bit-identical to
    /// the old host <see cref="AdaLNModulation.ApplyModulation"/>. Pass <paramref name="shift"/> null for scale-only
    /// modulation <c>out = x*(1+scale)</c> (Z-Image / Ideogram-4 adaLN blocks that omit the shift term).</summary>
    public static Tensor Modulate(IBackend backend, Tensor x, Tensor? shift, Tensor scale, TensorShape shape)
    {
        Tensor scalePlus1 = new Tensor(scale.Shape, DType.F32);
        backend.AddScalar(scalePlus1, scale, 1.0f);
        Tensor output = new Tensor(shape, DType.F32);
        backend.AffineBroadcastLastDim(output, x, scalePlus1, shift);
        scalePlus1.Dispose();
        return output;
    }
}
