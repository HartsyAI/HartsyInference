using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Audio.Models.Whisper;

/// <summary>Shared low-level helpers used by both the Whisper encoder and decoder.
/// These mirror the pattern in <c>ClipTextEncoder.cs</c> — small, pointer-walking
/// primitives that the diffusion package found necessary because the existing
/// <see cref="IBackend"/> doesn't expose a "linear + bias + multi-head reshape" path.
///
/// <para>All helpers operate on F32 tensors via unsafe <c>float*</c> indexing. Weight
/// tensors are pre-cast to F32 at load time (via <see cref="EnsureF32"/>) so that the
/// pointer math never trips over BF16 / FP16 / FP8 layouts that the safetensors loader
/// may hand us. Inference dtype follows the input tensors.</para>
///
/// <para>Keep these methods small and obvious. The Whisper encoder/decoder modules
/// already carry meaningful per-step state; broadening these helpers into stateful
/// modules would just push complexity around.</para></summary>
internal static unsafe class WhisperOps
{
    /// <summary>Returns the tensor as F32, casting if needed. Pass-through when the
    /// tensor is already F32 — the caller does not own the returned cast result and
    /// should treat the input as the disposal owner.</summary>
    public static Tensor EnsureF32(Tensor t) =>
        t.DType != DType.F32 ? t.CastTo(DType.F32) : t;

    /// <summary>Linear projection: output = input @ weight^T + bias. The HF safetensors
    /// store linear-layer weights as <c>[outDim, inDim]</c> (PyTorch convention) so we
    /// transpose into a scratch tensor before dispatching to <see cref="IBackend.BatchedMatMul"/>.
    /// Same approach as <c>ClipTransformerLayer.ProjectLinear</c>; consolidated here to
    /// keep encoder + decoder code aligned.</summary>
    public static Tensor ProjectLinear(IBackend backend, Tensor input, Tensor weight, Tensor? bias, int batch, int seqLen, int inDim, int outDim)
    {
        TensorShape outShape = new(batch, seqLen, outDim);
        Tensor output = new(outShape, DType.F32);

        TensorShape weightTShape = new(inDim, outDim);
        Tensor weightT = new(weightTShape, DType.F32);
        TransposeMatrix(weight, weightT, outDim, inDim);

        backend.BatchedMatMul(output, input, weightT);
        weightT.Dispose();

        if (bias is not null) AddBiasBroadcast(output, bias, batch, seqLen, outDim);
        return output;
    }

    /// <summary>Transposes a 2-D float matrix [rows, cols] → [cols, rows].</summary>
    public static void TransposeMatrix(Tensor src, Tensor dst, int rows, int cols)
    {
        float* srcPtr = (float*)src.DataPointer;
        float* dstPtr = (float*)dst.DataPointer;
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                dstPtr[c * rows + r] = srcPtr[r * cols + c];
    }

    /// <summary>Adds bias [outDim] to output [batch, seqLen, outDim] in place.</summary>
    public static void AddBiasBroadcast(Tensor output, Tensor bias, int batch, int seqLen, int outDim)
    {
        float* outPtr = (float*)output.DataPointer;
        float* biasPtr = (float*)bias.DataPointer;
        for (int b = 0; b < batch; b++)
            for (int s = 0; s < seqLen; s++)
            {
                int offset = (b * seqLen + s) * outDim;
                for (int d = 0; d < outDim; d++) outPtr[offset + d] += biasPtr[d];
            }
    }

    /// <summary>Reshapes [B, S, H*D] → [B, H, S, D] via element copy. The 4-D layout
    /// is what <see cref="IBackend.ScaledDotProductAttention"/> expects.</summary>
    public static void ReshapeToMultiHead4D(Tensor output, Tensor input, int batch, int seqLen, int numHeads, int headDim)
    {
        float* inPtr = (float*)input.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        for (int b = 0; b < batch; b++)
            for (int s = 0; s < seqLen; s++)
                for (int h = 0; h < numHeads; h++)
                {
                    int inOffset = (b * seqLen + s) * (numHeads * headDim) + h * headDim;
                    int outOffset = ((b * numHeads + h) * seqLen + s) * headDim;
                    for (int d = 0; d < headDim; d++) outPtr[outOffset + d] = inPtr[inOffset + d];
                }
    }

    /// <summary>Reshapes [B, H, S, D] → [B, S, H*D] via element copy.</summary>
    public static void ReshapeFromMultiHead4D(Tensor output, Tensor input, int batch, int seqLen, int numHeads, int headDim)
    {
        float* inPtr = (float*)input.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        for (int b = 0; b < batch; b++)
            for (int s = 0; s < seqLen; s++)
                for (int h = 0; h < numHeads; h++)
                {
                    int inOffset = ((b * numHeads + h) * seqLen + s) * headDim;
                    int outOffset = (b * seqLen + s) * (numHeads * headDim) + h * headDim;
                    for (int d = 0; d < headDim; d++) outPtr[outOffset + d] = inPtr[inOffset + d];
                }
    }

    /// <summary>Builds a causal attention mask [seqLen, seqLen]: 0 on/below diagonal, -inf above.
    /// Used by the decoder self-attention.</summary>
    public static Tensor BuildCausalMask(int seqLen)
    {
        TensorShape shape = new(seqLen, seqLen);
        Tensor mask = new(shape, DType.F32);
        float* p = (float*)mask.DataPointer;
        for (int i = 0; i < seqLen; i++)
            for (int j = 0; j < seqLen; j++)
                p[i * seqLen + j] = j <= i ? 0f : float.NegativeInfinity;
        return mask;
    }

    /// <summary>Expands a 2-D mask [S, S] to 4-D [1, 1, S, S] by memcpy. SDPA expects
    /// the mask to broadcast across batch and head dims.</summary>
    public static Tensor Expand2DMaskTo4D(Tensor mask2d, int seqLen)
    {
        TensorShape shape = new(1, 1, seqLen, seqLen);
        Tensor mask4d = new(shape, DType.F32);
        long bytes = seqLen * (long)seqLen * sizeof(float);
        Buffer.MemoryCopy((void*)mask2d.DataPointer, (void*)mask4d.DataPointer, bytes, bytes);
        return mask4d;
    }
}
