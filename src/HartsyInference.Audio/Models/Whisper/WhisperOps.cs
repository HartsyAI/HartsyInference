using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Whisper;

/// <summary>Shared low-level helpers used by both the Whisper encoder and decoder, mirroring the pattern in <c>ClipTextEncoder.cs</c> for a "linear + bias + multi-head reshape" path the <see cref="IBackend"/> doesn't expose directly.</summary>
/// <remarks>All helpers operate on F32 tensors via unsafe <c>float*</c> indexing. Weight tensors are pre-cast to F32 at load time (via <see cref="EnsureF32"/>) so that the pointer math never trips over BF16 / FP16 / FP8 layouts that the safetensors loader may hand us; inference dtype follows the input tensors. Keep these methods small and obvious — the Whisper encoder/decoder modules already carry meaningful per-step state, and broadening these helpers into stateful modules would just push complexity around.</remarks>
internal static unsafe class WhisperOps
{
    /// <summary>Returns the tensor as F32, casting if needed; on the pass-through (already-F32) path, the caller does not own the returned result and should treat the input as the disposal owner.</summary>
    public static Tensor EnsureF32(Tensor t) =>
        t.DType != DType.F32 ? t.CastTo(DType.F32) : t;

    /// <summary>Linear projection dispatched to <see cref="IBackend.Linear"/>, which takes the HF/PyTorch <c>[outDim, inDim]</c> weight as-is.</summary>
    /// <remarks>The unmodified weight tensor is what reaches the backend, so its device cache (weight auto-promotion) keeps it GPU-resident across calls — the previous per-call CPU transpose into a fresh scratch tensor could never be cached and re-crossed PCIe on every linear.</remarks>
    public static Tensor ProjectLinear(IBackend backend, Tensor input, Tensor weight, Tensor? bias, int batch, int seqLen, int inDim, int outDim)
    {
        Tensor output = new(new TensorShape(batch, seqLen, outDim), DType.F32);
        backend.Linear(output, input, weight, bias);
        return output;
    }

    /// <summary>Reshapes [B, S, H*D] → [B, H, S, D] via element copy — the 4-D layout <see cref="IBackend.ScaledDotProductAttention"/> expects.</summary>
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

    /// <summary>Builds a causal attention mask [seqLen, seqLen] for decoder self-attention: 0 on/below diagonal, -inf above.</summary>
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
}
