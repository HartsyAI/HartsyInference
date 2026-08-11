using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.CosyVoice;

/// <summary>Builds additive attention masks for CosyVoice2's chunk-aware causal flow decoder — the same
/// convention <c>LlamaStyleEncoder.BuildCausalMask</c> uses (<c>[T, T]</c>, 0 = visible, large-negative =
/// masked, broadcasts across batch/heads inside <see cref="Core.Backends.IBackend.ScaledDotProductAttention"/>).</summary>
internal static unsafe class MaskBuilder
{
    private const float NegInf = -1e30f;

    /// <summary>Block-diagonal-causal ("chunk-M"): a query in chunk <c>k</c> (<c>= i / chunkSize</c>) attends
    /// to every key in chunks <c>&lt;= k</c> — unbounded left context, chunk-granular right boundary. This is
    /// one of the masks CosyVoice2's flow decoder was actually trained on (randomly sampled per batch
    /// alongside non-causal/full-causal/chunk-2M), not an ad-hoc mask bolted onto an unmasked checkpoint.</summary>
    public static Tensor BuildChunkCausalMask(int totalLen, int chunkSize)
    {
        if (chunkSize <= 0) throw new ArgumentException("chunkSize must be positive.", nameof(chunkSize));
        Tensor mask = new(new TensorShape(totalLen, totalLen), DType.F32);
        float* p = (float*)mask.DataPointer;
        for (int i = 0; i < totalLen; i++)
        {
            int qChunk = i / chunkSize;
            long row = (long)i * totalLen;
            for (int j = 0; j < totalLen; j++)
            {
                int kChunk = j / chunkSize;
                p[row + j] = kChunk > qChunk ? NegInf : 0f;
            }
        }
        return mask;
    }
}
