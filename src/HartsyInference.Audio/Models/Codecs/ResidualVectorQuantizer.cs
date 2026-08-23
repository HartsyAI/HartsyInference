using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Codecs;

/// <summary>Residual Vector Quantization — the codebook stack at the heart of EnCodec, SoundStream, and most of their derivatives, greedily encoding against N learned KxD codebooks in sequence against the shrinking residual.</summary>
/// <remarks>
/// <para>The per-frame work is <c>O(N * K * D)</c> nearest-neighbor lookups. For the
/// EnCodec 24 kHz config (<c>K=1024, D=128, N=8</c>) that's ~1M flops per frame, ~75M
/// flops per second of audio — negligible compared to the SEANet encoder cost.</para>
///
/// <para>State-dict layout follows the <c>vector_quantize_pytorch::EuclideanCodebook</c>
/// convention used by EnCodec: <c>{prefix}.vq.layers.{q}._codebook.embed</c> for each
/// codebook <c>q</c>. The <c>cluster_size</c> / <c>embed_avg</c> buffers exist in
/// training checkpoints but aren't needed at inference.</para>
///
/// <para>Used by EnCodec (this file's primary consumer), and re-used by DAC (which
/// switches to L2-normalized cosine codebooks but keeps the same residual structure),
/// XCodec, and several other Section 4 codecs. The L2-norm variant is enabled via the
/// <see cref="L2Normalize"/> flag at construction.</para>
/// </remarks>
internal sealed unsafe class ResidualVectorQuantizer
{
    public int MaxCodebooks { get; }
    public int CodebookSize { get; }
    public int Dim { get; }
    public bool L2Normalize { get; }

    private readonly Tensor?[] _codebooks;

    public ResidualVectorQuantizer(int maxCodebooks, int codebookSize, int dim, bool l2Normalize = false)
    {
        if (maxCodebooks <= 0) throw new ArgumentOutOfRangeException(nameof(maxCodebooks));
        if (codebookSize <= 0) throw new ArgumentOutOfRangeException(nameof(codebookSize));
        if (dim <= 0) throw new ArgumentOutOfRangeException(nameof(dim));
        MaxCodebooks = maxCodebooks;
        CodebookSize = codebookSize;
        Dim = dim;
        L2Normalize = l2Normalize;
        _codebooks = new Tensor?[maxCodebooks];
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        for (int q = 0; q < MaxCodebooks; q++)
        {
            _codebooks[q] = WhisperOps.EnsureF32(w[$"{prefix}.vq.layers.{q}._codebook.embed"]);
            if ((int)_codebooks[q]!.Shape[0] != CodebookSize || (int)_codebooks[q]!.Shape[1] != Dim)
                throw new InvalidDataException($"Codebook {q} shape mismatch: expected [{CodebookSize}, {Dim}], got {_codebooks[q]!.Shape}.");
        }
    }

    /// <summary>Greedy residual encoding of a continuous latent. <paramref name="latent"/> is channels-first <c>[batch, dim, T]</c>. Returns codes of shape <c>[nQ, batch, T]</c> as Int32 — the first axis indexes the codebook, matching the layout upstream EnCodec uses for its LM-side token sequences. <paramref name="nQ"/> selects how many of the available codebooks to use (Bark = 8 ⇒ 6 kbps; MusicGen-small = 4 ⇒ 3 kbps) and must be ≤ <see cref="MaxCodebooks"/>.</summary>
    public Tensor Encode(Tensor latent, int batch, int t, int nQ)
    {
        if (_codebooks[0] is null) throw new InvalidOperationException("ResidualVectorQuantizer weights not loaded.");
        if (nQ <= 0 || nQ > MaxCodebooks) throw new ArgumentOutOfRangeException(nameof(nQ), nQ, $"nQ must be in [1, {MaxCodebooks}].");

        Tensor codes = new(new TensorShape(nQ, batch, t), DType.I32);
        float* lp = (float*)latent.DataPointer;
        int* cp = (int*)codes.DataPointer;

        // Working residual buffer: [Dim] per (batch, t). Reused across timesteps.
        Span<float> residual = stackalloc float[Dim];

        for (int b = 0; b < batch; b++)
        {
            for (int ti = 0; ti < t; ti++)
            {
                for (int d = 0; d < Dim; d++)
                    residual[d] = lp[(b * Dim + d) * t + ti];

                if (L2Normalize) NormalizeInPlace(residual);

                for (int q = 0; q < nQ; q++)
                {
                    Tensor cb = _codebooks[q]!;
                    float* cbp = (float*)cb.DataPointer;
                    int idx = NearestCodewordIndex(cbp, residual);
                    cp[(q * batch + b) * t + ti] = idx;

                    for (int d = 0; d < Dim; d++)
                        residual[d] -= cbp[idx * Dim + d];
                }
            }
        }
        return codes;
    }

    /// <summary>Decoding: sum the chosen codewords across all codebooks per frame. <paramref name="codes"/> has shape <c>[nQ, batch, T]</c> Int32. Returns the reconstructed channels-first latent <c>[batch, dim, T]</c>.</summary>
    public Tensor Decode(Tensor codes, int batch, int t)
    {
        if (_codebooks[0] is null) throw new InvalidOperationException("ResidualVectorQuantizer weights not loaded.");
        if (codes.Shape.Rank != 3) throw new ArgumentException($"codes must be rank-3 [nQ, B, T], got {codes.Shape}.");
        int nQ = (int)codes.Shape[0];
        if (nQ > MaxCodebooks) throw new ArgumentException($"codes nQ ({nQ}) exceeds MaxCodebooks ({MaxCodebooks}).");
        if (codes.DType != DType.I32) throw new ArgumentException($"codes must be Int32, got {codes.DType}.");

        Tensor latent = new(new TensorShape(batch, Dim, t), DType.F32);
        float* lp = (float*)latent.DataPointer;
        int* cp = (int*)codes.DataPointer;

        long total = (long)batch * Dim * t;
        for (long i = 0; i < total; i++) lp[i] = 0f;

        for (int q = 0; q < nQ; q++)
        {
            Tensor cb = _codebooks[q]!;
            float* cbp = (float*)cb.DataPointer;
            for (int b = 0; b < batch; b++)
            {
                for (int ti = 0; ti < t; ti++)
                {
                    int idx = cp[(q * batch + b) * t + ti];
                    for (int d = 0; d < Dim; d++)
                        lp[(b * Dim + d) * t + ti] += cbp[idx * Dim + d];
                }
            }
        }
        return latent;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        for (int q = 0; q < MaxCodebooks; q++)
            if (_codebooks[q] is not null) yield return _codebooks[q]!;
    }

    /// <summary>Returns the index of the codeword with the smallest L2 distance to <paramref name="query"/>. Codebook is stored row-major as <c>[K, D]</c>.</summary>
    private int NearestCodewordIndex(float* codebookPtr, ReadOnlySpan<float> query)
    {
        int bestIdx = 0;
        float bestDist = float.MaxValue;
        for (int k = 0; k < CodebookSize; k++)
        {
            float dist = 0f;
            int rowBase = k * Dim;
            for (int d = 0; d < Dim; d++)
            {
                float diff = query[d] - codebookPtr[rowBase + d];
                dist += diff * diff;
                if (dist >= bestDist) break;     // early exit when partial distance already exceeds best
            }
            if (dist < bestDist)
            {
                bestDist = dist;
                bestIdx = k;
            }
        }
        return bestIdx;
    }

    private static void NormalizeInPlace(Span<float> v)
    {
        double sumSq = 0d;
        for (int i = 0; i < v.Length; i++) sumSq += (double)v[i] * v[i];
        float invNorm = (float)(1.0 / Math.Sqrt(sumSq + 1e-12));
        for (int i = 0; i < v.Length; i++) v[i] *= invNorm;
    }
}
