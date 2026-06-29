using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Codecs.XCodec;

/// <summary>The real xcodec_mini_infer (SoundStream) residual vector quantizer — an EMA
/// <c>EuclideanCodebook</c> RVQ, NOT the factorized DAC RVQ. Each of the 12 codebooks is a single
/// <c>[codebook_size, dimension]</c> embedding table (<c>quantizer.vq.layers.{i}._codebook.embed</c>,
/// shape <c>[1024, 1024]</c>); there is no per-codebook in/out projection (the upstream
/// <c>VectorQuantization</c> has <c>codebook_dim == dimension</c> so <c>project_in/out</c> are
/// <c>nn.Identity</c>).
///
/// <para><b>Decode</b> (the only path YuE needs) mirrors
/// <c>ResidualVectorQuantization.decode</c> / <c>EuclideanCodebook.decode</c>:
/// <code>
///   quantized_out = 0
///   for i, indices in enumerate(q_indices):   # q_indices: [n_q, B, T]
///       q = F.embedding(indices, embed_i)      # [B, T, D]
///       q = rearrange(q, "b n d -> b d n")     # [B, D, T]
///       quantized_out += q
/// </code>
/// The lookup is a pure table gather — the EMA codebook is already the dequantized vector
/// (no cosine normalization, unlike DAC). YuE Stage-1 produces only codebook-0 (vocal cb0) tokens,
/// so the decode is invoked with <c>n_q = 1</c> and only <c>embed[0]</c> is summed. Decoding extra
/// zero-filled codebooks would be WRONG: index 0 is a valid (non-zero) codeword, not silence.</para>
///
/// <para><b>Encode</b> mirrors the residual nearest-neighbor loop (Euclidean argmin = max of
/// <c>2·x·e − ||e||²</c>) and is provided for round-trip parity; the YuE decode path does not use
/// it.</para></summary>
internal sealed unsafe class XCodecEmaResidualVectorQuantizer
{
    public int NCodebooks { get; }
    public int CodebookSize { get; }
    public int Dimension { get; }

    private readonly Tensor?[] _embed;          // [codebook_size, dimension] per layer
    private readonly float[]?[] _embedSqNorm;   // ||embed[k]||^2 per row, cached for encode

    public XCodecEmaResidualVectorQuantizer(int nCodebooks, int codebookSize, int dimension)
    {
        NCodebooks = nCodebooks;
        CodebookSize = codebookSize;
        Dimension = dimension;
        _embed = new Tensor?[nCodebooks];
        _embedSqNorm = new float[]?[nCodebooks];
    }

    /// <summary>Loads the EMA codebook tables from <c>{prefix}.vq.layers.{i}._codebook.embed</c>.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        for (int q = 0; q < NCodebooks; q++)
        {
            _embed[q] = WhisperOps.EnsureF32(w[$"{prefix}.vq.layers.{q}._codebook.embed"]);
            if (_embed[q]!.Shape[0] != CodebookSize || _embed[q]!.Shape[1] != Dimension)
                throw new InvalidOperationException(
                    $"XCodec codebook {q} embed shape {_embed[q]!.Shape} != [{CodebookSize}, {Dimension}].");
        }
    }

    /// <summary>Decodes integer codes back to the continuous latent. Codes <c>[nQ, batch, T]</c>
    /// Int32; output channels-first <c>[batch, dimension, T]</c>. Sums the per-codebook embedding
    /// lookups across the first <c>nQ = codes.Shape[0]</c> codebooks.</summary>
    public Tensor Decode(IBackend backend, Tensor codes, int batch, int t)
    {
        _ = backend;
        if (_embed[0] is null) throw new InvalidOperationException("XCodecEmaResidualVectorQuantizer weights not loaded.");
        int nQ = (int)codes.Shape[0];
        if (nQ > NCodebooks) throw new ArgumentException($"codes nQ ({nQ}) exceeds NCodebooks ({NCodebooks}).");

        Tensor latent = new(new TensorShape(batch, Dimension, t), DType.F32);
        float* lp = (float*)latent.DataPointer;
        long total = latent.ElementCount;
        for (long i = 0; i < total; i++) lp[i] = 0f;

        int* cp = (int*)codes.DataPointer;
        for (int q = 0; q < nQ; q++)
        {
            float* emb = (float*)_embed[q]!.DataPointer;
            for (int b = 0; b < batch; b++)
            {
                for (int ti = 0; ti < t; ti++)
                {
                    int idx = cp[(q * batch + b) * t + ti];
                    if ((uint)idx >= (uint)CodebookSize) idx = 0;
                    int rowBase = idx * Dimension;
                    // latent[b, :, ti] += embed[idx]  (channels-first stride)
                    for (int d = 0; d < Dimension; d++)
                        lp[(b * Dimension + d) * t + ti] += emb[rowBase + d];
                }
            }
        }
        return latent;
    }

    /// <summary>Encodes a continuous latent <c>[batch, dimension, T]</c> into codes <c>[nQ, batch, T]</c>
    /// via the residual Euclidean nearest-neighbor loop. Provided for round-trip parity testing.</summary>
    public Tensor Encode(IBackend backend, Tensor latent, int batch, int t, int? nQOverride = null)
    {
        _ = backend;
        if (_embed[0] is null) throw new InvalidOperationException("XCodecEmaResidualVectorQuantizer weights not loaded.");
        int nQ = nQOverride ?? NCodebooks;
        if (nQ <= 0 || nQ > NCodebooks) throw new ArgumentOutOfRangeException(nameof(nQOverride), nQ, $"nQ must be in [1, {NCodebooks}].");
        EnsureSqNorms();

        Tensor codes = new(new TensorShape(nQ, batch, t), DType.I32);
        int* outp = (int*)codes.DataPointer;

        // Residual buffer (channels-first copy of latent).
        Tensor residual = new(latent.Shape, DType.F32);
        long bytes = latent.ElementCount * sizeof(float);
        Buffer.MemoryCopy((void*)latent.DataPointer, (void*)residual.DataPointer, bytes, bytes);
        float* rp = (float*)residual.DataPointer;

        Span<float> x = stackalloc float[Dimension <= 4096 ? Dimension : 0];
        float[]? xHeap = Dimension > 4096 ? new float[Dimension] : null;

        for (int q = 0; q < nQ; q++)
        {
            float* emb = (float*)_embed[q]!.DataPointer;
            float[] sq = _embedSqNorm[q]!;
            for (int b = 0; b < batch; b++)
            {
                for (int ti = 0; ti < t; ti++)
                {
                    // Gather x = residual[b, :, ti].
                    for (int d = 0; d < Dimension; d++)
                    {
                        float v = rp[(b * Dimension + d) * t + ti];
                        if (xHeap is not null) xHeap[d] = v; else x[d] = v;
                    }
                    // argmax over k of (2·x·e_k − ||e_k||²)  ==  argmin ||x − e_k||².
                    int bestIdx = 0;
                    float bestScore = float.MinValue;
                    for (int k = 0; k < CodebookSize; k++)
                    {
                        int rowBase = k * Dimension;
                        float dot = 0f;
                        if (xHeap is not null)
                            for (int d = 0; d < Dimension; d++) dot += xHeap[d] * emb[rowBase + d];
                        else
                            for (int d = 0; d < Dimension; d++) dot += x[d] * emb[rowBase + d];
                        float score = 2f * dot - sq[k];
                        if (score > bestScore) { bestScore = score; bestIdx = k; }
                    }
                    outp[(q * batch + b) * t + ti] = bestIdx;
                    // residual -= embed[bestIdx].
                    int rb = bestIdx * Dimension;
                    for (int d = 0; d < Dimension; d++)
                        rp[(b * Dimension + d) * t + ti] -= emb[rb + d];
                }
            }
        }
        residual.Dispose();
        return codes;
    }

    private void EnsureSqNorms()
    {
        for (int q = 0; q < NCodebooks; q++)
        {
            if (_embedSqNorm[q] is not null) continue;
            float[] sq = new float[CodebookSize];
            float* emb = (float*)_embed[q]!.DataPointer;
            for (int k = 0; k < CodebookSize; k++)
            {
                double s = 0d;
                int rowBase = k * Dimension;
                for (int d = 0; d < Dimension; d++) { float v = emb[rowBase + d]; s += (double)v * v; }
                sq[k] = (float)s;
            }
            _embedSqNorm[q] = sq;
        }
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        for (int q = 0; q < NCodebooks; q++)
            if (_embed[q] is not null) yield return _embed[q]!;
    }
}
