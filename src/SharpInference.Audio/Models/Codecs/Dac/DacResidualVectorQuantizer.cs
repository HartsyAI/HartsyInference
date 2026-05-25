using SharpInference.Audio.Models.Whisper;
using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Audio.Models.Codecs.Dac;

/// <summary>DAC-style residual vector quantizer. Each codebook layer has its own
/// projection convs that map the latent down to a small <c>codebook_dim</c> for the
/// nearest-neighbor lookup, then back up to the latent dimension for the residual
/// subtraction. Both projections are 1×1 weight-normed Conv1d, fused at load time.
///
/// <para>This differs from the simpler <see cref="ResidualVectorQuantizer"/> we built
/// for EnCodec, which has no per-layer projection. The DAC variant gives each codebook
/// its own subspace and dramatically improves codebook utilization — that's the whole
/// reason DAC's reconstructions outperform EnCodec at comparable bitrates.</para>
///
/// <para>The nearest-neighbor lookup is in L2-normalized space (cosine similarity).
/// We L2-normalize both the query (post-in_proj latent slice) and the codebook entries
/// before comparing distances. The result is equivalent to picking the codeword with
/// the largest dot product against the (normalized) query.</para>
///
/// <para>State-dict layout (matches <c>descript-audio-codec</c>):
/// <list type="bullet">
///   <item><c>quantizer.quantizers.{i}.in_proj.weight_g</c> / <c>weight_v</c> / <c>bias</c></item>
///   <item><c>quantizer.quantizers.{i}.out_proj.weight_g</c> / <c>weight_v</c> / <c>bias</c></item>
///   <item><c>quantizer.quantizers.{i}.codebook.weight</c> — <c>[codebook_size, codebook_dim]</c></item>
/// </list></para></summary>
internal sealed unsafe class DacResidualVectorQuantizer
{
    public int NCodebooks { get; }
    public int CodebookSize { get; }
    public int CodebookDim { get; }
    public int LatentDim { get; }

    private readonly Tensor?[] _inProjW;
    private readonly Tensor?[] _inProjB;
    private readonly Tensor?[] _outProjW;
    private readonly Tensor?[] _outProjB;
    private readonly Tensor?[] _codebooks;        // [codebook_size, codebook_dim]
    private readonly Tensor?[] _codebooksNorm;    // L2-normalized version of each codebook, cached at load

    public DacResidualVectorQuantizer(int nCodebooks, int codebookSize, int codebookDim, int latentDim)
    {
        NCodebooks = nCodebooks;
        CodebookSize = codebookSize;
        CodebookDim = codebookDim;
        LatentDim = latentDim;
        _inProjW = new Tensor?[nCodebooks];
        _inProjB = new Tensor?[nCodebooks];
        _outProjW = new Tensor?[nCodebooks];
        _outProjB = new Tensor?[nCodebooks];
        _codebooks = new Tensor?[nCodebooks];
        _codebooksNorm = new Tensor?[nCodebooks];
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        for (int q = 0; q < NCodebooks; q++)
        {
            string p = $"{prefix}.quantizers.{q}";
            _inProjW[q] = LoadFusedWeight(w, $"{p}.in_proj");
            _inProjB[q] = WhisperOps.EnsureF32(w[$"{p}.in_proj.bias"]);
            _outProjW[q] = LoadFusedWeight(w, $"{p}.out_proj");
            _outProjB[q] = WhisperOps.EnsureF32(w[$"{p}.out_proj.bias"]);
            _codebooks[q] = WhisperOps.EnsureF32(w[$"{p}.codebook.weight"]);
            _codebooksNorm[q] = L2NormalizeRows(_codebooks[q]!, CodebookSize, CodebookDim);
        }
    }

    /// <summary>Encodes a continuous latent into integer codes. <paramref name="latent"/>
    /// is channels-first <c>[batch, latent_dim, T]</c>; output is <c>[nQ, batch, T]</c>
    /// Int32.</summary>
    public Tensor Encode(IBackend backend, Tensor latent, int batch, int t, int? nQOverride = null)
    {
        if (_inProjW[0] is null) throw new InvalidOperationException("DacResidualVectorQuantizer weights not loaded.");
        int nQ = nQOverride ?? NCodebooks;
        if (nQ <= 0 || nQ > NCodebooks) throw new ArgumentOutOfRangeException(nameof(nQOverride), nQ, $"nQ must be in [1, {NCodebooks}].");

        Tensor codes = new(new TensorShape(nQ, batch, t), DType.I32);
        int* cp = (int*)codes.DataPointer;

        // Working residual buffer in latent space. We mutate this as we subtract each
        // codebook's contribution. Allocate as a tensor so the per-layer in_proj can
        // run via backend.Conv1d on it.
        Tensor residual = new(latent.Shape, DType.F32);
        long bytes = latent.ElementCount * sizeof(float);
        Buffer.MemoryCopy((void*)latent.DataPointer, (void*)residual.DataPointer, bytes, bytes);

        // Reusable scratch query buffer for the nearest-neighbor inner loop. Allocated
        // once outside the codebook loop per CA2014 — CodebookDim is constant for the
        // life of the quantizer (8 in every published DAC config) so reuse is safe.
        Span<float> normalizedQuery = stackalloc float[CodebookDim];

        for (int q = 0; q < nQ; q++)
        {
            // in_proj: latent_dim → codebook_dim (1×1 conv).
            Tensor projected = new(new TensorShape(batch, CodebookDim, t), DType.F32);
            backend.Conv1d(projected, residual, _inProjW[q]!, _inProjB[q],
                stride: 1, padLeft: 0, padRight: 0, dilation: 1, groups: 1);

            // Nearest-neighbor lookup in L2-normalized cosine space. Per (b, t):
            //   query = projected[b, :, t] / ||projected[b, :, t]||
            //   idx = argmax_k dot(query, codebookNorm[k, :])
            float* pp = (float*)projected.DataPointer;
            float* cbNorm = (float*)_codebooksNorm[q]!.DataPointer;

            for (int b = 0; b < batch; b++)
            {
                for (int ti = 0; ti < t; ti++)
                {
                    // Build [B, codebook_dim, T] is channels-first so query is at
                    // index (b * codebook_dim + d) * t + ti for d in [0, codebook_dim).
                    double sumSq = 0d;
                    for (int d = 0; d < CodebookDim; d++)
                    {
                        float v = pp[(b * CodebookDim + d) * t + ti];
                        normalizedQuery[d] = v;
                        sumSq += (double)v * v;
                    }
                    float invNorm = (float)(1.0 / Math.Sqrt(sumSq + 1e-12));
                    for (int d = 0; d < CodebookDim; d++) normalizedQuery[d] *= invNorm;

                    // Find nearest codeword by maximum dot product.
                    int bestIdx = 0;
                    float bestDot = float.MinValue;
                    for (int k = 0; k < CodebookSize; k++)
                    {
                        float dot = 0f;
                        int rowBase = k * CodebookDim;
                        for (int d = 0; d < CodebookDim; d++)
                            dot += normalizedQuery[d] * cbNorm[rowBase + d];
                        if (dot > bestDot)
                        {
                            bestDot = dot;
                            bestIdx = k;
                        }
                    }
                    cp[(q * batch + b) * t + ti] = bestIdx;
                }
            }
            projected.Dispose();

            // Reconstruct the quantized codebook_dim vector, run out_proj back to latent_dim,
            // subtract from residual.
            Tensor quantized = new(new TensorShape(batch, CodebookDim, t), DType.F32);
            float* qp = (float*)quantized.DataPointer;
            float* cb = (float*)_codebooks[q]!.DataPointer;
            for (int b = 0; b < batch; b++)
            {
                for (int ti = 0; ti < t; ti++)
                {
                    int idx = cp[(q * batch + b) * t + ti];
                    int rowBase = idx * CodebookDim;
                    for (int d = 0; d < CodebookDim; d++)
                        qp[(b * CodebookDim + d) * t + ti] = cb[rowBase + d];
                }
            }

            Tensor reproj = new(new TensorShape(batch, LatentDim, t), DType.F32);
            backend.Conv1d(reproj, quantized, _outProjW[q]!, _outProjB[q],
                stride: 1, padLeft: 0, padRight: 0, dilation: 1, groups: 1);
            quantized.Dispose();

            // residual -= reproj.
            float* rp = (float*)residual.DataPointer;
            float* xp = (float*)reproj.DataPointer;
            long n = residual.ElementCount;
            for (long i = 0; i < n; i++) rp[i] -= xp[i];
            reproj.Dispose();
        }

        residual.Dispose();
        return codes;
    }

    /// <summary>Decodes integer codes back to a continuous latent. Codes shape
    /// <c>[nQ, batch, T]</c>; output channels-first <c>[batch, latent_dim, T]</c>.</summary>
    public Tensor Decode(IBackend backend, Tensor codes, int batch, int t)
    {
        if (_inProjW[0] is null) throw new InvalidOperationException("DacResidualVectorQuantizer weights not loaded.");
        int nQ = (int)codes.Shape[0];
        if (nQ > NCodebooks) throw new ArgumentException($"codes nQ ({nQ}) exceeds NCodebooks ({NCodebooks}).");

        Tensor latent = new(new TensorShape(batch, LatentDim, t), DType.F32);
        long total = latent.ElementCount;
        float* lp = (float*)latent.DataPointer;
        for (long i = 0; i < total; i++) lp[i] = 0f;

        int* cp = (int*)codes.DataPointer;

        for (int q = 0; q < nQ; q++)
        {
            // Reconstruct quantized vector from codes.
            Tensor quantized = new(new TensorShape(batch, CodebookDim, t), DType.F32);
            float* qp = (float*)quantized.DataPointer;
            float* cb = (float*)_codebooks[q]!.DataPointer;
            for (int b = 0; b < batch; b++)
            {
                for (int ti = 0; ti < t; ti++)
                {
                    int idx = cp[(q * batch + b) * t + ti];
                    int rowBase = idx * CodebookDim;
                    for (int d = 0; d < CodebookDim; d++)
                        qp[(b * CodebookDim + d) * t + ti] = cb[rowBase + d];
                }
            }

            // out_proj to latent space.
            Tensor reproj = new(new TensorShape(batch, LatentDim, t), DType.F32);
            backend.Conv1d(reproj, quantized, _outProjW[q]!, _outProjB[q],
                stride: 1, padLeft: 0, padRight: 0, dilation: 1, groups: 1);
            quantized.Dispose();

            // latent += reproj.
            float* xp = (float*)reproj.DataPointer;
            for (long i = 0; i < total; i++) lp[i] += xp[i];
            reproj.Dispose();
        }

        return latent;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        for (int q = 0; q < NCodebooks; q++)
        {
            Tensor?[] all = [_inProjW[q], _inProjB[q], _outProjW[q], _outProjB[q], _codebooks[q]];
            foreach (Tensor? t in all) if (t is not null) yield return t;
        }
    }

    private static Tensor LoadFusedWeight(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        Tensor g = WhisperOps.EnsureF32(w[$"{prefix}.weight_g"]);
        Tensor v = WhisperOps.EnsureF32(w[$"{prefix}.weight_v"]);
        return WeightNormFusion.Fuse(g, v);
    }

    /// <summary>L2-normalizes each row of a 2D codebook tensor. Cached once at load time
    /// so the cosine-similarity inner loop is a pure dot product.</summary>
    private static Tensor L2NormalizeRows(Tensor src, int rows, int dim)
    {
        Tensor result = new(src.Shape, DType.F32);
        float* sp = (float*)src.DataPointer;
        float* dp = (float*)result.DataPointer;
        for (int r = 0; r < rows; r++)
        {
            double sumSq = 0d;
            int rowBase = r * dim;
            for (int d = 0; d < dim; d++) sumSq += (double)sp[rowBase + d] * sp[rowBase + d];
            float invNorm = (float)(1.0 / Math.Sqrt(sumSq + 1e-12));
            for (int d = 0; d < dim; d++) dp[rowBase + d] = sp[rowBase + d] * invNorm;
        }
        return result;
    }
}
