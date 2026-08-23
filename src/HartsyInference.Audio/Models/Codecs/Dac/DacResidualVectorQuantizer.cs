using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Codecs.Dac;

/// <summary>DAC-style residual vector quantizer — each codebook layer has its own 1×1 weight-normed projection convs down to <c>codebook_dim</c> for the nearest-neighbor lookup and back up to the latent dimension for the residual subtraction, fused at load time.</summary>
/// <remarks>
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
/// </list></para>
/// </remarks>
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
            _codebooksNorm[q] = VqOps.L2NormalizeRows(_codebooks[q]!, CodebookSize, CodebookDim);
        }
    }

    /// <summary>Encodes a continuous latent into integer codes. <paramref name="latent"/> is channels-first <c>[batch, latent_dim, T]</c>; output is <c>[nQ, batch, T]</c> Int32.</summary>
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

        for (int q = 0; q < nQ; q++)
        {
            // in_proj: latent_dim → codebook_dim (1×1 conv).
            Tensor projected = new(new TensorShape(batch, CodebookDim, t), DType.F32);
            backend.Conv1d(projected, residual, _inProjW[q]!, _inProjB[q],
                stride: 1, padLeft: 0, padRight: 0, dilation: 1, groups: 1);

            // Codes for codebook q occupy the [batch, t] plane at offset q in the [nQ, batch, T] output.
            int* qCodes = cp + (long)q * batch * t;
            VqOps.NearestCodebookIndices((float*)projected.DataPointer, (float*)_codebooksNorm[q]!.DataPointer,
                qCodes, batch, t, CodebookDim, CodebookSize);
            projected.Dispose();

            // Reconstruct the quantized codebook_dim vector, run out_proj back to latent_dim,
            // subtract from residual.
            Tensor quantized = VqOps.GatherCodebookVectors(_codebooks[q]!, qCodes, batch, t, CodebookDim);

            Tensor reproj = new(new TensorShape(batch, LatentDim, t), DType.F32);
            backend.Conv1d(reproj, quantized, _outProjW[q]!, _outProjB[q],
                stride: 1, padLeft: 0, padRight: 0, dilation: 1, groups: 1);
            quantized.Dispose();

            float* rp = (float*)residual.DataPointer;
            float* xp = (float*)reproj.DataPointer;
            long n = residual.ElementCount;
            for (long i = 0; i < n; i++) rp[i] -= xp[i];
            reproj.Dispose();
        }

        residual.Dispose();
        return codes;
    }

    /// <summary>Decodes integer codes back to a continuous latent. Codes shape <c>[nQ, batch, T]</c>; output channels-first <c>[batch, latent_dim, T]</c>.</summary>
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
            Tensor quantized = VqOps.GatherCodebookVectors(_codebooks[q]!, cp + (long)q * batch * t, batch, t, CodebookDim);

            // out_proj to latent space.
            Tensor reproj = new(new TensorShape(batch, LatentDim, t), DType.F32);
            backend.Conv1d(reproj, quantized, _outProjW[q]!, _outProjB[q],
                stride: 1, padLeft: 0, padRight: 0, dilation: 1, groups: 1);
            quantized.Dispose();

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
        return WeightNormFusion.Compose(w, prefix);
    }
}
