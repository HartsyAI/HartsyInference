using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Codecs.Mimi;

/// <summary>Mimi's split EMA residual vector quantizer DECODE path (kyutai/mimi, HF transformers layout).
/// The real codec splits the codebooks into a <b>semantic</b> RVQ (the first <c>num_semantic_quantizers</c>,
/// = 1) and an <b>acoustic</b> RVQ (the rest). Each codebook is an EMA codebook stored as
/// <c>embed_sum [vocab,dim]</c> + <c>cluster_usage [vocab]</c> with <c>embed = embed_sum / max(cluster_usage,
/// 1e-5)</c>; decode = sum of per-codebook <c>embedding(code)</c> lookups, then a 1x1 <c>output_proj</c>
/// (codebook_dim -> latent). Total = semantic + acoustic. Replaces the wrong factorized
/// <see cref="Dac.DacResidualVectorQuantizer"/> the old Mimi scaffold reused.</summary>
internal sealed unsafe class MimiSplitRvq
{
    private const float Eps = 1e-5f;
    private readonly string _p;
    private readonly int _numSemantic;
    private readonly int _numTotal;
    private readonly int _codebookDim;
    private readonly int _latentDim;

    // per-codebook reconstructed embeddings [vocab, dim] (semantic first, then acoustic)
    private Tensor?[] _embed = [];
    private Tensor? _semOutW, _semOutB, _acoOutW, _acoOutB;
    private Tensor? _semInW, _semInB, _acoInW, _acoInB;   // input_proj (latent -> codebook_dim), encode path

    public MimiSplitRvq(string prefix, int numSemantic, int numTotal, int codebookDim, int latentDim)
    {
        _p = prefix; _numSemantic = numSemantic; _numTotal = numTotal; _codebookDim = codebookDim; _latentDim = latentDim;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _embed = new Tensor?[_numTotal];
        for (int i = 0; i < _numTotal; i++)
        {
            bool sem = i < _numSemantic;
            string rvq = sem ? "semantic_residual_vector_quantizer" : "acoustic_residual_vector_quantizer";
            int li = sem ? i : i - _numSemantic;
            Tensor embedSum = WhisperOps.EnsureF32(w[$"{_p}.{rvq}.layers.{li}.codebook.embed_sum"]);   // [vocab, dim]
            Tensor usage = WhisperOps.EnsureF32(w[$"{_p}.{rvq}.layers.{li}.codebook.cluster_usage"]);   // [vocab]
            _embed[i] = ReconstructEmbed(embedSum, usage);
        }
        _semOutW = WhisperOps.EnsureF32(w[$"{_p}.semantic_residual_vector_quantizer.output_proj.weight"]);
        _semOutB = TryGet(w, $"{_p}.semantic_residual_vector_quantizer.output_proj.bias");
        _acoOutW = WhisperOps.EnsureF32(w[$"{_p}.acoustic_residual_vector_quantizer.output_proj.weight"]);
        _acoOutB = TryGet(w, $"{_p}.acoustic_residual_vector_quantizer.output_proj.bias");
        // Encode-path input projections (latent -> codebook_dim); optional (only present for the encode path).
        _semInW = TryGet(w, $"{_p}.semantic_residual_vector_quantizer.input_proj.weight");
        _semInB = TryGet(w, $"{_p}.semantic_residual_vector_quantizer.input_proj.bias");
        _acoInW = TryGet(w, $"{_p}.acoustic_residual_vector_quantizer.input_proj.weight");
        _acoInB = TryGet(w, $"{_p}.acoustic_residual_vector_quantizer.input_proj.bias");
    }

    private static Tensor? TryGet(IReadOnlyDictionary<string, Tensor> w, string k)
        => w.TryGetValue(k, out Tensor? t) ? WhisperOps.EnsureF32(t) : null;

    private Tensor ReconstructEmbed(Tensor embedSum, Tensor usage)
    {
        int vocab = (int)embedSum.Shape[0];
        int dim = (int)embedSum.Shape[1];
        Tensor outp = new(new TensorShape(vocab, dim), DType.F32);
        float* es = (float*)embedSum.DataPointer;
        float* cu = (float*)usage.DataPointer;
        float* op = (float*)outp.DataPointer;
        for (int v = 0; v < vocab; v++)
        {
            float denom = MathF.Max(cu[v], Eps);
            long b = (long)v * dim;
            for (int d = 0; d < dim; d++) op[b + d] = es[b + d] / denom;
        }
        return outp;
    }

    /// <summary>codes <c>[B, K, T]</c> (Int32) -> quantized embeddings <c>[B, latentDim, T]</c>.</summary>
    public Tensor Decode(IBackend backend, Tensor codes, int batch, int t)
    {
        // accumulate semantic and acoustic codebook-dim sums separately, then output_proj each, then add.
        Tensor semSum = new(new TensorShape(batch, _codebookDim, t), DType.F32);
        Tensor acoSum = new(new TensorShape(batch, _codebookDim, t), DType.F32);
        Zero(semSum); Zero(acoSum);
        int* cp = (int*)codes.DataPointer;
        float* sp = (float*)semSum.DataPointer;
        float* ap = (float*)acoSum.DataPointer;

        for (int i = 0; i < _numTotal; i++)
        {
            bool sem = i < _numSemantic;
            float* acc = sem ? sp : ap;
            float* emb = (float*)_embed[i]!.DataPointer;   // [vocab, dim]
            for (int b = 0; b < batch; b++)
                for (int ti = 0; ti < t; ti++)
                {
                    int code = cp[((long)b * _numTotal + i) * t + ti];
                    long eb = (long)code * _codebookDim;
                    for (int d = 0; d < _codebookDim; d++)
                        acc[((long)b * _codebookDim + d) * t + ti] += emb[eb + d];
                }
        }

        Tensor semOut = new(new TensorShape(batch, _latentDim, t), DType.F32);
        backend.Conv1d(semOut, semSum, _semOutW!, _semOutB, stride: 1, padLeft: 0, padRight: 0, dilation: 1, groups: 1);
        Tensor acoOut = new(new TensorShape(batch, _latentDim, t), DType.F32);
        backend.Conv1d(acoOut, acoSum, _acoOutW!, _acoOutB, stride: 1, padLeft: 0, padRight: 0, dilation: 1, groups: 1);
        semSum.Dispose(); acoSum.Dispose();

        Tensor total = new(new TensorShape(batch, _latentDim, t), DType.F32);
        backend.Add(total, semOut, acoOut);
        semOut.Dispose(); acoOut.Dispose();
        return total;
    }

    /// <summary>latent embeddings <c>[B, latentDim, T]</c> -> codes <c>[B, K, T]</c> (Int32), semantic
    /// codebooks first then acoustic. Mirror of <see cref="Decode"/>: the semantic and acoustic RVQs each run
    /// independently over the SAME latent (their own <c>input_proj</c>), and within each RVQ the codes are a
    /// nearest-neighbour residual chain — pick the closest codebook vector, subtract it, repeat. Requires the
    /// <c>input_proj</c> weights (present in the full codec checkpoint).</summary>
    public Tensor Encode(IBackend backend, Tensor latent, int batch, int t)
    {
        if (_semInW is null || _acoInW is null)
            throw new InvalidOperationException("MimiSplitRvq.Encode requires the quantizer input_proj weights.");
        // Codebook-major [K, batch, T] — the layout Mimi consumers read (nQ = Shape[0]); the RVQ decode's
        // [B, K, T] reader sees the identical k-major memory at batch=1.
        Tensor codes = new(new TensorShape(_numTotal, batch, t), DType.I32);
        int* cp = (int*)codes.DataPointer;

        // Semantic RVQ (codebooks [0, _numSemantic)) then acoustic RVQ (the rest), each over its own projection.
        EncodeRvq(backend, latent, batch, t, _semInW!, _semInB, 0, _numSemantic, cp);
        EncodeRvq(backend, latent, batch, t, _acoInW!, _acoInB, _numSemantic, _numTotal, cp);
        return codes;
    }

    /// <summary>Runs one residual VQ (codebooks <c>[first, last)</c>) over <paramref name="latent"/> projected by
    /// <paramref name="inW"/>, writing indices into <paramref name="cp"/> at codebook rows <c>[first, last)</c>.</summary>
    private void EncodeRvq(IBackend backend, Tensor latent, int batch, int t, Tensor inW, Tensor? inB, int first, int last, int* cp)
    {
        // input_proj: 1x1 conv (latentDim -> codebook_dim) per frame.
        Tensor proj = new(new TensorShape(batch, _codebookDim, t), DType.F32);
        backend.Conv1d(proj, latent, inW, inB, stride: 1, padLeft: 0, padRight: 0, dilation: 1, groups: 1);
        float* rp = (float*)proj.DataPointer;   // residual, updated in place; layout [B, dim, T] (channel-major)

        for (int i = first; i < last; i++)
        {
            float* emb = (float*)_embed[i]!.DataPointer;   // [vocab, dim]
            int vocab = (int)_embed[i]!.Shape[0];
            for (int b = 0; b < batch; b++)
                for (int ti = 0; ti < t; ti++)
                {
                    // Nearest codebook vector to residual[b, :, ti] (channel stride = t).
                    long baseIdx = (long)b * _codebookDim * t + ti;
                    int best = 0; float bestDist = float.MaxValue;
                    for (int v = 0; v < vocab; v++)
                    {
                        long eb = (long)v * _codebookDim;
                        float dist = 0f;
                        for (int d = 0; d < _codebookDim; d++)
                        {
                            float diff = rp[baseIdx + (long)d * t] - emb[eb + d];
                            dist += diff * diff;
                        }
                        if (dist < bestDist) { bestDist = dist; best = v; }
                    }
                    cp[((long)i * batch + b) * t + ti] = best;   // [K, batch, T]
                    // Subtract the chosen codebook vector to form the next residual.
                    long cb = (long)best * _codebookDim;
                    for (int d = 0; d < _codebookDim; d++) rp[baseIdx + (long)d * t] -= emb[cb + d];
                }
        }
        proj.Dispose();
    }

    private static void Zero(Tensor t)
    {
        float* p = (float*)t.DataPointer;
        long n = t.Shape.ElementCount;
        for (long i = 0; i < n; i++) p[i] = 0f;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor? e in _embed) if (e is not null) yield return e;
        if (_semOutW is not null) yield return _semOutW;
        if (_semOutB is not null) yield return _semOutB;
        if (_acoOutW is not null) yield return _acoOutW;
        if (_acoOutB is not null) yield return _acoOutB;
        if (_semInW is not null) yield return _semInW;
        if (_semInB is not null) yield return _semInB;
        if (_acoInW is not null) yield return _acoInW;
        if (_acoInB is not null) yield return _acoInB;
    }
}
