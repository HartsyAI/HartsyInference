using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Codecs.Snac;

/// <summary>SNAC's hierarchical residual VQ. Each codebook has its own <c>stride</c> — bigger stride = coarser temporal resolution.</summary>
/// <remarks>
/// <para>Per-codebook forward (encode):</para>
/// <list type="number">
///   <item>If <c>stride &gt; 1</c>: <c>avg_pool1d</c> the residual to <c>T / stride</c> samples.</item>
///   <item><c>in_proj</c> 1×1 conv: <c>latent_dim → codebook_dim</c>.</item>
///   <item>L2-normalize the query and look up the nearest codeword (cosine similarity).</item>
///   <item>Index the codebook, run <c>out_proj</c> 1×1 conv back to <c>latent_dim</c>.</item>
///   <item>If <c>stride &gt; 1</c>: repeat-interleave back up to the bottleneck rate.</item>
///   <item>Subtract from the residual.</item>
/// </list>
///
/// <para>Codes for codebook <c>i</c> have shape <c>[batch, T / VqStrides[i]]</c> — that's
/// why SNAC returns a list of variable-length code tensors rather than the rectangular
/// <c>[nQ, B, T]</c> layout used by EnCodec / DAC.</para></remarks>
internal sealed unsafe class SnacResidualVectorQuantizer
{
    public int NCodebooks { get; }
    public int CodebookSize { get; }
    public int CodebookDim { get; }
    public int LatentDim { get; }
    public IReadOnlyList<int> VqStrides { get; }

    private readonly Tensor?[] _inProjW;
    private readonly Tensor?[] _inProjB;
    private readonly Tensor?[] _outProjW;
    private readonly Tensor?[] _outProjB;
    private readonly Tensor?[] _codebooks;
    private readonly Tensor?[] _codebooksNorm;

    public SnacResidualVectorQuantizer(int nCodebooks, int codebookSize, int codebookDim, int latentDim, IReadOnlyList<int> vqStrides)
    {
        if (vqStrides.Count != nCodebooks)
            throw new ArgumentException($"vqStrides.Count ({vqStrides.Count}) must equal nCodebooks ({nCodebooks}).");
        NCodebooks = nCodebooks;
        CodebookSize = codebookSize;
        CodebookDim = codebookDim;
        LatentDim = latentDim;
        VqStrides = vqStrides;
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

    /// <summary>Encodes a latent. Output is a list of length <see cref="NCodebooks"/>; entry <c>i</c> is shape <c>[batch, T / VqStrides[i]]</c> Int32. Caller owns disposal of every returned tensor.</summary>
    public Tensor[] Encode(IBackend backend, Tensor latent, int batch, int t)
    {
        if (_inProjW[0] is null) throw new InvalidOperationException("SnacResidualVectorQuantizer weights not loaded.");

        Tensor residual = new(latent.Shape, DType.F32);
        long bytes = latent.ElementCount * sizeof(float);
        Buffer.MemoryCopy((void*)latent.DataPointer, (void*)residual.DataPointer, bytes, bytes);

        Tensor[] result = new Tensor[NCodebooks];
        Span<float> normalizedQuery = stackalloc float[CodebookDim];

        for (int q = 0; q < NCodebooks; q++)
        {
            int stride = VqStrides[q];
            int tQuant = t / stride;
            if (tQuant * stride != t)
                throw new InvalidOperationException($"Latent length {t} is not divisible by codebook {q} stride {stride}.");

            // Downsample the residual via average pooling.
            Tensor downsampled = stride == 1
                ? Clone(residual)
                : AvgPool1d(residual, batch, LatentDim, t, stride);

            // in_proj 1×1 conv: latent_dim → codebook_dim.
            Tensor projected = new(new TensorShape(batch, CodebookDim, tQuant), DType.F32);
            backend.Conv1d(projected, downsampled, _inProjW[q]!, _inProjB[q],
                stride: 1, padLeft: 0, padRight: 0, dilation: 1, groups: 1);
            downsampled.Dispose();

            // Nearest-neighbor lookup.
            Tensor codes = new(new TensorShape(batch, tQuant), DType.I32);
            int* cp = (int*)codes.DataPointer;
            float* pp = (float*)projected.DataPointer;
            float* cbNorm = (float*)_codebooksNorm[q]!.DataPointer;

            for (int b = 0; b < batch; b++)
            {
                for (int ti = 0; ti < tQuant; ti++)
                {
                    double sumSq = 0d;
                    for (int d = 0; d < CodebookDim; d++)
                    {
                        float v = pp[(b * CodebookDim + d) * tQuant + ti];
                        normalizedQuery[d] = v;
                        sumSq += (double)v * v;
                    }
                    float invNorm = (float)(1.0 / Math.Sqrt(sumSq + 1e-12));
                    for (int d = 0; d < CodebookDim; d++) normalizedQuery[d] *= invNorm;

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
                    cp[b * tQuant + ti] = bestIdx;
                }
            }
            projected.Dispose();
            result[q] = codes;

            // Reconstruct + out_proj + repeat-interleave + subtract from residual.
            Tensor quantizedSmall = new(new TensorShape(batch, CodebookDim, tQuant), DType.F32);
            float* qp = (float*)quantizedSmall.DataPointer;
            float* cb = (float*)_codebooks[q]!.DataPointer;
            for (int b = 0; b < batch; b++)
            {
                for (int ti = 0; ti < tQuant; ti++)
                {
                    int idx = cp[b * tQuant + ti];
                    int rowBase = idx * CodebookDim;
                    for (int d = 0; d < CodebookDim; d++)
                        qp[(b * CodebookDim + d) * tQuant + ti] = cb[rowBase + d];
                }
            }

            Tensor reprojSmall = new(new TensorShape(batch, LatentDim, tQuant), DType.F32);
            backend.Conv1d(reprojSmall, quantizedSmall, _outProjW[q]!, _outProjB[q],
                stride: 1, padLeft: 0, padRight: 0, dilation: 1, groups: 1);
            quantizedSmall.Dispose();

            Tensor reprojFull = stride == 1
                ? reprojSmall
                : RepeatInterleave(reprojSmall, batch, LatentDim, tQuant, stride);
            if (stride > 1) reprojSmall.Dispose();

            float* rp = (float*)residual.DataPointer;
            float* xp = (float*)reprojFull.DataPointer;
            long n = residual.ElementCount;
            for (long i = 0; i < n; i++) rp[i] -= xp[i];
            reprojFull.Dispose();
        }

        residual.Dispose();
        return result;
    }

    /// <summary>Decodes per-codebook codes back to a latent. <paramref name="codes"/> must have length <see cref="NCodebooks"/>; entry <c>i</c> shape <c>[B, T / VqStrides[i]]</c>. Output is channels-first <c>[B, latent_dim, T]</c> where <c>T</c> is derived from the largest code stream (codebook with stride=1).</summary>
    public Tensor Decode(IBackend backend, IReadOnlyList<Tensor> codes, int batch)
    {
        if (codes.Count != NCodebooks)
            throw new ArgumentException($"Expected {NCodebooks} code tensors, got {codes.Count}.");

        // Derive the full T from the codebook with stride=1 (or smallest stride).
        int t = -1;
        for (int q = 0; q < NCodebooks; q++)
        {
            int tQuant = (int)codes[q].Shape[1];
            int tFull = tQuant * VqStrides[q];
            if (t == -1) t = tFull;
            else if (t != tFull)
                throw new InvalidOperationException($"Codes are inconsistent: codebook 0 implies T={t}, codebook {q} implies T={tFull}.");
        }
        if (t == -1) t = 0;

        Tensor latent = new(new TensorShape(batch, LatentDim, t), DType.F32);
        long total = latent.ElementCount;
        float* lp = (float*)latent.DataPointer;
        for (long i = 0; i < total; i++) lp[i] = 0f;

        for (int q = 0; q < NCodebooks; q++)
        {
            int stride = VqStrides[q];
            int tQuant = t / stride;
            int* cp = (int*)codes[q].DataPointer;

            // Reconstruct codebook_dim quantized vector.
            Tensor quantizedSmall = new(new TensorShape(batch, CodebookDim, tQuant), DType.F32);
            float* qp = (float*)quantizedSmall.DataPointer;
            float* cb = (float*)_codebooks[q]!.DataPointer;
            for (int b = 0; b < batch; b++)
            {
                for (int ti = 0; ti < tQuant; ti++)
                {
                    int idx = cp[b * tQuant + ti];
                    int rowBase = idx * CodebookDim;
                    for (int d = 0; d < CodebookDim; d++)
                        qp[(b * CodebookDim + d) * tQuant + ti] = cb[rowBase + d];
                }
            }

            Tensor reprojSmall = new(new TensorShape(batch, LatentDim, tQuant), DType.F32);
            backend.Conv1d(reprojSmall, quantizedSmall, _outProjW[q]!, _outProjB[q],
                stride: 1, padLeft: 0, padRight: 0, dilation: 1, groups: 1);
            quantizedSmall.Dispose();

            Tensor reprojFull = stride == 1
                ? reprojSmall
                : RepeatInterleave(reprojSmall, batch, LatentDim, tQuant, stride);
            if (stride > 1) reprojSmall.Dispose();

            float* xp = (float*)reprojFull.DataPointer;
            for (long i = 0; i < total; i++) lp[i] += xp[i];
            reprojFull.Dispose();
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

    private static Tensor Clone(Tensor src)
    {
        Tensor copy = new(src.Shape, DType.F32);
        long bytes = src.ElementCount * sizeof(float);
        Buffer.MemoryCopy((void*)src.DataPointer, (void*)copy.DataPointer, bytes, bytes);
        return copy;
    }

    /// <summary>1D average pool — non-overlapping windows of length <paramref name="stride"/>. Input <c>[B, C, T]</c> → output <c>[B, C, T/stride]</c>.</summary>
    private static Tensor AvgPool1d(Tensor src, int batch, int channels, int t, int stride)
    {
        int tOut = t / stride;
        Tensor result = new(new TensorShape(batch, channels, tOut), DType.F32);
        float* sp = (float*)src.DataPointer;
        float* dp = (float*)result.DataPointer;
        float invStride = 1f / stride;
        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                int srcBase = (b * channels + c) * t;
                int dstBase = (b * channels + c) * tOut;
                for (int j = 0; j < tOut; j++)
                {
                    float sum = 0f;
                    int srcStart = j * stride;
                    for (int k = 0; k < stride; k++) sum += sp[srcBase + srcStart + k];
                    dp[dstBase + j] = sum * invStride;
                }
            }
        }
        return result;
    }

    /// <summary>Repeat-interleave along the time axis. Each element repeats <paramref name="repeats"/> times — matches PyTorch's <c>x.repeat_interleave(stride, dim=-1)</c>.</summary>
    private static Tensor RepeatInterleave(Tensor src, int batch, int channels, int tIn, int repeats)
    {
        int tOut = tIn * repeats;
        Tensor result = new(new TensorShape(batch, channels, tOut), DType.F32);
        float* sp = (float*)src.DataPointer;
        float* dp = (float*)result.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                int srcBase = (b * channels + c) * tIn;
                int dstBase = (b * channels + c) * tOut;
                for (int j = 0; j < tIn; j++)
                {
                    float v = sp[srcBase + j];
                    int outStart = j * repeats;
                    for (int k = 0; k < repeats; k++) dp[dstBase + outStart + k] = v;
                }
            }
        }
        return result;
    }
}
