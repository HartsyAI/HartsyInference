using HartsyInference.Audio.Dsp;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.IndexTts2;

/// <summary>IndexTTS-2's explicit 8-dim emotion-vector mode (the real <c>infer_generator</c>'s <c>emo_vector</c>
/// argument) — a plain nearest-neighbor lookup against two small bundled exemplar banks shipped in the main repo
/// (<c>feat1.pt</c> <c>[73, 192]</c>, <c>feat2.pt</c> <c>[73, 1280]</c>; both bare-tensor-root pickles, no learned
/// model involved). Real source read in full (<c>model_v2.py</c>'s <c>QwenEmotion</c>-adjacent helpers and
/// <c>infer_v2_5.py</c>'s <c>infer_generator</c>):
/// <code>
/// def find_most_similar_cosine(query_vector, matrix):
///     similarities = F.cosine_similarity(query_vector.float(), matrix.float(), dim=1)
///     return torch.argmax(similarities)
///
/// def normalize_emo_vec(self, emo_vector, apply_bias=True):
///     if apply_bias:
///         emo_bias = [0.9375, 0.875, 1.0, 1.0, 0.9375, 0.9375, 0.6875, 0.5625]
///         emo_vector = [vec * bias for vec, bias in zip(emo_vector, emo_bias)]
///     emo_sum = sum(emo_vector)
///     if emo_sum > 0.8:
///         emo_vector = [vec * (0.8 / emo_sum) for vec in emo_vector]
///     return emo_vector
///
/// # infer_generator, emo_vector branch (weights used exactly as given — only the WebUI calls normalize_emo_vec):
/// weight_vector = torch.tensor(emo_vector)
/// random_index = [find_most_similar_cosine(style, bank) for bank in self.spk_matrix]   # or random, if use_random
/// emo_matrix = torch.cat([bank[i].unsqueeze(0) for i, bank in zip(random_index, self.emo_matrix)], 0)
/// emovec_mat = torch.sum(weight_vector.unsqueeze(1) * emo_matrix, 0).unsqueeze(0)
/// </code>
/// <paramref name="style"/> is the SAME CAM++ embedding already computed for speaker/S2Mel conditioning (one
/// forward pass serves this too). The result feeds the GPT's <c>emo_vec</c> directly (it is already in the GPT's
/// 1280-dim hidden space — <c>feat2.pt</c>'s own width), bypassing <see cref="IndexTts2T2sDecoder.ComputeEmoVec"/>'s
/// Conformer+Perceiver path entirely: confirmed from the real source, explicit-vector mode never calls
/// <c>get_emo_conditioning</c>.</summary>
internal sealed unsafe class IndexTts2EmotionVectorLookup : IDisposable
{
    /// <summary>happy, angry, sad, afraid, disgusted, melancholic, surprised, calm — the real fixed category order
    /// and per-category exemplar-bank sizes (confirmed: <c>sum(EmoNum) == 73 == feat1.pt/feat2.pt's row count</c>).</summary>
    public static readonly int[] EmoNum = [3, 17, 2, 8, 4, 5, 10, 24];

    /// <summary>Real <c>normalize_emo_vec</c>'s fixed per-category de-emphasis bias.</summary>
    public static readonly float[] Bias = [0.9375f, 0.875f, 1.0f, 1.0f, 0.9375f, 0.9375f, 0.6875f, 0.5625f];

    private readonly Tensor _spkMatrix; // [73, styleDim] — the real feat1.pt
    private readonly Tensor _emoMatrix; // [73, hidden]   — the real feat2.pt
    private readonly int[] _offsets;    // length EmoNum.Length + 1, cumulative row starts per category
    private int _disposed;

    /// <param name="spkMatrix">Real <c>feat1.pt</c>'s bare tensor, <c>[sum(EmoNum), styleDim]</c> (styleDim 192 —
    /// the CAM++ embedding width).</param>
    /// <param name="emoMatrix">Real <c>feat2.pt</c>'s bare tensor, <c>[sum(EmoNum), hidden]</c> (hidden 1280 — the
    /// GPT's own width). Ownership of both tensors transfers to this instance.</param>
    public IndexTts2EmotionVectorLookup(Tensor spkMatrix, Tensor emoMatrix)
    {
        int total = EmoNum.Sum();
        _spkMatrix = WhisperOps.EnsureF32(spkMatrix);
        _emoMatrix = WhisperOps.EnsureF32(emoMatrix);
        if (_spkMatrix.Shape.Rank != 2 || _spkMatrix.Shape[0] != total)
            throw new ArgumentException($"feat1 (spk_matrix) must be [{total}, styleDim]; got {_spkMatrix.Shape}.", nameof(spkMatrix));
        if (_emoMatrix.Shape.Rank != 2 || _emoMatrix.Shape[0] != total)
            throw new ArgumentException($"feat2 (emo_matrix) must be [{total}, hidden]; got {_emoMatrix.Shape}.", nameof(emoMatrix));

        _offsets = new int[EmoNum.Length + 1];
        for (int i = 0; i < EmoNum.Length; i++) _offsets[i + 1] = _offsets[i] + EmoNum[i];
    }

    /// <summary>Real <c>normalize_emo_vec</c>: per-category bias, then rescale so the sum never exceeds 0.8.</summary>
    public static float[] NormalizeEmoVec(ReadOnlySpan<float> emoVector, bool applyBias = true)
    {
        if (emoVector.Length != EmoNum.Length)
            throw new ArgumentException($"Expected an {EmoNum.Length}-dim emotion vector, got {emoVector.Length}.", nameof(emoVector));
        float[] result = new float[EmoNum.Length];
        float sum = 0f;
        for (int i = 0; i < result.Length; i++)
        {
            result[i] = applyBias ? emoVector[i] * Bias[i] : emoVector[i];
            sum += result[i];
        }
        if (sum > 0.8f)
        {
            float scale = 0.8f / sum;
            for (int i = 0; i < result.Length; i++) result[i] *= scale;
        }
        return result;
    }

    /// <summary>Real weighted exemplar-selection producing the final <c>emovec_mat</c>, shape <c>[1, hidden]</c>.
    /// <paramref name="weights"/> are used exactly as given (apply <see cref="NormalizeEmoVec"/> first for the
    /// WebUI's de-emphasis behaviour). <paramref name="style"/> is the current speaker's CAM++ embedding, <c>[1, styleDim]</c> or <c>[styleDim]</c>.
    /// <paramref name="useRandom"/> mirrors the real (debug-only) <c>use_random</c> path — a uniform random index
    /// per category instead of the cosine-nearest exemplar.</summary>
    public Tensor ComputeEmoVecMat(ReadOnlySpan<float> weights, Tensor style, bool useRandom, ref uint rngState)
    {
        ThrowIfDisposed();
        if (weights.Length != EmoNum.Length)
            throw new ArgumentException($"Expected an {EmoNum.Length}-dim emotion vector, got {weights.Length}.", nameof(weights));
        int styleDim = (int)_spkMatrix.Shape[1];
        int hidden = (int)_emoMatrix.Shape[1];
        float* stylePtr = (float*)style.DataPointer;

        Tensor result = new(new TensorShape(1, hidden), DType.F32);
        float* rp = (float*)result.DataPointer;
        for (int c = 0; c < hidden; c++) rp[c] = 0f;

        for (int cat = 0; cat < EmoNum.Length; cat++)
        {
            int n = EmoNum[cat];
            int offset = _offsets[cat];
            int index = useRandom
                ? Math.Min((int)(DeterministicRng.NextUniform(ref rngState) * n), n - 1)
                : FindMostSimilarCosine(stylePtr, styleDim, offset, n);

            float* row = (float*)_emoMatrix.DataPointer + (long)(offset + index) * hidden;
            float w = weights[cat];
            for (int c = 0; c < hidden; c++) rp[c] += w * row[c];
        }
        return result;
    }

    /// <summary>Real <c>find_most_similar_cosine</c>: argmax cosine similarity between <paramref name="style"/> and
    /// each of the <paramref name="count"/> exemplar rows starting at <paramref name="offset"/> in <see cref="_spkMatrix"/>.</summary>
    private int FindMostSimilarCosine(float* style, int styleDim, int offset, int count)
    {
        float styleNorm = 0f;
        for (int c = 0; c < styleDim; c++) styleNorm += style[c] * style[c];
        styleNorm = MathF.Sqrt(styleNorm);

        int best = 0;
        float bestSim = float.NegativeInfinity;
        for (int i = 0; i < count; i++)
        {
            float* row = (float*)_spkMatrix.DataPointer + (long)(offset + i) * styleDim;
            float dot = 0f, rowNorm = 0f;
            for (int c = 0; c < styleDim; c++) { dot += style[c] * row[c]; rowNorm += row[c] * row[c]; }
            rowNorm = MathF.Sqrt(rowNorm);
            float sim = rowNorm > 0f && styleNorm > 0f ? dot / (styleNorm * rowNorm) : float.NegativeInfinity;
            if (sim > bestSim) { bestSim = sim; best = i; }
        }
        return best;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        yield return _spkMatrix;
        yield return _emoMatrix;
    }

    private void ThrowIfDisposed()
    {
        if (Volatile.Read(ref _disposed) != 0) throw new ObjectDisposedException(nameof(IndexTts2EmotionVectorLookup));
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _spkMatrix.Dispose();
        _emoMatrix.Dispose();
        GC.SuppressFinalize(this);
    }
}
