using HartsyInference.Audio.Dsp;

namespace HartsyInference.Audio.Sampling;

/// <summary>Hugging Face <c>generate()</c> beam search and beam sampling (<c>_beam_search</c> of transformers 4.52,
/// the version IndexTTS pins), for one sequence over an <see cref="IBeamSearchModel"/>. Each step scores every live
/// beam, turns the logits into log-probabilities, applies the processors HF builds for the settings
/// (repetition penalty, <c>min_new_tokens</c>, and when sampling temperature, top-k and top-p, each keeping at least
/// two tokens), adds the beam's running score, then takes the best <c>2 × beams</c> continuations over all beams (or
/// samples them without replacement). Continuations that end (EOS or the length cap) enter the finished set, scored by
/// <c>sum / length^length_penalty</c>; the rest continue. Generation stops when no running beam can beat the worst
/// finished one or every continuation has ended (HF's <c>early_stopping=False</c> rule).</summary>
/// <remarks>Sampling draws come from <see cref="DeterministicRng"/>, so sampled output differs from torch's for the
/// same seed; greedy beam search (<see cref="BeamSearchSettings.DoSample"/> false) is deterministic and matches HF.
/// Scores use HF's -1e9 sentinel in float32, so ties against it resolve as they do there.</remarks>
public static class BeamSearch
{
    private const float Sentinel = -1.0e9f;
    private const int MinTokensToKeep = 2;

    /// <summary>Runs the search and returns the best finished sequence's generated tokens (ending with the EOS token
    /// when it finished on one).</summary>
    /// <param name="promptTokens">The prompt ids the repetition penalty sees in every beam.</param>
    /// <param name="startToken">The token fed at step 0 (the prompt's last token).</param>
    public static int[] Generate(IBeamSearchModel model, BeamSearchSettings settings, ReadOnlySpan<int> promptTokens,
        int startToken, int vocabSize, ref uint rngState)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(settings);
        ArgumentOutOfRangeException.ThrowIfLessThan(settings.NumBeams, 2);
        ArgumentOutOfRangeException.ThrowIfLessThan(settings.MaxNewTokens, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(vocabSize, 2);
        int n = settings.NumBeams, keep = 2 * n, maxNew = settings.MaxNewTokens;

        int[][] running = NewRows(n, maxNew), nextRunning = NewRows(n, maxNew), finished = NewRows(n, maxNew);
        int[] finishedLength = new int[n];
        bool[] finishedDone = new bool[n];
        float[] finishedScore = new float[n];
        Array.Fill(finishedScore, Sentinel);
        float[] runningScore = new float[n];
        for (int b = 1; b < n; b++) runningScore[b] = Sentinel;

        float[] scores = new float[n * vocabSize];
        float[] scratch = new float[vocabSize];
        int[] order = new int[vocabSize];
        HashSet<int> seen = [];
        int[] candidate = new int[keep];
        float[] candidateScore = new float[keep];
        bool[] hits = new bool[keep];
        int[] parents = new int[n], nextTokens = new int[n];
        float[] nextScores = new float[n];
        float[] keys = settings.DoSample ? new float[scores.Length] : [];
        int[] mergeOrder = new int[n + keep];
        float[] mergeScore = new float[n + keep];
        int[][] mergedRows = NewRows(n, maxNew);
        int[] mergedLength = new int[n];
        bool[] mergedDone = new bool[n];
        float[] mergedScore = new float[n];

        for (int gen = 0; ; gen++)
        {
            for (int b = 0; b < n; b++)
            {
                Span<float> row = scores.AsSpan(b * vocabSize, vocabSize);
                // A beam still at the sentinel can never be picked over a live one; HF scores it anyway, but its state
                // (an empty cache at step 0) is not valid to run.
                if (runningScore[b] <= Sentinel)
                {
                    row.Fill(float.NegativeInfinity);
                    continue;
                }
                model.Step(b, gen == 0 ? startToken : running[b][gen - 1], gen, row);
                seen.Clear();
                foreach (int t in promptTokens) seen.Add(t);
                for (int i = 0; i < gen; i++) seen.Add(running[b][i]);
                ProcessLogits(row, settings, gen, seen, scratch, order);
                float add = runningScore[b];
                for (int v = 0; v < vocabSize; v++) row[v] += add;
            }

            if (settings.DoSample) SampleCandidates(scores, keys, candidate, ref rngState);
            else TopCandidates(scores, candidate);
            for (int c = 0; c < keep; c++)
            {
                candidateScore[c] = scores[candidate[c]];
                int token = candidate[c] % vocabSize;
                hits[c] = token == settings.EosToken || gen + 1 >= maxNew;
            }

            // Running beams for the next step: the best non-finished continuations.
            for (int c = 0; c < keep; c++) mergeScore[c] = candidateScore[c] + (hits[c] ? Sentinel : 0f);
            StableTopIndices(mergeScore.AsSpan(0, keep), mergeOrder.AsSpan(0, n));
            for (int b = 0; b < n; b++)
            {
                int c = mergeOrder[b];
                parents[b] = candidate[c] / vocabSize;
                nextTokens[b] = candidate[c] % vocabSize;
                nextScores[b] = mergeScore[c];
            }

            // Finished set: previous entries first, then this step's continuations; only the first `n` candidates
            // that ended may enter.
            float lengthScale = MathF.Pow(gen + 1, settings.LengthPenalty);
            for (int i = 0; i < n; i++) mergeScore[i] = finishedScore[i];
            for (int c = 0; c < keep; c++)
            {
                float s = candidateScore[c] / lengthScale;
                if (!(hits[c] && c < n)) s += Sentinel;
                mergeScore[n + c] = s;
            }
            StableTopIndices(mergeScore.AsSpan(0, n + keep), mergeOrder.AsSpan(0, n));
            for (int i = 0; i < n; i++)
            {
                int m = mergeOrder[i];
                mergedScore[i] = mergeScore[m];
                if (m < n)
                {
                    Array.Copy(finished[m], mergedRows[i], finishedLength[m]);
                    mergedLength[i] = finishedLength[m];
                    mergedDone[i] = finishedDone[m];
                }
                else
                {
                    int c = m - n;
                    Array.Copy(running[candidate[c] / vocabSize], mergedRows[i], gen);
                    mergedRows[i][gen] = candidate[c] % vocabSize;
                    mergedLength[i] = gen + 1;
                    mergedDone[i] = hits[c] && c < n;
                }
            }
            (finished, mergedRows) = (mergedRows, finished);
            Array.Copy(mergedLength, finishedLength, n);
            Array.Copy(mergedDone, finishedDone, n);
            Array.Copy(mergedScore, finishedScore, n);

            for (int b = 0; b < n; b++)
            {
                Array.Copy(running[parents[b]], nextRunning[b], gen);
                nextRunning[b][gen] = nextTokens[b];
                runningScore[b] = nextScores[b];
            }
            (running, nextRunning) = (nextRunning, running);
            model.Reorder(parents);

            int generated = gen + 1;
            float bestPossible = runningScore[0] / MathF.Pow(generated, settings.LengthPenalty);
            float worstFinished = float.PositiveInfinity;
            for (int i = 0; i < n; i++) worstFinished = MathF.Min(worstFinished, finishedScore[i]);
            bool improvementPossible = false;
            for (int i = 0; i < n; i++)
                if (bestPossible > (finishedDone[i] ? worstFinished : Sentinel)) improvementPossible = true;
            bool validContinuation = false;
            for (int c = 0; c < keep; c++) validContinuation |= !hits[c];
            if (!improvementPossible || !validContinuation) break;
        }

        return finished[0][..finishedLength[0]];
    }

    /// <summary>log_softmax, then HF's processors in their list order.</summary>
    internal static void ProcessLogits(Span<float> row, BeamSearchSettings s, int generated, HashSet<int> seen,
        float[] scratch, int[] order)
    {
        LogSoftmax(row);
        if (s.RepetitionPenalty != 1f)
        {
            foreach (int t in seen)
            {
                if ((uint)t >= (uint)row.Length) continue;
                float v = row[t];
                row[t] = v < 0f ? v * s.RepetitionPenalty : v / s.RepetitionPenalty;
            }
        }
        if (generated < s.MinNewTokens && (uint)s.EosToken < (uint)row.Length) row[s.EosToken] = float.NegativeInfinity;
        if (!s.DoSample) return;
        if (s.Temperature != 1f)
            for (int v = 0; v < row.Length; v++) row[v] /= s.Temperature;
        if (s.TopK != 0) TopK(row, Math.Max(s.TopK, MinTokensToKeep), scratch);
        if (s.TopP < 1f) TopP(row, s.TopP, scratch, order);
    }

    private static void LogSoftmax(Span<float> row)
    {
        float max = float.NegativeInfinity;
        foreach (float v in row) max = MathF.Max(max, v);
        double sum = 0;
        foreach (float v in row) sum += MathF.Exp(v - max);
        float lse = max + (float)Math.Log(sum);
        for (int v = 0; v < row.Length; v++) row[v] -= lse;
    }

    // HF TopKLogitsWarper: everything below the k-th largest score is removed (ties with it are kept).
    private static void TopK(Span<float> row, int k, float[] scratch)
    {
        k = Math.Min(k, row.Length);
        Span<float> sorted = scratch.AsSpan(0, row.Length);
        row.CopyTo(sorted);
        sorted.Sort();
        float threshold = sorted[row.Length - k];
        for (int v = 0; v < row.Length; v++)
            if (row[v] < threshold) row[v] = float.NegativeInfinity;
    }

    // HF TopPLogitsWarper: sort ascending, drop the tokens whose cumulative probability stays <= 1 - top_p, but
    // always keep the last MinTokensToKeep.
    private static void TopP(Span<float> row, float topP, float[] scratch, int[] order)
    {
        int n = row.Length;
        Span<float> sorted = scratch.AsSpan(0, n);
        Span<int> idx = order.AsSpan(0, n);
        row.CopyTo(sorted);
        for (int i = 0; i < n; i++) idx[i] = i;
        sorted.Sort(idx);
        float max = sorted[n - 1];
        double total = 0;
        for (int i = 0; i < n; i++) total += MathF.Exp(sorted[i] - max);
        double cumulative = 0;
        for (int i = 0; i < n - MinTokensToKeep; i++)
        {
            cumulative += MathF.Exp(sorted[i] - max) / total;
            if ((float)cumulative <= 1f - topP) row[idx[i]] = float.NegativeInfinity;
        }
    }

    // torch.topk over the flattened scores: the best `candidate.Length`, descending, lower index first on ties.
    private static void TopCandidates(float[] scores, int[] candidate) => StableTopIndices(scores, candidate);

    // torch.multinomial without replacement over softmax(scores), as torch implements it: the top keys of
    // p / Exp(1) noise, which is sequential sampling without replacement. Zero-probability entries fill any
    // remaining places in index order.
    private static void SampleCandidates(float[] scores, float[] keys, int[] candidate, ref uint rngState)
    {
        float max = float.NegativeInfinity;
        foreach (float v in scores) max = MathF.Max(max, v);
        for (int i = 0; i < scores.Length; i++)
        {
            float u = DeterministicRng.NextUniform(ref rngState) + 0.5f / 16777216f;
            float exponential = -MathF.Log(u);
            keys[i] = scores[i] - max - MathF.Log(exponential);
        }
        StableTopIndices(keys, candidate);
    }

    /// <summary>Writes the indices of the <c>top.Length</c> largest values, descending, lower index first on ties.</summary>
    internal static void StableTopIndices(ReadOnlySpan<float> values, Span<int> top)
    {
        int k = top.Length, filled = 0;
        for (int i = 0; i < values.Length; i++)
        {
            float v = values[i];
            if (filled == k && !(v > values[top[k - 1]])) continue;
            int pos = filled < k ? filled++ : k - 1;
            while (pos > 0 && v > values[top[pos - 1]])
            {
                top[pos] = top[pos - 1];
                pos--;
            }
            top[pos] = i;
        }
        if (filled < k) throw new ArgumentException("Fewer values than requested indices.", nameof(values));
    }

    private static int[][] NewRows(int count, int length)
    {
        int[][] rows = new int[count][];
        for (int i = 0; i < count; i++) rows[i] = new int[length];
        return rows;
    }
}
