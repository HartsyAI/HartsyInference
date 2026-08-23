using HartsyInference.Audio.Dsp;
using HartsyInference.Audio.Models.LanguageModels.Gpt;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Audio.Sampling;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.Audio.Models.Bark;

/// <summary>A causal Bark stage (semantic or coarse) — a single token embedding table + the shared
/// <see cref="GptBackbone"/> + an output head, generating tokens autoregressively with the shared
/// <see cref="NucleusSampler"/>. Decodes incrementally: one cache-capturing prefill, then KV-cached
/// single-token steps against a device-resident <see cref="IKvCache"/>.</summary>
public sealed unsafe class BarkCausalStage : IDisposable
{
    private readonly GptConfig _gpt;
    private readonly int _inVocab;
    private readonly int _outVocab;
    private readonly GptBackbone _backbone;
    private int _disposed;

    private Tensor? _inputEmbed;   // [inVocab, hidden]
    private Tensor? _lmHead;       // [outVocab, hidden]

    public BarkCausalStage(GptConfig gpt, int inVocab, int outVocab)
    {
        _gpt = gpt;
        _inVocab = inVocab;
        _outVocab = outVocab;
        _backbone = new GptBackbone(gpt);
    }

    /// <summary>Loads under the HF Bark stage prefix (e.g. <c>semantic</c> / <c>coarse_acoustics</c>):
    /// <c>{p}.input_embeds_layer.weight</c>, <c>{p}.position_embeds_layer.weight</c>, <c>{p}.layers.{i}.*</c>,
    /// <c>{p}.layernorm_final.*</c>, <c>{p}.lm_head.weight</c>.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        string p = prefix.Length == 0 ? "" : prefix + ".";
        _inputEmbed = WhisperOps.EnsureF32(w[$"{p}input_embeds_layer.weight"]);
        _backbone.LoadWeights(w, $"{p}position_embeds_layer.weight", $"{p}layers",
            $"{p}layernorm_final.weight", $"{p}layernorm_final.bias");
        _lmHead = WhisperOps.EnsureF32(w[$"{p}lm_head.weight"]);
    }

    /// <summary>Autoregressively generates up to <paramref name="maxTokens"/> tokens after the prompt,
    /// stopping at <paramref name="eosToken"/>. Returns the generated token IDs (EOS excluded).
    /// One cache-capturing prefill over the prompt, then O(T) single-token steps.</summary>
    public List<int> Generate(IBackend backend, IReadOnlyList<int> promptTokenIds, int maxTokens,
        float temperature, int topK, float topP, int eosToken, int seed)
    {
        if (_inputEmbed is null) throw new InvalidOperationException("BarkCausalStage weights not loaded.");
        int h = _gpt.Hidden;
        List<int> seq = new(promptTokenIds);
        List<int> generated = new(Math.Min(maxTokens, 256));
        uint rng = DeterministicRng.Seed(seed);

        using IKvCache cache = _backbone.CreateCache();
        int t0 = Math.Min(seq.Count, _gpt.BlockSize);
        Tensor embeds = EmbedTail(seq, t0, h);
        Tensor hidden = _backbone.Forward(backend, embeds, nonCausal: false, cache);
        embeds.Dispose();
        Tensor last = SliceLast(hidden, h);
        hidden.Dispose();

        for (int step = 0; step < maxTokens; step++)
        {
            Tensor logits = WhisperOps.ProjectLinear(backend, last, _lmHead!, bias: null, 1, 1, h, _outVocab);
            last.Dispose();
            int next = NucleusSampler.Draw(new Span<float>((void*)logits.DataPointer, _outVocab),
                _outVocab, temperature, topK, topP, ref rng);
            logits.Dispose();
            if (next == eosToken) break;
            generated.Add(next);
            seq.Add(next);
            if (seq.Count >= _gpt.BlockSize) break;

            Tensor stepEmbed = EmbedTail(seq, 1, h);
            last = _backbone.ForwardStep(backend, stepEmbed, cache);
            stepEmbed.Dispose();
        }
        last.Dispose(); // no-op when the loop already consumed it (EOS/block-size break)
        return generated;
    }

    /// <summary>Faithful port of upstream <c>generate_text_semantic</c>: text tokens are truncated/padded to
    /// 256 (<see cref="BarkConfig.TextPadToken"/>), the (absent) voice history is 256 pad tokens, and the two
    /// 256-token halves are merged by SUMMING their embeddings into one 256-position context
    /// (<c>merge_context=True</c>) before the infer token — so the prefill is 257 positions, exactly the layout
    /// the checkpoint was trained on. Sampling is restricted to the 10 000 real semantic ids plus the EOS slot,
    /// with upstream's <c>min_eos_p</c> early stop. Returns the generated semantic ids (all &lt; 10 000).</summary>
    public List<int> GenerateSemantic(IBackend backend, IReadOnlyList<int> textTokens, BarkConfig bark,
        int seed, int maxTokens = 768)
    {
        if (_inputEmbed is null) throw new InvalidOperationException("BarkCausalStage weights not loaded.");
        int h = _gpt.Hidden;
        int semVocab = bark.SemanticPadToken;                 // 10 000 = SEMANTIC_VOCAB_SIZE; index 10 000 = EOS slot
        uint rng = DeterministicRng.Seed(seed);

        // Merged 257-token prefill: emb(text[i]) + emb(history[i]) for 256 positions, then emb(infer).
        Tensor prefill = new(new TensorShape(1, 257, h), DType.F32);
        float* pp = (float*)prefill.DataPointer;
        float* tab = (float*)_inputEmbed.DataPointer;
        for (int i = 0; i < 256; i++)
        {
            int textId = i < textTokens.Count && i < 256 ? textTokens[i] : bark.TextPadToken;
            if ((uint)textId >= (uint)_inVocab) textId = 0;
            float* tRow = tab + (long)textId * h;
            float* hRow = tab + (long)bark.SemanticPadToken * h;
            long dst = (long)i * h;
            for (int c = 0; c < h; c++) pp[dst + c] = tRow[c] + hRow[c];
        }
        Buffer.MemoryCopy(tab + (long)bark.SemanticInferToken * h, pp + 256L * h, h * 4, h * 4);

        using IKvCache cache = _backbone.CreateCache();
        Tensor hidden = _backbone.Forward(backend, prefill, nonCausal: false, cache);
        prefill.Dispose();
        Tensor last = SliceLast(hidden, h);
        hidden.Dispose();

        List<int> generated = new(256);
        for (int n = 0; n < maxTokens; n++)
        {
            Tensor logits = WhisperOps.ProjectLinear(backend, last, _lmHead!, bias: null, 1, 1, h, _outVocab);
            last.Dispose();
            // Sampling domain = [0, 10 000) real ids + the contiguous EOS slot at 10 000.
            Span<float> slice = new((void*)logits.DataPointer, semVocab + 1);
            int next = SampleTopPWithProb(slice, bark.SemanticTemperature, bark.TopP, ref rng, semVocab, out float eosProb);
            logits.Dispose();
            if (next == semVocab || eosProb >= bark.MinEosP)
            {
                break;
            }
            generated.Add(next);
            if (n == maxTokens - 1 || cache.CurrentLength >= _gpt.BlockSize)
            {
                break;
            }
            Tensor stepEmbed = EmbedSingle(next, h);
            last = _backbone.ForwardStep(backend, stepEmbed, cache);
            stepEmbed.Dispose();
        }
        last.Dispose(); // no-op when already consumed by the final iteration
        return generated;
    }

    /// <summary>Faithful port of upstream <c>generate_coarse</c>: the number of coarse steps is derived from
    /// the semantic length (75 Hz / 49.9 Hz × 2 codebooks ≈ 3.01 coarse tokens per semantic token), and
    /// generation runs in 60-step sliding windows — each window re-prefills a fresh context of
    /// [≤256 semantic tokens (right-padded), COARSE_INFER, ≤630 coarse history] so the sequence never crosses
    /// the 1024 block size. Every step samples ONLY from the 1024-entry codebook range that alternates
    /// per step (book 0 / book 1). Returns <c>[NumCoarseCodebooks, T]</c> codes in <c>[0, CodebookSize)</c>.</summary>
    public int[,] GenerateCoarse(IBackend backend, IReadOnlyList<int> semantic, BarkConfig bark, int seed,
        int slidingWindowLen = 60)
    {
        if (_inputEmbed is null) throw new InvalidOperationException("BarkCausalStage weights not loaded.");
        int h = _gpt.Hidden;
        int semVocab = 10_000;
        double ratio = 75.0 / 49.9 * bark.NumCoarseCodebooks;      // semantic_to_coarse_ratio
        int maxSemHistory = (int)Math.Floor(bark.MaxCoarseHistory / ratio);
        int nSteps = (int)(Math.Floor(semantic.Count * ratio / bark.NumCoarseCodebooks) * bark.NumCoarseCodebooks);
        uint rng = DeterministicRng.Seed(seed);

        List<int> coarse = new(nSteps);   // model-space ids (already offset by semVocab + book*CodebookSize)
        int nStep = 0;
        // One device-resident cache reused across windows: each window re-prefills a fresh context (≤256
        // semantic + COARSE_INFER + ≤630 coarse history + ≤60 steps < block size), so Reset() clears it rather
        // than reallocating the per-layer GPU buffers every window.
        using IKvCache cache = _backbone.CreateCache();
        while (nStep < nSteps)
        {
            cache.Reset();
            // Rebuild the window context: semantic slice around the current position, right-padded to 256.
            int semIdx = (int)Math.Round(nStep / ratio);
            int s0 = Math.Max(0, semIdx - maxSemHistory);
            List<int> prompt = new(256 + 1 + bark.MaxCoarseHistory);
            for (int i = 0; i < 256; i++)
            {
                prompt.Add(s0 + i < semantic.Count ? semantic[s0 + i] : bark.CoarseSemanticPadToken);
            }
            prompt.Add(bark.CoarseInferToken);
            int histStart = Math.Max(0, coarse.Count - bark.MaxCoarseHistory);
            for (int i = histStart; i < coarse.Count; i++)
            {
                prompt.Add(coarse[i]);
            }

            Tensor embeds = EmbedTail(prompt, prompt.Count, h);
            Tensor hidden = _backbone.Forward(backend, embeds, nonCausal: false, cache);
            embeds.Dispose();
            Tensor last = SliceLast(hidden, h);
            hidden.Dispose();

            for (int k = 0; k < slidingWindowLen && nStep < nSteps; k++)
            {
                bool major = nStep % bark.NumCoarseCodebooks == 0;
                int logitStart = semVocab + (major ? 0 : bark.CodebookSize);
                Tensor logits = WhisperOps.ProjectLinear(backend, last, _lmHead!, bias: null, 1, 1, h, _outVocab);
                last.Dispose();
                Span<float> slice = new((float*)logits.DataPointer + logitStart, bark.CodebookSize);
                int tok = NucleusSampler.Draw(slice, bark.CodebookSize, bark.CoarseTemperature, bark.TopK, bark.TopP, ref rng)
                    + logitStart;
                logits.Dispose();
                coarse.Add(tok);
                nStep++;
                if (k < slidingWindowLen - 1 && nStep < nSteps)
                {
                    Tensor stepEmbed = EmbedSingle(tok, h);
                    last = _backbone.ForwardStep(backend, stepEmbed, cache);
                    stepEmbed.Dispose();
                }
            }
            last.Dispose(); // no-op when the inner loop consumed it
        }

        // De-interleave [b0, b1, b0, b1, …] → [2, T] and strip the semantic + per-book offsets.
        int t = coarse.Count / bark.NumCoarseCodebooks;
        int[,] result = new int[bark.NumCoarseCodebooks, t];
        for (int j = 0; j < t; j++)
        {
            for (int cb = 0; cb < bark.NumCoarseCodebooks; cb++)
            {
                int v = coarse[j * bark.NumCoarseCodebooks + cb] - semVocab - cb * bark.CodebookSize;
                result[cb, j] = Math.Clamp(v, 0, bark.CodebookSize - 1);
            }
        }
        return result;
    }

    /// <summary>Upstream semantic sampling: top-p filter on the raw logits (cumulative softmax order), THEN
    /// softmax over <c>filtered / temperature</c>, multinomial draw. Also reports the final probability of
    /// <paramref name="eosSlot"/> for the <c>min_eos_p</c> early stop.</summary>
    private static int SampleTopPWithProb(Span<float> logits, float temperature, float topP, ref uint rng,
        int eosSlot, out float eosProb)
    {
        int n = logits.Length;
        float[] work = logits.ToArray();
        if (topP > 0f && topP < 1f)
        {
            int[] order = new int[n];
            for (int i = 0; i < n; i++) order[i] = i;
            float[] w = work;
            Array.Sort(order, (a, b) => w[b].CompareTo(w[a]));
            // softmax over sorted logits for the cumulative cut (upstream does this pre-temperature)
            double sum = 0;
            float max = w[order[0]];
            double[] probs = new double[n];
            for (int r = 0; r < n; r++) { probs[r] = Math.Exp(w[order[r]] - max); sum += probs[r]; }
            double cum = 0;
            for (int r = 0; r < n; r++)
            {
                cum += probs[r] / sum;
                if (cum > topP && r > 0)
                {
                    for (int rr = r; rr < n; rr++) work[order[rr]] = float.NegativeInfinity;
                    break;
                }
            }
        }
        float temp = temperature > 0 ? temperature : 1f;
        float mx = float.NegativeInfinity;
        for (int i = 0; i < n; i++) { work[i] /= temp; if (work[i] > mx) mx = work[i]; }
        double total = 0;
        for (int i = 0; i < n; i++) { work[i] = MathF.Exp(work[i] - mx); total += work[i]; }
        float inv = (float)(1.0 / total);
        for (int i = 0; i < n; i++) work[i] *= inv;
        eosProb = (uint)eosSlot < (uint)n ? work[eosSlot] : 0f;
        float r2 = DeterministicRng.NextUniform(ref rng);
        float acc = 0f;
        for (int i = 0; i < n; i++)
        {
            acc += work[i];
            if (r2 <= acc) return i;
        }
        return n - 1;
    }

    private Tensor EmbedSingle(int id, int h)
    {
        if ((uint)id >= (uint)_inVocab) id = 0;
        Tensor t = new(new TensorShape(1, 1, h), DType.F32);
        Buffer.MemoryCopy((float*)_inputEmbed!.DataPointer + (long)id * h, (void*)t.DataPointer, h * 4, h * 4);
        return t;
    }

    /// <summary>Parity-debug only: teacher-forced logits for the given token ids. Returns <c>[1, T, outVocab]</c>
    /// (causal full-sequence forward + lm_head over every position). Not used in production.</summary>
    /// <remarks>Kept unwired as the C# half of <c>tests/python-reference/bark_reference/dump_bark_reference.py</c>, which names this method as its counterpart.</remarks>
    public Tensor ForwardLogits(IBackend backend, IReadOnlyList<int> tokenIds)
    {
        if (_inputEmbed is null) throw new InvalidOperationException("BarkCausalStage weights not loaded.");
        int h = _gpt.Hidden;
        int t = tokenIds.Count;
        List<int> seq = new(tokenIds);
        Tensor embeds = EmbedTail(seq, t, h);
        Tensor hidden = _backbone.Forward(backend, embeds, nonCausal: false);
        embeds.Dispose();
        Tensor logits = WhisperOps.ProjectLinear(backend, hidden, _lmHead!, bias: null, 1, t, h, _outVocab);
        hidden.Dispose();
        return logits;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_inputEmbed is not null) yield return _inputEmbed;
        foreach (Tensor t in _backbone.EnumerateWeights()) yield return t;
        if (_lmHead is not null) yield return _lmHead;
    }

    private Tensor EmbedTail(List<int> seq, int t, int h)
    {
        int start = seq.Count - t;
        Tensor embeds = new(new TensorShape(1, t, h), DType.F32);
        float* ep = (float*)embeds.DataPointer;
        float* tab = (float*)_inputEmbed!.DataPointer;
        for (int s = 0; s < t; s++)
        {
            int id = seq[start + s];
            if ((uint)id >= (uint)_inVocab) id = 0;
            Buffer.MemoryCopy(tab + (long)id * h, ep + (long)s * h, h * 4, h * 4);
        }
        return embeds;
    }

    private static Tensor SliceLast(Tensor hidden, int h)
    {
        int t = (int)hidden.Shape[1];
        Tensor last = new(new TensorShape(1, 1, h), DType.F32);
        Buffer.MemoryCopy((float*)hidden.DataPointer + (long)(t - 1) * h, (void*)last.DataPointer, h * 4, h * 4);
        return last;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _backbone.Dispose();
        GC.SuppressFinalize(this);
    }
}
