using System.Diagnostics;
using HartsyInference.Audio.Dsp;
using HartsyInference.Audio.Models.Codecs.Snac;
using HartsyInference.Audio.Models.LanguageModels.Qwen2;
using HartsyInference.Audio.Models.Orpheus;
using HartsyInference.Audio.Sampling;
using HartsyInference.LLM.Transformer;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Pipelines;
using HartsyInference.Core.Tensors;
using SnacModel = HartsyInference.Audio.Models.Codecs.Snac.Snac;

namespace HartsyInference.Audio.Pipelines;

/// <summary>Orpheus TTS pipeline: a Llama-3.2-3B causal LM (<see cref="Qwen2Model"/>) autoregressively emits
/// a flat stream of SNAC audio tokens, which are redistributed into 3 hierarchical codebooks and decoded by
/// the built <see cref="SnacModel">SNAC</see> 24 kHz codec. Takes pre-tokenized text ids (the Audio package
/// carries no text-BPE dependency); the pipeline adds the model's human-frame control tokens. Reuses
/// <see cref="NucleusSampler"/> with repetition penalty applied by pre-shaping the logit buffer.</summary>
public sealed unsafe class OrpheusPipeline : IDisposable
{
    private readonly OrpheusConfig _cfg;
    private readonly Qwen2Model _backbone;
    private readonly SnacModel _codec;
    private int _disposed;

    public OrpheusPipeline(OrpheusConfig cfg)
    {
        _cfg = cfg;
        _backbone = new Qwen2Model(cfg.Llm);
        _codec = new SnacModel(cfg.Codec);
    }

    /// <summary>Sample rate of the emitted audio (24 kHz for SNAC 24 kHz).</summary>
    public int SampleRate => _cfg.Codec.SampleRate;

    /// <summary>Loads the Llama backbone (HF <c>LlamaForCausalLM</c> layout — <c>model.*</c> + tied head)
    /// and the SNAC codec weights.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> backbone, IReadOnlyDictionary<string, Tensor> snac,
        string backbonePrefix = "model")
    {
        _backbone.LoadWeights(backbone, backbonePrefix);
        _codec.LoadWeights(snac);
    }

    /// <summary>Synthesizes 24 kHz mono PCM from pre-tokenized text ids (e.g. the Llama BPE of
    /// <c>"{voice}: {text}"</c>). The pipeline wraps them in the <c>[StartOfHuman] … [EndOfText, EndOfHuman]</c>
    /// frame, AR-generates audio tokens until <see cref="OrpheusConfig.EndOfSpeech"/>, then SNAC-decodes.</summary>
    public float[] Synthesize(IBackend backend, ReadOnlySpan<int> textTokenIds, int maxTokens = 1200,
        int seed = 0, Action<GenerationProgress>? progress = null)
    {
        ThrowIfDisposed();
        Stopwatch sw = Stopwatch.StartNew();
        List<int> generated = GenerateTokens(backend, BuildPrompt(textTokenIds), maxTokens, seed, progress);

        int[] codes = OrpheusCodeFrames.ExtractAudioCodes(generated, _cfg.CodeStart, _cfg.EndOfSpeech, _cfg.TokensPerFrame);
        int groups = codes.Length / _cfg.TokensPerFrame;
        if (groups == 0)
        {
            Logs.Warning("Orpheus: no audio codes generated.");
            return [];
        }

        (int[] s1, int[] s2, int[] s3) = OrpheusCodeFrames.Redistribute(codes, _cfg.AudioCodeBase, _cfg.CodebookSize);
        Tensor l1 = ToCodeTensor(s1);
        Tensor l2 = ToCodeTensor(s2);
        Tensor l3 = ToCodeTensor(s3);
        Tensor audioT = _codec.Decode(backend, [l1, l2, l3], batch: 1);
        l1.Dispose(); l2.Dispose(); l3.Dispose();

        int n = (int)audioT.Shape[audioT.Shape.Rank - 1];
        float[] audio = new float[n];
        Buffer.MemoryCopy((void*)audioT.DataPointer,
            System.Runtime.CompilerServices.Unsafe.AsPointer(ref audio[0]), n * 4, n * 4);
        audioT.Dispose();
        sw.Stop();
        Logs.Info($"Orpheus: {groups} SNAC frames → {audio.Length} samples ({audio.Length / (double)SampleRate:F1}s) in {sw.ElapsedMilliseconds}ms.");
        return audio;
    }

    /// <summary>Streaming counterpart to <see cref="Synthesize"/>: yields 24 kHz PCM chunks as generation
    /// progresses, using the same windowed-decode-and-discard-the-edges recipe as the official Orpheus reference
    /// client (<c>orpheus_tts/decoder.py</c>'s <c>tokens_decoder</c>/<c>convert_to_audio</c>) — SNAC's decoder
    /// uses symmetric ("same") padding throughout (verified against the real weights: no causal-conv variant
    /// exists for this architecture), so unlike Mimi's exact state-carrying reconstruction, this is a genuine
    /// approximation of the monolithic decode, not a bit-identical one; that's the same tradeoff upstream's own
    /// client makes, not a shortcut unique to this port.
    ///
    /// <para>Windowing: every 4 consecutive 7-token groups (28 tokens = 16 native-rate SNAC latent frames) are
    /// decoded together and only the SECOND group's span is kept (1 group of real left context, 2 of real right
    /// context/look-ahead, discarding the window's own edges) — this is upstream's own empirically-tuned choice,
    /// confirmed against the reference source, not derived here. <see cref="SnacDecoder.Forward"/>'s
    /// <c>callSeed</c> parameter keys each window's noise draw so overlapping windows don't replay identical
    /// noise (a bug upstream's fresh-`torch.randn`-per-call design doesn't have, but this port's fixed-seed RNG
    /// would without the fix).</para>
    ///
    /// <para>Unlike the reference client, this does NOT drop the first ~85 ms or last ≤170 ms of audio: the
    /// standard window needs a group before and two after the one it keeps, so group 0 and the last two groups
    /// structurally fall outside any window's kept span — handled here by decoding them as their own small,
    /// context-limited (not upstream-windowed) chunks instead of silently discarding them.</para></summary>
    public IEnumerable<float[]> SynthesizeStreamChunks(IBackend backend, ReadOnlySpan<int> textTokenIds,
        int maxTokens = 1200, int seed = 0, Action<GenerationProgress>? progress = null)
    {
        // ReadOnlySpan<int> (a ref struct) can't be a parameter of an iterator method — build the prompt array
        // here, outside the iterator, and hand the plain int[] to the actual iterator below.
        ThrowIfDisposed();
        return SynthesizeStreamChunksCore(backend, BuildPrompt(textTokenIds), maxTokens, seed, progress);
    }

    private IEnumerable<float[]> SynthesizeStreamChunksCore(IBackend backend, int[] prompt,
        int maxTokens, int seed, Action<GenerationProgress>? progress)
    {
        int groupTokens = _cfg.TokensPerFrame; // 7: 1 L1 + 2 L2 + 4 L3 codes per SNAC frame.
        const int groupsPerWindow = 4;  // Upstream's window size.
        const int keepGroupInWindow = 1; // Which of the 4 groups (0-indexed) the window's kept slice is.

        List<int> allTokens = new(maxTokens);
        List<int[]> pendingGroups = new(groupsPerWindow); // last <=4 completed groups' raw 7-token slices.
        int emittedThroughGroup = 0; // exclusive upper bound of groups already emitted, absolute index.
        int windowCallSeed = 0;

        foreach (int token in GenerateTokensStream(backend, prompt, maxTokens, seed, progress))
        {
            // CodeStart (start-of-speech) is inside the sampler-eligible range, so the model can legally re-emit
            // it mid-generation as a restart marker — Synthesize's ExtractAudioCodes handles this by keeping only
            // tokens after the LAST occurrence, which a batch decode can do because it sees the whole sequence
            // first. A true incremental stream can't retroactively un-send chunks already emitted before the
            // restart — that's an inherent property of streaming, not a bug — so this resets buffering state and
            // starts a fresh utterance from here; anything already yielded stays yielded.
            if (token == _cfg.CodeStart)
            {
                allTokens.Clear();
                pendingGroups.Clear();
                emittedThroughGroup = 0;
                continue;
            }
            allTokens.Add(token);
            if (allTokens.Count % groupTokens != 0) continue;

            int groupStart = allTokens.Count - groupTokens;
            int[] group = allTokens.GetRange(groupStart, groupTokens).ToArray();
            pendingGroups.Add(group);
            int totalGroups = allTokens.Count / groupTokens;

            if (totalGroups == 1)
            {
                // Head chunk: no left/right context exists yet for the standard window, so decode group 0 alone
                // (a legitimate 1-group monolithic-style decode — exactly what Synthesize does for any input,
                // just applied to a slice) and emit ALL of it rather than waiting or discarding.
                float[]? head = DecodeGroups(backend, [group], groupTokens, keepStart: null, keepLen: null, windowCallSeed++);
                if (head is not null) { yield return head; emittedThroughGroup = 1; }
                continue;
            }

            if (totalGroups < groupsPerWindow) continue;
            if (pendingGroups.Count > groupsPerWindow) pendingGroups.RemoveAt(0);
            if (pendingGroups.Count < groupsPerWindow) continue;

            // Window is groups [totalGroups-4, totalGroups); kept group is (totalGroups-4)+1 = totalGroups-3.
            int keptGroupIndex = totalGroups - groupsPerWindow + keepGroupInWindow;
            if (keptGroupIndex < emittedThroughGroup) continue; // already covered (shouldn't happen, defensive).
            float[]? chunk = DecodeGroups(backend, pendingGroups, groupTokens, keepStart: keepGroupInWindow, keepLen: 1, windowCallSeed++);
            if (chunk is not null) { yield return chunk; emittedThroughGroup = keptGroupIndex + 1; }
        }

        int finalTotalGroups = allTokens.Count / groupTokens;
        if (finalTotalGroups > emittedThroughGroup)
        {
            // Tail flush: whatever groups the sliding window structurally never covers (the last 1-2 groups, or
            // everything if the utterance was too short to ever fill a window) — decode them as their own
            // context-limited chunk and emit all of it, rather than silently dropping trailing audio the way the
            // reference client does.
            int tailStart = emittedThroughGroup * groupTokens;
            int tailGroupCount = finalTotalGroups - emittedThroughGroup;
            List<int[]> tailGroups = new(tailGroupCount);
            for (int g = 0; g < tailGroupCount; g++)
                tailGroups.Add(allTokens.GetRange(tailStart + g * groupTokens, groupTokens).ToArray());
            float[]? tail = DecodeGroups(backend, tailGroups, groupTokens, keepStart: null, keepLen: null, windowCallSeed++);
            if (tail is not null) yield return tail;
        }
    }

    /// <summary>Decodes a window of consecutive 7-token groups and returns either the whole window's PCM
    /// (<paramref name="keepStart"/>/<paramref name="keepLen"/> both null — the head/tail edge-chunk case) or
    /// just <paramref name="keepLen"/> groups' worth starting at group index <paramref name="keepStart"/> within
    /// the window (the standard sliding-window kept-middle case). A separate instance method, not a local
    /// function inside <see cref="SynthesizeStreamChunksCore"/> — C# disallows unsafe/pointer code inside an
    /// iterator method's lexical scope even via a local function marked <c>unsafe</c>.</summary>
    private unsafe float[]? DecodeGroups(IBackend backend, IReadOnlyList<int[]> groups, int groupTokens,
        int? keepStart, int? keepLen, int callSeed)
    {
        int[] flat = new int[groups.Count * groupTokens];
        for (int g = 0; g < groups.Count; g++) groups[g].CopyTo(flat, g * groupTokens);
        (int[] s1, int[] s2, int[] s3) = OrpheusCodeFrames.Redistribute(flat, _cfg.AudioCodeBase, _cfg.CodebookSize);
        Tensor l1 = ToCodeTensor(s1);
        Tensor l2 = ToCodeTensor(s2);
        Tensor l3 = ToCodeTensor(s3);
        Tensor audioT = _codec.Decode(backend, [l1, l2, l3], batch: 1, callSeed);
        l1.Dispose(); l2.Dispose(); l3.Dispose();

        int n = (int)audioT.Shape[audioT.Shape.Rank - 1];
        int samplesPerGroup = n / groups.Count;
        int start = keepStart is int ks ? ks * samplesPerGroup : 0;
        int len = keepLen is int kl ? kl * samplesPerGroup : n;
        if (len <= 0) { audioT.Dispose(); return null; }
        float[] outp = new float[len];
        Buffer.MemoryCopy((float*)audioT.DataPointer + start, System.Runtime.CompilerServices.Unsafe.AsPointer(ref outp[0]),
            len * 4L, len * 4L);
        audioT.Dispose();
        return outp;
    }

    /// <summary>Orpheus prompt frame (matches the reference orpheus_tts._format_prompt): the caller's text ids
    /// already carry the BOS token; wraps them as
    /// <c>[StartOfHuman] textTokens [EndOfText, EndOfHuman, StartOfAi, StartOfSpeech]</c>. The trailing
    /// StartOfAi + StartOfSpeech are what tell the model to begin emitting the audio-code stream — without them
    /// the model produces unconditioned/gibberish speech (verified vs the transformers reference).</summary>
    private int[] BuildPrompt(ReadOnlySpan<int> textTokenIds)
    {
        int[] prompt = new int[textTokenIds.Length + 5];
        prompt[0] = _cfg.StartOfHuman;
        textTokenIds.CopyTo(prompt.AsSpan(1));
        prompt[^4] = _cfg.EndOfText;
        prompt[^3] = _cfg.EndOfHuman;
        prompt[^2] = _cfg.StartOfAi;
        prompt[^1] = _cfg.CodeStart;   // start-of-speech
        return prompt;
    }

    private List<int> GenerateTokens(IBackend backend, int[] prompt, int maxTokens, int seed, Action<GenerationProgress>? progress)
    {
        List<int> generated = new(maxTokens);
        foreach (int token in GenerateTokensStream(backend, prompt, maxTokens, seed, progress)) generated.Add(token);
        return generated;
    }

    /// <summary>The AR token loop itself, as an iterator — <see cref="GenerateTokens"/> (used by the non-streaming
    /// <see cref="Synthesize"/>) just drains this into a list unchanged; <see cref="SynthesizeStreamChunks"/>
    /// consumes it directly so it can react to each token as it's produced.</summary>
    private IEnumerable<int> GenerateTokensStream(IBackend backend, int[] prompt, int maxTokens, int seed, Action<GenerationProgress>? progress)
    {
        Stopwatch sw = Stopwatch.StartNew();
        int cacheCap = Math.Min(_cfg.Llm.MaxPositionEmbeddings, prompt.Length + maxTokens + 8);
        using IKvCache cache = _backbone.CreateDecodeCache(cacheCap);

        uint rng = DeterministicRng.Seed(seed);
        int vocab = _cfg.Llm.VocabSize;
        float penalty = _cfg.RepetitionPenalty;
        HashSet<int> seen = new();

        // The only tokens Orpheus validly emits are EndOfSpeech (128258) and the SNAC audio codes
        // (AudioCodeBase 128266 …), all ≥ CodeStart. Restricting the sampler to [CodeStart, vocab) skips the
        // ~128k orthographic-text logits (a small cut on the per-token host softmax+argsort). No quality change:
        // the excluded ids are never part of a valid speech continuation.
        // NOTE (perf pass 2026-07-14): the decode is GPU-compute-bound on the F32 backbone (~245 ms/token on a
        // 3060), not host- or launch-bound — sampler restriction and a CUDA-graph decode both measured flat. The
        // real lever is an F16 compute path in the shared GenericTransformer (benefits all LLM-backed audio
        // models); scoped as a separate cross-cutting project. See docs/Checklists/AUDIO_TTS_BRINGUP_PLAN.md.
        int sampleStart = _cfg.CodeStart;
        int sampleCount = vocab - sampleStart;

        bool prof = Environment.GetEnvironmentVariable("HARTSY_ORPHEUS_PROF") == "1";
        double tLogits = 0, tSample = 0, tFwd = 0; int profSteps = 0;
        Stopwatch psw = new();

        Tensor hidden = _backbone.Forward(backend, prompt, batch: 1, posStart: 0, cache);
        for (int step = 0; step < maxTokens; step++)
        {
            Tensor last = SliceLastFrame(hidden, _cfg.Llm.HiddenSize);
            hidden.Dispose();
            if (prof) { backend.Sync(); psw.Restart(); }
            Tensor logitsT = _backbone.ProjectLogits(backend, last, batch: 1, t: 1);
            if (prof) { backend.Sync(); tLogits += psw.Elapsed.TotalMilliseconds; psw.Restart(); }
            last.Dispose();

            // Pointer construction pulled into a helper: C# disallows unsafe code directly inside an iterator
            // method's body (this method has `yield return` below) even though the class itself is `unsafe`.
            Span<float> logits = AsFloatSpan(logitsT, vocab);
            if (penalty != 1f)
                foreach (int tok in seen)
                    logits[tok] = logits[tok] > 0 ? logits[tok] / penalty : logits[tok] * penalty;

            int next = sampleStart + NucleusSampler.Draw(logits.Slice(sampleStart, sampleCount), sampleCount,
                _cfg.Temperature, _cfg.TopK, _cfg.TopP, ref rng);
            logitsT.Dispose();
            if (prof) { tSample += psw.Elapsed.TotalMilliseconds; }

            if (next == _cfg.EndOfSpeech) break;
            seen.Add(next);
            if (progress != null && (step & 63) == 0) progress(new GenerationProgress(step, maxTokens, sw.Elapsed.TotalMilliseconds));
            yield return next;

            if (prof) { backend.Sync(); psw.Restart(); }
            hidden = _backbone.Forward(backend, [next], batch: 1, posStart: cache.CurrentLength, cache);
            if (prof) { backend.Sync(); tFwd += psw.Elapsed.TotalMilliseconds; profSteps++; }
            if (cache.CurrentLength >= cacheCap - 2) break;
        }
        if (prof && profSteps > 0)
            Logs.Info($"[Orpheus prof] {profSteps} steps: backbone-fwd {tFwd / profSteps:0.0}ms/tok, lm_head {tLogits / profSteps:0.0}ms/tok, sample {tSample / profSteps:0.0}ms/tok");
    }

    private static Tensor ToCodeTensor(int[] codes)
    {
        Tensor t = new(new TensorShape(1, codes.Length), DType.I32);
        int* p = (int*)t.DataPointer;
        for (int i = 0; i < codes.Length; i++) p[i] = codes[i];
        return t;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _backbone.EnumerateWeights()) yield return t;
        foreach (Tensor t in _codec.EnumerateWeights()) yield return t;
    }

    private static Tensor SliceLastFrame(Tensor hidden, int h)
    {
        int t = (int)hidden.Shape[1];
        Tensor last = new(new TensorShape(1, 1, h), DType.F32);
        float* sp = (float*)hidden.DataPointer + (long)(t - 1) * h;
        Buffer.MemoryCopy(sp, (void*)last.DataPointer, h * 4, h * 4);
        return last;
    }

    /// <summary>Wraps a tensor's data pointer as a <see cref="Span{Single}"/>. Pulled out into its own (non-
    /// iterator) method for the same reason as <see cref="SliceLastFrame"/> — <see cref="GenerateTokensStream"/>
    /// needs this but can't contain the raw pointer cast itself.</summary>
    private static Span<float> AsFloatSpan(Tensor t, int length) => new((void*)t.DataPointer, length);

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _backbone.Dispose();
        GC.SuppressFinalize(this);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed != 0) throw new ObjectDisposedException(nameof(OrpheusPipeline));
    }
}
