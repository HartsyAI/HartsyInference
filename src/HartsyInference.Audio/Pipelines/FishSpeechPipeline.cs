using HartsyInference.Audio.Dsp;
using HartsyInference.Audio.Models.FishSpeech;
using HartsyInference.Audio.Preprocessing;
using HartsyInference.Core.Backends;
using HartsyInference.LLM.Transformer;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Pipelines;

/// <summary>Fish-Speech 1.5 pipeline: the DualAR model AR-generates frames (a semantic token + 8 audio
/// codebooks each) over the text prompt, then the firefly decoder turns the codebook grid into 44.1 kHz audio.
/// Takes pre-tokenized text ids (caller tokenizes); stops at <paramref name="endToken"/> or the frame cap.</summary>
public sealed unsafe class FishSpeechPipeline : IDisposable
{
    private readonly FishSpeechConfig _cfg;
    private readonly FishSpeechDualAr _model;
    private readonly FireflyDecoder _codec;
    private int _disposed;

    public FishSpeechPipeline(FishSpeechConfig cfg)
    {
        _cfg = cfg;
        _model = new FishSpeechDualAr(cfg);
        _codec = new FireflyDecoder(cfg.NumCodebooks, cfg.CodebookSize);
    }

    public int SampleRate => _codec.SampleRate;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> model, IReadOnlyDictionary<string, Tensor> codec)
    {
        _model.LoadWeights(model);
        _codec.LoadWeights(codec);
    }

    /// <summary>Synthesizes audio from pre-tokenized text ids. Mirrors upstream <c>generate</c>: one batched
    /// prefill over the prompt (sampling only at its last position), then the AR loop feeds each sampled frame
    /// (row-0 = semantic vocab id + N codebooks) back in until <paramref name="endToken"/> or the frame cap.
    /// When <paramref name="onFrame"/> is supplied, it's invoked with each genuine (non-end-token) frame right
    /// after it's added — same shape as <c>MoshiTtsGenerator.Generate</c>'s <c>onValidFrame</c> — so a streaming
    /// caller can batch frames into codec decode calls as they're produced. Passing null (the default) changes
    /// nothing about this method's behavior.</summary>
    public float[] Synthesize(IBackend backend, ReadOnlySpan<int> textTokens, int endToken, int maxFrames = 0,
        int seed = 0, bool normalizeLoudness = true, Action<int[]>? onFrame = null)
    {
        ThrowIfDisposed();
        if (textTokens.Length == 0) { Logs.Warning("FishSpeech: empty prompt."); return []; }
        int n = _cfg.NumCodebooks;
        int max = maxFrames > 0 ? maxFrames : _cfg.MaxNewTokens;
        int cap = textTokens.Length + max + 2;
        using IKvCache slow = _model.CreateSlowCache(cap);
        uint rng = DeterministicRng.Seed(seed);

        // Batched prefill (prompt frames: row-0 = text id, codebook rows padded; EmbedPrompt zeroes the codebook
        // contribution for non-semantic rows), sampling the first frame from the last prompt position.
        Tensor promptEmbeds = _model.EmbedPrompt(textTokens);
        Tensor hidden = _model.ForwardHidden(backend, promptEmbeds, textTokens.Length, 0, slow);
        promptEmbeds.Dispose();
        (int sem, int[] codes) = _model.SampleFrame(backend, hidden, ref rng);
        hidden.Dispose();

        // AR loop: previous frame in, next frame out. The end-token frame carries no audio and is dropped.
        List<int[]> frames = new(max);
        List<int[]> window = new(_cfg.RepetitionWindow);   // trailing [semantic, codes...] rows for rep penalty
        int pos = textTokens.Length;
        while (sem != endToken && frames.Count < max && pos < cap - 1)
        {
            frames.Add(codes);
            onFrame?.Invoke(codes);
            int[] row = new int[n + 1];
            row[0] = sem; codes.CopyTo(row, 1);
            if (window.Count == _cfg.RepetitionWindow) window.RemoveAt(0);
            window.Add(row);

            Tensor e = _model.EmbedFrame(sem, codes);
            hidden = _model.ForwardHidden(backend, e, 1, pos++, slow);
            e.Dispose();
            (sem, codes) = _model.SampleFrame(backend, hidden, ref rng, window);
            hidden.Dispose();
        }

        if (frames.Count == 0) { Logs.Warning("FishSpeech: no audio frames generated."); return []; }
        int t = frames.Count;
        int[,] grid = new int[n, t];
        for (int j = 0; j < t; j++)
            for (int i = 0; i < n; i++) grid[i, j] = frames[j][i];
        float[] audio = _codec.Decode(backend, grid, t);
        // Free-running DualAR codes render ~10x quiet vs the reference; peak-normalize as a product step
        // (disabled for parity, which must match the reference's no-gain output). See LoudnessNormalizer.
        return normalizeLoudness ? LoudnessNormalizer.PeakNormalize(audio) : audio;
    }

    /// <summary>Decodes an arbitrary batch of already-sampled frames (as produced by <see cref="Synthesize"/>'s
    /// <c>onFrame</c> callback) through the Firefly codec independently — naive, stateless per-batch decode, no
    /// carried conv-tail state (unlike Mimi). This is exactly Fish-Speech's OWN
    /// real streaming mechanism, confirmed by reading the actual upstream inference engine
    /// (<c>fish_speech/inference_engine/vq_manager.py</c>'s <c>decode_vq_tokens</c> is completely stateless,
    /// decoding only the codes handed to it each call) — upstream chunks at whole-sentence granularity via the
    /// LM's <c>chunk_length</c> text-splitting; the engine-layer streaming wiring (<c>FishSpeechTtsModel</c>)
    /// chunks at the codec's own frame axis instead, needing no AR-loop restart per chunk. Firefly's convs are
    /// causal (verified against the real weights — <c>FishConvNet</c>/<c>FishTransConvNet</c>, left-pad only), so
    /// each chunk's only artifact is a brief transient from zero left-context at its own start, not a
    /// bidirectional-context problem like SNAC/DAC.
    ///
    /// <para>Not loudness-normalized: <see cref="LoudnessNormalizer.PeakNormalize"/> is a whole-utterance peak
    /// scan that can't apply per-chunk without either buffering the whole utterance first (defeating streaming)
    /// or a different running-gain algorithm; callers needing consistent loudness across the streaming and
    /// non-streaming paths should normalize client-side.</para></summary>
    public float[] DecodeChunk(IBackend backend, IReadOnlyList<int[]> frames)
    {
        int n = _cfg.NumCodebooks;
        int t = frames.Count;
        int[,] grid = new int[n, t];
        for (int j = 0; j < t; j++)
            for (int i = 0; i < n; i++) grid[i, j] = frames[j][i];
        return _codec.Decode(backend, grid, t);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _model.EnumerateWeights()) yield return t;
        foreach (Tensor t in _codec.EnumerateWeights()) yield return t;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _model.Dispose(); _codec.Dispose();
        GC.SuppressFinalize(this);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed != 0) throw new ObjectDisposedException(nameof(FishSpeechPipeline));
    }
}
