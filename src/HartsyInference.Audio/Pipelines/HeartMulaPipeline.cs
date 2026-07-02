using HartsyInference.Audio.Dsp;
using HartsyInference.Audio.Models.Csm;
using HartsyInference.Audio.Models.HeartMula;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Pipelines;

/// <summary>HeartMuLa music-generation pipeline. The LM is the **already-built+verified** CSM dual-transformer
/// (<see cref="CsmModel"/>): a Llama-3B global backbone + Llama-300M depth decoder emit 8 RVQ codebooks per
/// 12.5 Hz frame, conditioned on lyrics tokens (+ a MuQ-MuLan style embedding, staged). Returns the codebook
/// grid; HeartCodec waveform decode (48 kHz flow-matching codec) is staged.</summary>
public sealed unsafe class HeartMulaPipeline : IDisposable
{
    private readonly HeartMulaConfig _cfg;
    private readonly CsmModel _lm;
    private readonly HeartCodecDecoder _codec;
    private int _disposed;

    public HeartMulaPipeline(HeartMulaConfig cfg)
    {
        _cfg = cfg;
        _lm = new CsmModel(cfg.Lm);
        _codec = new HeartCodecDecoder(cfg);
    }

    public int SampleRate => _cfg.Lm.SampleRate;

    public HeartCodecDecoder Codec => _codec;

    /// <summary>Loads the LM weights from the **raw** HeartMuLa checkpoint (torchtune layout). The keys are
    /// remapped onto the engine layout by <see cref="HeartMulaCheckpointConverter"/> first.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w) =>
        _lm.LoadWeights(HeartMulaCheckpointConverter.Remap(w, _cfg));

    /// <summary>Loads the HeartCodec decoder weights (separate checkpoint from the LM). Call before
    /// <see cref="Generate"/>; <see cref="GenerateCodes"/> alone needs only the LM weights.</summary>
    public void LoadCodecWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "") =>
        _codec.LoadWeights(w, prefix);

    /// <summary>Generates the 8-codebook grid <c>[NumCodebooks, T]</c> from lyrics token ids (the audio EOS
    /// frame stops generation), optionally prefixed with a MuQ style-conditioning embedding already projected
    /// into the LM hidden (<c>[1, hiddenSize]</c>, from <see cref="MuqEmbedder.ProjectToLmHidden"/>).
    /// <para>The generation params default to the config (<see cref="HeartMulaConfig.Temperature"/> 1.0,
    /// <see cref="HeartMulaConfig.TopK"/> 50, <see cref="HeartMulaConfig.TopP"/> 1.0) but are exposed per-call.
    /// Classifier-free guidance (<paramref name="cfgScale"/>, default <see cref="HeartMulaConfig.CfgScale"/> 1.5)
    /// re-runs the LM on an unconditional context (no lyrics/style, same AR audio frames) and blends logits;
    /// pass 1.0 to disable.</para></summary>
    public int[,] GenerateCodes(IBackend backend, ReadOnlySpan<int> lyricsTokens, int maxFrames, int seed = 0,
        Tensor? muqLmEmbed = null, float? temperature = null, int? topK = null, float? topP = null, float? cfgScale = null,
        CancellationToken cancel = default, Action<int, int>? onFrame = null)
    {
        ThrowIfDisposed();
        int nb = _cfg.Lm.NumCodebooks;
        float temp = temperature ?? _cfg.Temperature;
        int tk = topK ?? _cfg.TopK;
        float tp = topP ?? _cfg.TopP;
        float g = cfgScale ?? _cfg.CfgScale;
        bool useCfg = g != 1f;
        uint rng = DeterministicRng.Seed(seed);
        List<int[]> frames = new(maxFrames);
        int bh = _cfg.Lm.Backbone.HiddenSize;
        int prefixLen = (muqLmEmbed is not null ? 1 : 0) + lyricsTokens.Length;
        List<int[]> noFrames = new(0);

        // Persistent backbone KV cache(s): the first step feeds the whole conditioning prefix (MuQ style + lyrics),
        // and every later step appends ONLY the previous frame's summed audio embedding (one row) — so the 3B
        // backbone runs O(1) per frame instead of re-scanning the whole growing context (the old O(n²) hot path).
        // Numerically identical to the stateless GenerateFrame path (same kernels, RoPE positions, causal key set).
        // The uncond stream (CFG) is the audio frames only; its frame-0 context is a single zeroed row that must not
        // persist, so it runs "standalone" (throwaway cache) — matching the stateless uncond positions exactly.
        using CsmModel.DecodeSession session = _lm.CreateSession(prefixLen + maxFrames + 4, maxFrames + 4, useCfg);
        for (int f = 0; f < maxFrames; f++)
        {
            cancel.ThrowIfCancellationRequested();   // per-frame cancellation checkpoint (Stop Generation)
            Tensor condNew;
            Tensor? uncondNew;
            bool standalone;
            if (f == 0)
            {
                condNew = BuildContext(lyricsTokens, noFrames, muqLmEmbed);
                uncondNew = useCfg ? new Tensor(new TensorShape(1, 1, bh), DType.F32) : null;   // zeroed
                standalone = true;
            }
            else
            {
                condNew = _lm.EmbedAudioFrame(frames[f - 1]);
                uncondNew = useCfg ? _lm.EmbedAudioFrame(frames[f - 1]) : null;
                standalone = false;
            }
            int[] codes = _lm.StepFrame(backend, session, condNew, ref rng, temp, tk, tp, useCfg ? g : 1f, uncondNew, standalone);
            condNew.Dispose();
            uncondNew?.Dispose();
            if (codes[0] >= _cfg.Lm.AudioEosToken) break;   // upstream stops on codebook-0 >= audio_eos_id
            frames.Add(codes);
            onFrame?.Invoke(frames.Count, maxFrames);       // progress: frames produced so far / cap
        }

        int t = frames.Count;
        int[,] grid = new int[nb, t];
        for (int j = 0; j < t; j++)
            for (int i = 0; i < nb; i++) grid[i, j] = frames[j][i];
        return grid;
    }

    /// <summary>End-to-end: lyrics tokens → CSM AR codes (8 codebooks) → HeartCodec flow-matching decode →
    /// 48 kHz mono waveform. <paramref name="muqLmEmbed"/> is the optional MuQ style conditioning projected
    /// into the LM hidden. Requires both the LM and codec weights to be loaded.</summary>
    public float[] Generate(IBackend backend, ReadOnlySpan<int> lyricsTokens, int maxFrames, int seed = 0,
        Tensor? muqLmEmbed = null, float? temperature = null, int? topK = null, float? topP = null, float? cfgScale = null,
        CancellationToken cancel = default, Action<int, int>? onFrame = null)
    {
        ThrowIfDisposed();
        int[,] codes = GenerateCodes(backend, lyricsTokens, maxFrames, seed, muqLmEmbed, temperature, topK, topP, cfgScale, cancel, onFrame);
        if (codes.GetLength(1) == 0) return [];
        cancel.ThrowIfCancellationRequested();   // don't start the (cheaper) codec decode if already cancelled
        return _codec.Decode(backend, codes, seed);
    }

    private Tensor BuildContext(ReadOnlySpan<int> lyrics, List<int[]> frames, Tensor? muqLmEmbed)
    {
        // Concatenate optional MuQ style row + lyrics text embeddings + prior audio-frame embeddings.
        int h = _cfg.Lm.Backbone.HiddenSize;
        int muqRows = muqLmEmbed is not null ? 1 : 0;
        int total = muqRows + lyrics.Length + frames.Count;
        Tensor ctx = new(new TensorShape(1, Math.Max(1, total), h), DType.F32);
        float* cp = (float*)ctx.DataPointer;
        int row = 0;
        if (muqLmEmbed is not null)
        {
            if (muqLmEmbed.ElementCount != h)
                throw new ArgumentException($"MuQ LM embedding must have {h} elements, got {muqLmEmbed.ElementCount}.", nameof(muqLmEmbed));
            Buffer.MemoryCopy((void*)muqLmEmbed.DataPointer, cp + (long)row++ * h, h * 4, h * 4);
        }
        for (int i = 0; i < lyrics.Length; i++)
        {
            Tensor e = _lm.EmbedText(lyrics[i]);
            Buffer.MemoryCopy((void*)e.DataPointer, cp + (long)row++ * h, h * 4, h * 4); e.Dispose();
        }
        foreach (int[] fr in frames)
        {
            Tensor e = _lm.EmbedAudioFrame(fr);
            Buffer.MemoryCopy((void*)e.DataPointer, cp + (long)row++ * h, h * 4, h * 4); e.Dispose();
        }
        return ctx;
    }

    /// <summary>Parity probe: builds the prompt-frame context from <paramref name="lyricsTokens"/> (text in the
    /// last column, audio columns empty/masked — matching the upstream frame-0 input) and returns the
    /// teacher-forced codebook logits for <paramref name="forcedCodes"/>. Not used in inference.</summary>
    public (float[] C0, float[][] Dec) DebugFrameLogits(IBackend backend, ReadOnlySpan<int> lyricsTokens, ReadOnlySpan<int> forcedCodes)
    {
        ThrowIfDisposed();
        Tensor ctx = BuildContext(lyricsTokens, new List<int[]>(), null);
        try { return _lm.DebugFrameLogits(backend, ctx, forcedCodes); }
        finally { ctx.Dispose(); }
    }

    public IEnumerable<Tensor> EnumerateWeights() => _lm.EnumerateWeights();

    /// <summary>Enumerates the HeartCodec decoder weights (for save/validation), separate from the LM set.</summary>
    public IEnumerable<Tensor> EnumerateCodecWeights() => _codec.EnumerateWeights();

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _lm.Dispose();
        _codec.Dispose();
        GC.SuppressFinalize(this);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed != 0) throw new ObjectDisposedException(nameof(HeartMulaPipeline));
    }
}
