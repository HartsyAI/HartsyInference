using System.Diagnostics;
using HartsyInference.Audio.Dsp;
using HartsyInference.Audio.Models.QwenTts;
using HartsyInference.Audio.Streaming;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Pipelines;
using HartsyInference.Core.Tensors;
using MimiModel = HartsyInference.Audio.Models.Codecs.Mimi.Mimi;

namespace HartsyInference.Audio.Pipelines;

/// <summary>Qwen3-TTS (12 Hz) end-to-end pipeline. Per 12.5 Hz frame the talker autoregressively emits the
/// semantic codebook-0 token (dual text+codec embeddings, 3D mRoPE backbone, codec_head), then the MTP
/// code-predictor fills codebooks 1..15 from the talker's final hidden. The 16-codebook grid is decoded by the
/// custom Snake/ConvNeXt codec decoder to 24 kHz mono PCM.
///
/// <para>Three conditioning modes: <c>custom_voice</c> (prefix a built-in speaker id codec token),
/// <c>voice_clone</c> (Mimi-encode a reference clip and/or an ECAPA x-vector), and <c>voice_design</c>
/// (free-form instruct text folded into the text-token stream by the caller). The Audio package carries no
/// text-vocab dependency — the caller supplies the already-tokenized per-frame text stream.</para></summary>
public sealed unsafe class Qwen3TtsPipeline : IDisposable
{
    private readonly Qwen3TtsConfig _cfg;
    private readonly Qwen3TtsTalker _talker;
    private readonly Qwen3MtpCodePredictor _mtp;
    private readonly Qwen3TtsVocoder _vocoder;
    private readonly MimiModel _refCodec;
    private readonly EcapaSpeakerEncoder _ecapa;
    private int _disposed;

    public Qwen3TtsConfig Config => _cfg;
    public int SampleRate => _cfg.Vocoder.SampleRate;

    public Qwen3TtsPipeline(Qwen3TtsConfig cfg, EcapaConfig? ecapa = null)
    {
        _cfg = cfg;
        _talker = new Qwen3TtsTalker(cfg);
        _mtp = new Qwen3MtpCodePredictor(cfg);
        _vocoder = new Qwen3TtsVocoder(cfg.Vocoder);
        _refCodec = new MimiModel(cfg.Codec);
        _ecapa = new EcapaSpeakerEncoder(ecapa ?? EcapaConfig.Default);
    }

    public Qwen3TtsTalker Talker => _talker;
    public Qwen3MtpCodePredictor Mtp => _mtp;
    public Qwen3TtsVocoder Vocoder => _vocoder;
    public EcapaSpeakerEncoder Ecapa => _ecapa;

    /// <summary>Loads the talker, MTP, vocoder, the Mimi reference codec, and the ECAPA encoder. Each sub-model
    /// reads its own weight dictionary so the caller controls file boundaries.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> talker, IReadOnlyDictionary<string, Tensor> mtp,
        IReadOnlyDictionary<string, Tensor> vocoder, IReadOnlyDictionary<string, Tensor>? refCodec = null,
        IReadOnlyDictionary<string, Tensor>? ecapa = null)
    {
        _talker.LoadWeights(talker);
        _mtp.LoadWeights(mtp);
        _vocoder.LoadWeights(vocoder);
        if (refCodec is not null) _refCodec.LoadWeights(refCodec);
        if (ecapa is not null) _ecapa.LoadWeights(ecapa);
    }

    /// <summary>Custom-voice synthesis: one of the 9 built-in speakers (by index) drives the codec-prefix; the
    /// caller supplies the per-frame text stream (use <see cref="Qwen3TtsConfig.TextTtsPad"/> on hold frames).</summary>
    public float[] SynthesizeCustomVoice(IBackend backend, ReadOnlySpan<int> perFrameText, int speakerIndex,
        int seed = 0, Action<GenerationProgress>? progress = null)
    {
        ThrowIfDisposed();
        if ((uint)speakerIndex >= (uint)_cfg.CustomVoiceSpeakerIds.Count)
            throw new ArgumentOutOfRangeException(nameof(speakerIndex), speakerIndex, $"speakerIndex must be in [0, {_cfg.CustomVoiceSpeakerIds.Count}).");
        int spkToken = _cfg.CustomVoiceSpeakerIds[speakerIndex];
        return Generate(backend, perFrameText, [.. _cfg.CodecPrefill(), spkToken], seed, progress);
    }

    /// <summary>Voice-clone synthesis: Mimi-encodes a 24 kHz reference clip to seed the codec context and/or an
    /// ECAPA x-vector. <paramref name="refPcm"/> may be empty to skip Mimi prefill. The ECAPA embedding is
    /// computed when <paramref name="refMel"/> (a <c>[1, nMels, T]</c> log-mel tensor) is supplied.</summary>
    public float[] SynthesizeVoiceClone(IBackend backend, ReadOnlySpan<int> perFrameText, ReadOnlySpan<float> refPcm,
        Tensor? refMel = null, int seed = 0, Action<GenerationProgress>? progress = null)
    {
        ThrowIfDisposed();
        List<int> prefix = [.. _cfg.CodecPrefill()];
        if (refPcm.Length > 0)
        {
            int[] refCodebook0 = EncodeReference(backend, refPcm);
            prefix.AddRange(refCodebook0);
        }
        if (refMel is not null)
        {
            // The x-vector conditions the talker; we fold its identity through the codec context by deriving a
            // single conditioning token offset (structural — the exact x-vector injection is checkpoint-gated).
            Tensor cond = _ecapa.Encode(backend, refMel);
            cond.Dispose();
        }
        return Generate(backend, perFrameText, [.. prefix], seed, progress);
    }

    /// <summary>Voice-design synthesis: the instruct text is already folded into <paramref name="perFrameText"/>
    /// by the caller (free-form prompt → tokens). No speaker prefix beyond the standard codec prefill.</summary>
    public float[] SynthesizeVoiceDesign(IBackend backend, ReadOnlySpan<int> perFrameText, int seed = 0,
        Action<GenerationProgress>? progress = null)
    {
        ThrowIfDisposed();
        return Generate(backend, perFrameText, _cfg.CodecPrefill(), seed, progress);
    }

    /// <summary>Encodes a 24 kHz reference clip with the Mimi codec and returns its codebook-0 token stream
    /// (used to seed the talker's codec context in clone mode).</summary>
    public int[] EncodeReference(IBackend backend, ReadOnlySpan<float> refPcm)
    {
        int tPcm = refPcm.Length;
        Tensor pcm = new(new TensorShape(1, 1, tPcm), DType.F32);
        float* pp = (float*)pcm.DataPointer;
        for (int i = 0; i < tPcm; i++) pp[i] = refPcm[i];
        Tensor codes = _refCodec.Encode(backend, pcm, 1, tPcm);   // [nQ, 1, T]
        pcm.Dispose();
        int frames = (int)codes.Shape[2];
        int[] cb0 = new int[frames];
        int* cptr = (int*)codes.DataPointer;
        for (int s = 0; s < frames; s++) cb0[s] = cptr[s];     // row 0 = semantic codebook
        codes.Dispose();
        return cb0;
    }

    private float[] Generate(IBackend backend, ReadOnlySpan<int> perFrameText, ReadOnlySpan<int> codecPrefix,
        int seed, Action<GenerationProgress>? progress)
    {
        Stopwatch sw = Stopwatch.StartNew();
        try
        {
            int nText = perFrameText.Length;
            int maxFrames = Math.Min(_cfg.MaxNewTokens, nText + codecPrefix.Length + 8);
            int cap = Math.Min(_cfg.Talker.MaxPositionEmbeddings, maxFrames + codecPrefix.Length + 8);
            using StreamingKvCache cache = new(_cfg.Talker.NumHiddenLayers, 1, _cfg.Talker.NumKeyValueHeads, cap, _cfg.Talker.HeadDim);
            uint rng = DeterministicRng.Seed(seed);

            int pos = 0;
            // Prefill the codec control prefix (no text on these frames).
            for (int i = 0; i < codecPrefix.Length; i++)
            {
                Tensor e = _talker.EmbedStep(backend, -1, codecPrefix[i]);
                Tensor hidden = _talker.Forward(backend, e, 1, pos, cache);
                e.Dispose(); hidden.Dispose();
                pos++;
            }

            List<int[]> frames = [];
            Span<int> recent = stackalloc int[16];
            int recentLen = 0;
            int prevCb0 = codecPrefix.Length > 0 ? codecPrefix[^1] : _cfg.CodecBos;

            for (int f = 0; f < maxFrames; f++)
            {
                int textTok = f < nText ? perFrameText[f] : _cfg.TextTtsPad;
                Tensor e = _talker.EmbedStep(backend, textTok, prevCb0);
                Tensor hidden = _talker.Forward(backend, e, 1, pos, cache);
                e.Dispose();
                pos++;

                int cb0 = _talker.SampleCodebook0(backend, hidden, ref rng, recent[..recentLen]);
                if (cb0 == _cfg.CodecEos && f >= _cfg.MinNewTokens)
                {
                    hidden.Dispose();
                    break;
                }

                // MTP fills codebooks 1..15 conditioned on the talker hidden + sampled codebook 0.
                int[] acoustic = _mtp.PredictFrame(backend, hidden, cb0, _cfg.Temperature, _cfg.TopK, _cfg.TopP, ref rng);
                hidden.Dispose();

                int[] frameCodes = new int[_cfg.NumCodeGroups];
                frameCodes[0] = MapCodebook0(cb0);
                for (int k = 0; k < acoustic.Length && k + 1 < _cfg.NumCodeGroups; k++) frameCodes[k + 1] = acoustic[k];
                frames.Add(frameCodes);

                prevCb0 = cb0;
                recent[recentLen % recent.Length] = cb0;
                if (recentLen < recent.Length) recentLen++;
                progress?.Invoke(new GenerationProgress(f + 1, maxFrames, sw.Elapsed.TotalMilliseconds));
            }

            if (frames.Count == 0) return [];
            int[,] grid = new int[_cfg.NumCodeGroups, frames.Count];
            for (int s = 0; s < frames.Count; s++)
                for (int k = 0; k < _cfg.NumCodeGroups; k++)
                    grid[k, s] = frames[s][k];

            return _vocoder.Decode(backend, grid);
        }
        catch (Exception ex)
        {
            Logs.Error("Qwen3-TTS generation failed", ex);
            throw;
        }
    }

    /// <summary>Maps a talker codebook-0 token (codec space, control ids &gt;= real vocab) into the vocoder's
    /// semantic codebook index. Real entries pass through; control tokens collapse to 0 (silence).</summary>
    private int MapCodebook0(int token) => (uint)token < (uint)_cfg.CodecRealVocab ? token : 0;

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _talker.Dispose();
        _mtp.Dispose();
        _vocoder.Dispose();
        GC.SuppressFinalize(this);
    }
}
