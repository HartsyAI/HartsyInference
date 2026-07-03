using System.Diagnostics;
using System.Runtime.InteropServices;
using HartsyInference.Audio.Dsp;
using HartsyInference.Audio.Models.QwenTts;
using HartsyInference.Core.Backends;
using HartsyInference.LLM.Transformer;
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
    private static readonly bool DebugCodes = Environment.GetEnvironmentVariable("QWEN3_DEBUG") == "1";

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

    /// <summary>Custom-voice synthesis. <paramref name="textTokens"/> is the Qwen BPE tokenization of the text to
    /// speak (NOT a per-frame stream — the pipeline builds the ChatML dual-stream prefill). <paramref name="speakerToken"/>
    /// is a codec-space preset-speaker id (e.g. Ryan/en 3061, Serena/zh 3066 on the CustomVoice checkpoint).
    /// <paramref name="languageId"/> is a codec-space language id (e.g. English 2050) or -1 to omit.</summary>
    public float[] SynthesizeCustomVoice(IBackend backend, ReadOnlySpan<int> textTokens, int speakerToken,
        int languageId = -1, int seed = 0, Action<GenerationProgress>? progress = null)
    {
        ThrowIfDisposed();
        return GenerateFromPrefill(backend, BuildPrefill(textTokens, speakerToken, languageId), seed, progress);
    }

    /// <summary>Voice-design synthesis: the voice comes from instruct text folded into <paramref name="textTokens"/>
    /// by the caller. No speaker token in the codec prefix.</summary>
    public float[] SynthesizeVoiceDesign(IBackend backend, ReadOnlySpan<int> textTokens, int languageId = -1,
        int seed = 0, Action<GenerationProgress>? progress = null)
    {
        ThrowIfDisposed();
        return GenerateFromPrefill(backend, BuildPrefill(textTokens, speakerToken: -1, languageId), seed, progress);
    }

    /// <summary>Voice-clone synthesis: Mimi-encodes a 24 kHz reference clip to seed the codec context and/or an
    /// ECAPA x-vector. <b>Structural/ICL path pending real-weights validation</b>; custom_voice/voice_design are the
    /// verified modes. <paramref name="refPcm"/> may be empty.</summary>
    public float[] SynthesizeVoiceClone(IBackend backend, ReadOnlySpan<int> textTokens, ReadOnlySpan<float> refPcm,
        Tensor? refMel = null, int languageId = -1, int seed = 0, Action<GenerationProgress>? progress = null)
    {
        ThrowIfDisposed();
        if (refMel is not null) { Tensor cond = _ecapa.Encode(backend, refMel); cond.Dispose(); }
        return GenerateFromPrefill(backend, BuildPrefill(textTokens, speakerToken: -1, languageId), seed, progress);
    }

    /// <summary>Builds the dual-stream prefill as ordered (text-side token, codec-side token) pairs, reproducing the
    /// reference layout: role prefix (text only) → codec control prefix (paired with tts_pad/tts_bos) → text content
    /// + tts_eos (paired with codec_pad) → final (tts_pad + codec_bos). A -1 on either side contributes nothing.</summary>
    private List<(int Text, int Codec)> BuildPrefill(ReadOnlySpan<int> textTokens, int speakerToken, int languageId)
    {
        List<(int, int)> pairs = new(textTokens.Length + 12);
        // Section 1: role prefix <|im_start|>assistant\n (text only).
        pairs.Add((_cfg.TextImStart, -1));
        pairs.Add((_cfg.TextAssistant, -1));
        pairs.Add((_cfg.TextNewline, -1));

        // Codec control prefix.
        List<int> codec = new(8);
        if (languageId >= 0) { codec.Add(_cfg.CodecThink); codec.Add(_cfg.CodecThinkBos); codec.Add(languageId); codec.Add(_cfg.CodecThinkEos); }
        else { codec.Add(_cfg.CodecNoThink); codec.Add(_cfg.CodecThinkBos); codec.Add(_cfg.CodecThinkEos); }
        if (speakerToken >= 0) codec.Add(speakerToken);
        codec.Add(_cfg.CodecPad);
        codec.Add(_cfg.CodecBos);

        // Section 2: codec prefix without its last (codec_bos); text = tts_pad..., last = tts_bos.
        int sec2 = codec.Count - 1;
        for (int i = 0; i < sec2; i++)
            pairs.Add((i == sec2 - 1 ? _cfg.TextTtsBos : _cfg.TextTtsPad, codec[i]));

        // Section 3: text content then tts_eos, each paired with codec_pad.
        for (int i = 0; i < textTokens.Length; i++) pairs.Add((textTokens[i], _cfg.CodecPad));
        pairs.Add((_cfg.TextTtsEod, _cfg.CodecPad));

        // Section 4: final position tts_pad + codec_bos.
        pairs.Add((_cfg.TextTtsPad, _cfg.CodecBos));
        return pairs;
    }

    /// <summary>Encodes a 24 kHz reference clip with the Mimi codec and returns its codebook-0 token stream.</summary>
    public int[] EncodeReference(IBackend backend, ReadOnlySpan<float> refPcm)
    {
        int tPcm = refPcm.Length;
        Tensor pcm = new(new TensorShape(1, 1, tPcm), DType.F32);
        float* pp = (float*)pcm.DataPointer;
        for (int i = 0; i < tPcm; i++) pp[i] = refPcm[i];
        Tensor codes = _refCodec.Encode(backend, pcm, 1, tPcm);
        pcm.Dispose();
        int frames = (int)codes.Shape[2];
        int[] cb0 = new int[frames];
        int* cptr = (int*)codes.DataPointer;
        for (int s = 0; s < frames; s++) cb0[s] = cptr[s];
        codes.Dispose();
        return cb0;
    }

    /// <summary>Runs the talker over the dual-stream prefill, then autoregressively generates audio frames until
    /// <see cref="Qwen3TtsConfig.CodecEos"/>. Each generation frame's input is the text-side <c>tts_pad</c> plus the
    /// previous frame's full 16-codebook codec feedback (talker codebook-0 embed + the 15 code-predictor acoustic
    /// embeds). The 16-code grid is then decoded to PCM.</summary>
    private float[] GenerateFromPrefill(IBackend backend, List<(int Text, int Codec)> prefill, int seed,
        Action<GenerationProgress>? progress)
    {
        Stopwatch sw = Stopwatch.StartNew();
        try
        {
            int h = _cfg.Talker.HiddenSize;
            int prefillLen = prefill.Count;
            int cap = Math.Min(_cfg.Talker.MaxPositionEmbeddings, prefillLen + _cfg.MaxNewTokens + 8);
            using IKvCache cache = _talker.Backbone.CreateDecodeCache(cap);
            uint rng = DeterministicRng.Seed(seed);

            // Build and run the prefill in one batched forward; the last position's hidden predicts codebook-0.
            Tensor prefillEmbed = new(new TensorShape(1, prefillLen, h), DType.F32);
            float* pe = (float*)prefillEmbed.DataPointer;
            for (int i = 0; i < prefillLen; i++)
            {
                Tensor e = _talker.EmbedStep(backend, prefill[i].Text, prefill[i].Codec);
                Buffer.MemoryCopy((void*)e.DataPointer, pe + (long)i * h, h * 4, h * 4);
                e.Dispose();
            }
            Tensor hiddenAll = _talker.Forward(backend, prefillEmbed, prefillLen, 0, cache);
            prefillEmbed.Dispose();
            Tensor? curHidden = SliceLastPosition(hiddenAll, prefillLen, h);
            hiddenAll.Dispose();
            if (DebugCodes)
            {
                float* ch = (float*)curHidden!.DataPointer;
                Console.Error.WriteLine($"[qwen3gen] prefill last_hidden[:8]=[{ch[0]:0.000000},{ch[1]:0.000000},{ch[2]:0.000000},{ch[3]:0.000000},{ch[4]:0.000000},{ch[5]:0.000000},{ch[6]:0.000000},{ch[7]:0.000000}]");
            }
            int pos = prefillLen;

            List<int[]> frames = [];
            // Repetition penalty runs over EVERY previously generated codebook-0 token (the reference penalizes the
            // whole generated sequence, once per unique id) — not a sliding window.
            List<int> generated = new(256);
            bool[] seen = new bool[_cfg.CodecHeadOut];

            for (int f = 0; f < _cfg.MaxNewTokens; f++)
            {
                Tensor hid = curHidden!;
                // EOS is masked until MinNewTokens frames exist (reference min_new_tokens=2); control ids other
                // than EOS are always suppressed inside SampleCodebook0 (reference suppress_tokens).
                int cb0 = _talker.SampleCodebook0(backend, hid, ref rng, CollectionsMarshal.AsSpan(generated),
                    allowEos: f >= _cfg.MinNewTokens);
                if (cb0 == _cfg.CodecEos)
                {
                    hid.Dispose();
                    break;
                }
                // Code predictor fills codebooks 1..15 (sampled, as in the reference), conditioned on the talker
                // hidden + the codebook-0 embedding (looked up through the talker's codec table).
                Tensor code0Embed = _talker.CodecEmbedRow(backend, cb0);
                int[] acoustic = _mtp.PredictFrame(backend, hid, code0Embed, ref rng);
                code0Embed.Dispose();
                hid.Dispose();

                int[] frameCodes = new int[_cfg.NumCodeGroups];
                frameCodes[0] = MapCodebook0(cb0);
                for (int k = 0; k < acoustic.Length && k + 1 < _cfg.NumCodeGroups; k++) frameCodes[k + 1] = acoustic[k];
                frames.Add(frameCodes);

                if ((uint)cb0 < (uint)seen.Length && !seen[cb0]) { seen[cb0] = true; generated.Add(cb0); }
                if (DebugCodes) Console.Error.WriteLine($"[qwen3gen] frame {f,3} cb0={cb0,5} acoustic[0..3]={acoustic[0]},{acoustic[1]},{acoustic[2]}");
                progress?.Invoke(new GenerationProgress(f + 1, _cfg.MaxNewTokens, sw.Elapsed.TotalMilliseconds));

                // Next-frame talker input: tts_pad (text) + this frame's 16-codebook codec feedback.
                Tensor e = _talker.EmbedStep(backend, _cfg.TextTtsPad, cb0);   // text_proj(pad) + talker codec_embed[cb0]
                _mtp.AddAcousticFeedback(e, acoustic);                          // + sum of the 15 acoustic embeds
                curHidden = _talker.Forward(backend, e, 1, pos, cache);
                e.Dispose();
                pos++;
            }

            if (frames.Count == 0) return [];
            int[,] grid = new int[_cfg.NumCodeGroups, frames.Count];
            for (int s = 0; s < frames.Count; s++)
                for (int k = 0; k < _cfg.NumCodeGroups; k++)
                    grid[k, s] = frames[s][k];

            // The talker loop leaves thousands of step activations + pool reservations on the device; the codes
            // are on the host now, so reclaim before the vocoder's large full-sequence decode (12GB cards OOM'd
            // here otherwise — 10.7GB peak observed at the decode of a ~600-step generation).
            backend.FreeActivations();
            backend.TrimMemoryPool();
            return _vocoder.Decode(backend, grid);
        }
        catch (Exception ex)
        {
            Logs.Error("Qwen3-TTS generation failed", ex);
            throw;
        }
    }

    /// <summary>Copies the last time position of a <c>[1, T, h]</c> hidden into a fresh <c>[1, 1, h]</c> tensor.</summary>
    private static Tensor SliceLastPosition(Tensor hiddenAll, int t, int h)
    {
        Tensor last = new(new TensorShape(1, 1, h), DType.F32);
        Buffer.MemoryCopy((byte*)hiddenAll.DataPointer + (long)(t - 1) * h * 4, (void*)last.DataPointer, h * 4, h * 4);
        return last;
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
