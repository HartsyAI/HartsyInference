using System.Diagnostics;
using HartsyInference.Audio.Models.Chatterbox;
using HartsyInference.Audio.Models.CosyVoice;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Pipelines;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Pipelines;

/// <summary>Resemble AI Chatterbox end-to-end text-to-speech pipeline. Wires the net-new T3 backbone
/// (<see cref="ChatterboxT3"/> — a Llama-style AR LM emitting 25 Hz S3 speech tokens) to the S3Gen stage,
/// which is architecturally identical to CosyVoice 2's S3Gen and is therefore reused verbatim:
/// <see cref="CosyVoiceFlow"/> (speech-token → mel conditional flow matching) + <see cref="HiFTNetVocoder"/>
/// (mel → 24 kHz waveform). Chatterbox ships its own weights for these shared modules.
///
/// <para>Two speaker embeddings exist in Chatterbox: a 256-d LSTM voice-encoder embedding feeds T3, and a
/// CAM++ x-vector (192-d) feeds the S3Gen flow. In zero-shot mode both (plus the T3 cond-prompt tokens and
/// the flow's prompt tokens + mel) are derived from the raw reference clip, replicating upstream
/// <c>prepare_conditionals</c>; in precomputed mode the caller passes them in (the <c>conds.pt</c> default
/// voice). Mirrors <see cref="CosyVoicePipeline"/>'s orchestration and disposal.</para>
///
/// <para><b>Text tokenization is the caller's responsibility</b> — <see cref="Synthesize"/> takes T3 BPE
/// token IDs, matching the token-IDs-in convention of the other HartsyInference audio pipelines.</para></summary>
public sealed class ChatterboxPipeline : IDisposable
{
    private readonly ChatterboxConfig _cfg;
    private readonly ChatterboxT3 _t3;
    private readonly CosyVoiceFlow _flow;
    private readonly HiFTNetVocoder _vocoder;
    private readonly CamPlusSpeakerEncoder? _speakerEncoder;
    private readonly S3Tokenizer? _s3;
    private readonly ChatterboxVoiceEncoder? _voiceEncoder;
    private int _disposed;

    public ChatterboxPipeline(ChatterboxConfig cfg, ChatterboxT3 t3, CosyVoiceFlow flow,
        HiFTNetVocoder vocoder, CamPlusSpeakerEncoder? speakerEncoder = null, S3Tokenizer? s3 = null,
        ChatterboxVoiceEncoder? voiceEncoder = null)
    {
        _cfg = cfg ?? throw new ArgumentNullException(nameof(cfg));
        _t3 = t3 ?? throw new ArgumentNullException(nameof(t3));
        _flow = flow ?? throw new ArgumentNullException(nameof(flow));
        _vocoder = vocoder ?? throw new ArgumentNullException(nameof(vocoder));
        _speakerEncoder = speakerEncoder;
        _s3 = s3;
        _voiceEncoder = voiceEncoder;
    }

    /// <summary>Synthesizes 24 kHz audio for the given T3 text token IDs.
    /// <list type="bullet">
    ///   <item><b>Precomputed mode</b> (default voice): pass <paramref name="refSpeakerEmbed"/> (the 256-d
    ///         voice-encoder embedding), <paramref name="flowSpeakerEmbed"/> (<c>[1, 192]</c> CAM++), and
    ///         <paramref name="t3PromptSpeechTokens"/>; leave <paramref name="referenceAudio"/> empty.</item>
    ///   <item><b>Zero-shot mode</b> (voice cloning): pass the raw <paramref name="referenceAudio"/> samples at
    ///         <paramref name="referenceSampleRate"/>. The pipeline derives everything upstream
    ///         <c>prepare_conditionals</c> does — the voice-encoder T3 embedding (full-length clip), T3
    ///         cond-prompt tokens (first 6 s, ≤150 tokens), and the S3Gen reference dict (CAM++ x-vector +
    ///         prompt tokens + prompt mel from the first 10 s). Requires the CAM++, S3-tokenizer, and (unless
    ///         <paramref name="refSpeakerEmbed"/> is supplied) voice-encoder modules to be attached.</item>
    /// </list></summary>
    public float[] Synthesize(IBackend backend,
        ReadOnlySpan<int> textTokens,
        Tensor? refSpeakerEmbed,
        float exaggeration = 0.5f,
        int seed = 0,
        Tensor? flowSpeakerEmbed = null,
        ReadOnlySpan<float> referenceAudio = default,
        int referenceSampleRate = 0,
        Action<GenerationProgress>? progress = null,
        ReadOnlySpan<int> t3PromptSpeechTokens = default,
        float cfgWeight = 0f)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(backend);
        Stopwatch sw = Stopwatch.StartNew();

        progress?.Invoke(new GenerationProgress(0, 3, 0));

        // Resolve conditioning: either derived from the raw reference clip (zero-shot) or passed in (precomputed).
        int[] promptSpeechTokens = [];
        ReadOnlySpan<int> t3Prompt = t3PromptSpeechTokens;
        Tensor? promptMel = null;
        Tensor? veEmbed = refSpeakerEmbed;
        Tensor? flowSpk = flowSpeakerEmbed;
        bool ownsVe = false, ownsFlowSpk = false;
        if (!referenceAudio.IsEmpty)
        {
            if (referenceSampleRate <= 0)
                throw new ArgumentException("referenceSampleRate must be set when referenceAudio is provided.", nameof(referenceSampleRate));
            if (_speakerEncoder is null || _s3 is null)
                throw new InvalidOperationException("Zero-shot mode requires a CamPlusSpeakerEncoder and an S3Tokenizer to be attached.");

            // prepare_conditionals: the voice encoder and T3 prompt tokenization see the full-length 16 kHz
            // reference; S3Gen's embed_ref sees the first 10 s (and derives its own 16 kHz copy from that).
            float[] ref16Full = S3GenReference.Resample(referenceAudio, referenceSampleRate, 16_000);
            ReadOnlySpan<float> refDec = referenceAudio[..Math.Min(referenceAudio.Length, _cfg.S3GenCondSeconds * referenceSampleRate)];
            float[] ref24Dec = S3GenReference.Resample(refDec, referenceSampleRate, 24_000);
            float[] ref16Dec = S3GenReference.Resample(refDec, referenceSampleRate, 16_000);

            if (veEmbed is null)
            {
                if (_voiceEncoder is null)
                    throw new InvalidOperationException("Zero-shot mode requires a ChatterboxVoiceEncoder (or a precomputed refSpeakerEmbed).");
                veEmbed = _voiceEncoder.EmbedUtterance(backend, ref16Full);
                ownsVe = true;
            }
            if (t3Prompt.IsEmpty)
            {
                int encSamples = Math.Min(ref16Full.Length, _cfg.T3CondSeconds * 16_000);
                int[] condTokens = S3GenReference.SpeechTokens(backend, _s3, ref16Full.AsSpan(0, encSamples));
                t3Prompt = condTokens.Length > _cfg.SpeechCondPromptLen ? condTokens[.._cfg.SpeechCondPromptLen] : condTokens;
            }

            promptSpeechTokens = S3GenReference.SpeechTokens(backend, _s3, ref16Dec);
            flowSpk = S3GenReference.SpeakerEmbedding(backend, _speakerEncoder, ref16Dec);
            ownsFlowSpk = true;
            promptMel = S3GenReference.FlowMel(ref24Dec);
            // embed_ref invariant: mel_len == 2 * token_len (50 Hz mel vs 25 Hz tokens); trim tokens to match.
            int melFrames = (int)promptMel.Shape[2];
            if (promptSpeechTokens.Length > melFrames / 2) promptSpeechTokens = promptSpeechTokens[..(melFrames / 2)];
            Logs.Info($"Chatterbox: reference → VE embedding + {t3Prompt.Length} T3 cond tokens + {promptSpeechTokens.Length} flow prompt tokens.");
        }
        else if (flowSpk is null)
        {
            throw new ArgumentException("Provide either flowSpeakerEmbed (precomputed) or referenceAudio (zero-shot).");
        }
        if (veEmbed is null)
            throw new ArgumentException("refSpeakerEmbed is required when no referenceAudio is supplied.", nameof(refSpeakerEmbed));

        // Stage 1: T3 — text tokens + voice-encoder embedding (+ optional perceiver-resampled cond-prompt
        // speech tokens) → S3 speech tokens.
        List<int> speechTokens = _t3.GenerateSpeechTokens(backend, textTokens, veEmbed,
            exaggeration, _cfg.MaxNewTokens, seed, t3Prompt, cfgWeight);
        if (ownsVe) veEmbed.Dispose();
        Logs.Info($"Chatterbox: T3 emitted {speechTokens.Count} speech tokens in {sw.ElapsedMilliseconds}ms.");
        progress?.Invoke(new GenerationProgress(1, 3, sw.Elapsed.TotalMilliseconds));

        // Stage 2: S3Gen flow — speech tokens → mel (CosyVoice-identical conditional flow matching).
        Tensor mel = _flow.Inference(backend, speechTokens.ToArray(), promptSpeechTokens, promptMel, flowSpk, seed);
        if (ownsFlowSpk) flowSpk.Dispose();
        promptMel?.Dispose();
        progress?.Invoke(new GenerationProgress(2, 3, sw.Elapsed.TotalMilliseconds));

        // Stage 3: HiFTNet vocoder — mel → 24 kHz waveform.
        float[] audio = _vocoder.Forward(backend, mel);
        mel.Dispose();

        sw.Stop();
        progress?.Invoke(new GenerationProgress(3, 3, sw.Elapsed.TotalMilliseconds));
        Logs.Info($"Chatterbox synthesis complete: {audio.Length} samples ({audio.Length / (double)_cfg.SampleRate:F2}s) in {sw.ElapsedMilliseconds}ms.");
        return audio;
    }

    /// <summary>Loads a fused Chatterbox checkpoint, routing each sub-module's weights by key prefix.
    /// Chatterbox ships separate files (<c>t3_cfg.safetensors</c>, <c>s3gen.safetensors</c>,
    /// <c>ve.safetensors</c>); callers may merge them into one dictionary with a per-file prefix or call
    /// the sub-module <c>LoadWeights</c> directly. The prefix map (best-effort, for checkpoint
    /// reconciliation):
    /// <list type="bullet">
    ///   <item><c>t3.*</c> → <see cref="ChatterboxT3"/> (strips <c>t3.</c>; T3 then reads its own
    ///         <c>tfmr.*</c> / <c>text_emb.*</c> / <c>speech_emb.*</c> / <c>*_pos_emb.*</c> /
    ///         <c>speech_head.*</c> / <c>cond_enc.*</c> keys).</item>
    ///   <item><c>s3gen.flow.*</c> → <see cref="CosyVoiceFlow"/> (strips <c>s3gen.</c>, loaded under the
    ///         <c>flow</c> prefix).</item>
    ///   <item><c>s3gen.mel2wav.*</c> / <c>s3gen.hift.*</c> → <see cref="HiFTNetVocoder"/> (the real
    ///         file uses <c>mel2wav.</c>; <c>hift.</c> accepted as an alias).</item>
    ///   <item><c>s3gen.speaker_encoder.*</c> → <see cref="CamPlusSpeakerEncoder"/> (the CAM++ flow
    ///         x-vector). <c>s3gen.tokenizer.*</c> → <see cref="S3Tokenizer"/> when attached.</item>
    ///   <item><c>ve.*</c> → <see cref="ChatterboxVoiceEncoder"/> when attached (from <c>ve.safetensors</c>).</item>
    /// </list></summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(weights);
        try
        {
            _t3.LoadWeights(SubDictionary(weights, "t3."));

            IReadOnlyDictionary<string, Tensor> s3gen = SubDictionary(weights, "s3gen.");
            _flow.LoadWeights(s3gen, "flow");
            _vocoder.LoadWeights(HasPrefix(s3gen, "mel2wav.") ? SubDictionary(s3gen, "mel2wav.") : SubDictionary(s3gen, "hift."));
            _speakerEncoder?.LoadWeights(s3gen, "speaker_encoder");
            if (_s3 is not null && HasPrefix(s3gen, "tokenizer."))
                _s3.LoadWeights(SubDictionary(s3gen, "tokenizer."));
            if (_voiceEncoder is not null && HasPrefix(weights, "ve."))
                _voiceEncoder.LoadWeights(SubDictionary(weights, "ve."));
        }
        catch (Exception ex)
        {
            Logs.Error("Failed to load Chatterbox weights — check the checkpoint prefix mapping.", ex);
            throw;
        }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _t3.Dispose();
        _flow.Dispose();
        _vocoder.Dispose();
        _speakerEncoder?.Dispose();
        _s3?.Dispose();
        _voiceEncoder?.Dispose();
        GC.SuppressFinalize(this);
    }

    private static bool HasPrefix(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        foreach (string key in w.Keys)
            if (key.StartsWith(prefix, StringComparison.Ordinal)) return true;
        return false;
    }

    private static IReadOnlyDictionary<string, Tensor> SubDictionary(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        Dictionary<string, Tensor> sub = new();
        foreach (KeyValuePair<string, Tensor> kv in w)
            if (kv.Key.StartsWith(prefix, StringComparison.Ordinal))
                sub[kv.Key[prefix.Length..]] = kv.Value;
        return sub;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed != 0) throw new ObjectDisposedException(nameof(ChatterboxPipeline));
    }
}
