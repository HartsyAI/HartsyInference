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
/// CAM++ x-vector (192-d) feeds the S3Gen flow. <see cref="Synthesize"/> takes the T3 voice-encoder
/// embedding directly (already produced by <see cref="ChatterboxVoiceEncoder"/>); the flow's CAM++
/// embedding is derived from the reference mel when one is supplied, otherwise the precomputed flow
/// embedding must be passed in. Mirrors <see cref="CosyVoicePipeline"/>'s orchestration and disposal.</para>
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
    private int _disposed;

    public ChatterboxPipeline(ChatterboxConfig cfg, ChatterboxT3 t3, CosyVoiceFlow flow,
        HiFTNetVocoder vocoder, CamPlusSpeakerEncoder? speakerEncoder = null, S3Tokenizer? s3 = null)
    {
        _cfg = cfg ?? throw new ArgumentNullException(nameof(cfg));
        _t3 = t3 ?? throw new ArgumentNullException(nameof(t3));
        _flow = flow ?? throw new ArgumentNullException(nameof(flow));
        _vocoder = vocoder ?? throw new ArgumentNullException(nameof(vocoder));
        _speakerEncoder = speakerEncoder;
        _s3 = s3;
    }

    /// <summary>Synthesizes 24 kHz audio for the given T3 text token IDs.
    /// <list type="bullet">
    ///   <item><b>T3 conditioning:</b> <paramref name="refSpeakerEmbed"/> is the 256-d voice-encoder
    ///         embedding (<see cref="ChatterboxConfig.SpeakerEmbedDim"/>) that conditions the T3 LM.</item>
    ///   <item><b>Flow conditioning (precomputed mode):</b> pass <paramref name="flowSpeakerEmbed"/>
    ///         (<c>[1, 192]</c>) and leave <paramref name="referenceMel"/> null.</item>
    ///   <item><b>Flow conditioning (zero-shot mode):</b> pass <paramref name="referenceMel"/>
    ///         (<c>[1, 80, T]</c>); the pipeline derives the CAM++ flow embedding (and prompt speech
    ///         tokens via the S3 tokenizer) from it. Requires the encoders to be attached.</item>
    /// </list></summary>
    public float[] Synthesize(IBackend backend,
        ReadOnlySpan<int> textTokens,
        Tensor refSpeakerEmbed,
        float exaggeration = 0.5f,
        int seed = 0,
        Tensor? flowSpeakerEmbed = null,
        Tensor? referenceMel = null,
        Action<GenerationProgress>? progress = null,
        ReadOnlySpan<int> t3PromptSpeechTokens = default)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(refSpeakerEmbed);
        Stopwatch sw = Stopwatch.StartNew();

        progress?.Invoke(new GenerationProgress(0, 3, 0));

        // Stage 1: T3 — text tokens + voice-encoder embedding (+ optional perceiver-resampled cond-prompt
        // speech tokens) → S3 speech tokens.
        List<int> speechTokens = _t3.GenerateSpeechTokens(backend, textTokens, refSpeakerEmbed,
            exaggeration, _cfg.MaxNewTokens, seed, t3PromptSpeechTokens);
        Logs.Info($"Chatterbox: T3 emitted {speechTokens.Count} speech tokens in {sw.ElapsedMilliseconds}ms.");
        progress?.Invoke(new GenerationProgress(1, 3, sw.Elapsed.TotalMilliseconds));

        // Resolve the CAM++ flow speaker embedding + optional prompt-speech tokens.
        int[] promptSpeechTokens = [];
        Tensor? promptMel = referenceMel;
        Tensor flowSpk;
        bool ownsFlowSpk = false;
        if (referenceMel is not null)
        {
            if (_speakerEncoder is null)
                throw new InvalidOperationException("Zero-shot mode requires a CamPlusSpeakerEncoder to be attached.");
            if (_s3 is not null)
                promptSpeechTokens = _s3.Forward(backend, referenceMel);
            flowSpk = _speakerEncoder.Forward(backend, referenceMel);
            ownsFlowSpk = true;
        }
        else if (flowSpeakerEmbed is not null)
        {
            flowSpk = flowSpeakerEmbed;
        }
        else
        {
            throw new ArgumentException("Provide either flowSpeakerEmbed (precomputed) or referenceMel (zero-shot).");
        }

        // Stage 2: S3Gen flow — speech tokens → mel (CosyVoice-identical conditional flow matching).
        Tensor mel = _flow.Inference(backend, speechTokens.ToArray(), promptSpeechTokens, promptMel, flowSpk, seed);
        if (ownsFlowSpk) flowSpk.Dispose();
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
