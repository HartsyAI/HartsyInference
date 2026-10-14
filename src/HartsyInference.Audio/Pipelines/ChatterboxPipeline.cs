using System.Diagnostics;
using HartsyInference.Audio.Io;
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
    private const int S3GenSampleRate = 24_000;
    private const int S3TokenizerSampleRate = 16_000;

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

    /// <summary>The CAM++ encoder zero-shot mode needs: upstream S3Gen runs <c>CAMPPlus</c> in PyTorch, so a partial
    /// context segment divides by the frames it covers, not by CosyVoice's ONNX divisor.</summary>
    public static CamPlusSpeakerEncoder CreateSpeakerEncoder(int embedDim) => new(embedDim, onnxSegmentPooling: false);

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
    public float[] Synthesize(IBackend backend, ReadOnlySpan<int> textTokens, Tensor? refSpeakerEmbed,
        float exaggeration = 0.5f, int seed = 0, Tensor? flowSpeakerEmbed = null,
        ReadOnlySpan<float> referenceAudio = default, int referenceSampleRate = 0,
        Action<GenerationProgress>? progress = null, ReadOnlySpan<int> t3PromptSpeechTokens = default,
        float cfgWeight = 0f, ReadOnlySpan<int> flowPromptTokens = default, Tensor? flowPromptMel = null)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(backend);
        Stopwatch sw = Stopwatch.StartNew();

        progress?.Invoke(new GenerationProgress(0, 3, 0));

        // Resolve conditioning: either derived from the raw reference clip (zero-shot) or passed in (precomputed).
        int[] promptSpeechTokens = flowPromptTokens.ToArray();
        ReadOnlySpan<int> t3Prompt = t3PromptSpeechTokens;
        Tensor? promptMel = null;
        if ((!flowPromptTokens.IsEmpty || flowPromptMel is not null) && !referenceAudio.IsEmpty)
            throw new ArgumentException("A precomputed flow prompt and referenceAudio are mutually exclusive.", nameof(flowPromptTokens));
        if (!flowPromptTokens.IsEmpty && (flowPromptMel is null || flowPromptMel.Shape[2] != 2L * flowPromptTokens.Length))
            throw new ArgumentException("flowPromptMel must be [1, 80, 2 * flowPromptTokens.Length] when prompt tokens are given.", nameof(flowPromptMel));
        Tensor? veEmbed = refSpeakerEmbed;
        Tensor? flowSpk = flowSpeakerEmbed;
        bool ownsVe = false, ownsFlowSpk = false;
        if (!referenceAudio.IsEmpty)
        {
            if (referenceSampleRate <= 0)
                throw new ArgumentException("referenceSampleRate must be set when referenceAudio is provided.", nameof(referenceSampleRate));
            if (_speakerEncoder is null || _s3 is null)
                throw new InvalidOperationException("Zero-shot mode requires a CamPlusSpeakerEncoder and an S3Tokenizer to be attached.");

            // prepare_conditionals: the reference is loaded at 24 kHz; the voice encoder and T3 prompt tokenization see
            // a full-length 16 kHz copy, S3Gen's embed_ref the first 10 s at 24 kHz.
            float[] ref24 = S3GenReference.Resample(referenceAudio, referenceSampleRate, S3GenSampleRate);
            float[] ref16Full = ResampleFullReference(ref24);

            if (veEmbed is null)
            {
                if (_voiceEncoder is null)
                    throw new InvalidOperationException("Zero-shot mode requires a ChatterboxVoiceEncoder (or a precomputed refSpeakerEmbed).");
                veEmbed = _voiceEncoder.EmbedUtterance(backend, ref16Full);
                ownsVe = true;
            }
            if (t3Prompt.IsEmpty) t3Prompt = T3CondPromptTokens(backend, ref16Full);

            (promptSpeechTokens, promptMel, flowSpk) = EmbedReference(backend,
                ref24.AsSpan(0, Math.Min(ref24.Length, _cfg.S3GenCondSeconds * S3GenSampleRate)));
            ownsFlowSpk = true;
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
        int[] flowTokens = DropInvalidSpeechTokens(speechTokens, _cfg.StartSpeechToken, _cfg.S3CodebookSize);
        Tensor mel = _flow.Inference(backend, flowTokens, promptSpeechTokens, promptMel ?? flowPromptMel, flowSpk, seed);
        if (ownsFlowSpk) flowSpk.Dispose();
        promptMel?.Dispose();
        double flowDoneMs = sw.Elapsed.TotalMilliseconds;
        progress?.Invoke(new GenerationProgress(2, 3, flowDoneMs));

        // Stage 3: HiFTNet vocoder — mel → 24 kHz waveform.
        float[] audio = _vocoder.Forward(backend, mel);
        mel.Dispose();
        ApplyTrimFade(audio, _cfg.SampleRate);
        Logs.Debug($"Chatterbox: HiFT vocoder in {sw.Elapsed.TotalMilliseconds - flowDoneMs:F1}ms.");

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

    /// <summary>The full-length 16 kHz reference the voice encoder and the T3 prompt tokens see. Upstream makes it with
    /// librosa's soxr_hq, which has no port here; torchaudio's filter (<see cref="SincResampler"/>), the one upstream's
    /// own <c>embed_ref</c> uses, stands in. The two flip a share of the prompt tokens upstream too.</summary>
    internal static float[] ResampleFullReference(ReadOnlySpan<float> ref24) =>
        SincResampler.Resample(ref24, S3GenSampleRate, S3TokenizerSampleRate);

    /// <summary>T3's cond-prompt speech tokens: the first <c>ENC_COND_LEN</c> (6 s) of the 16 kHz reference, at most
    /// <c>speech_cond_prompt_len</c> tokens.</summary>
    internal int[] T3CondPromptTokens(IBackend backend, ReadOnlySpan<float> ref16)
    {
        if (_s3 is null) throw new InvalidOperationException("T3 prompt tokens need an S3Tokenizer.");
        int[] tokens = S3GenReference.SpeechTokens(backend, _s3, ref16[..Math.Min(ref16.Length, _cfg.T3CondSeconds * S3TokenizerSampleRate)]);
        return tokens.Length > _cfg.SpeechCondPromptLen ? tokens[.._cfg.SpeechCondPromptLen] : tokens;
    }

    /// <summary>S3Gen <c>embed_ref</c> on a 24 kHz reference: the flow prompt mel from the 24 kHz audio, then torchaudio's
    /// 16 kHz copy for the CAM++ x-vector and the prompt speech tokens. When the mel is not twice the token count the
    /// tokens are cut to <c>mel_len // 2</c> and the mel is kept whole, as upstream does.</summary>
    internal (int[] Tokens, Tensor Mel, Tensor Speaker) EmbedReference(IBackend backend, ReadOnlySpan<float> ref24)
    {
        if (_speakerEncoder is null || _s3 is null)
            throw new InvalidOperationException("embed_ref needs a CamPlusSpeakerEncoder and an S3Tokenizer.");
        float[] ref16 = SincResampler.Resample(ref24, S3GenSampleRate, S3TokenizerSampleRate);
        Tensor mel = S3GenReference.FlowMel(ref24);
        Tensor speaker = S3GenReference.SpeakerEmbedding(backend, _speakerEncoder, ref16);
        int[] tokens = S3GenReference.SpeechTokens(backend, _s3, ref16);
        int half = (int)mel.Shape[2] / 2;
        return (tokens.Length > half ? tokens[..half] : tokens, mel, speaker);
    }

    /// <summary>Upstream S3Gen's <c>trim_fade</c> on the head of every output: <c>sr / 50</c> samples zeroed, then a raised-cosine
    /// fade-in over the next <c>sr / 50</c> (<c>(cos(linspace(π, 0, n)) + 1) / 2</c>), which hides reference spill-over
    /// at the prompt boundary.</summary>
    internal static void ApplyTrimFade(Span<float> audio, int sampleRate)
    {
        int n = sampleRate / 50;
        int zeroed = Math.Min(n, audio.Length);
        audio[..zeroed].Clear();
        for (int i = 0; i < n && n + i < audio.Length; i++)
        {
            double t = n > 1 ? Math.PI - Math.PI * i / (n - 1) : 0.0;
            audio[n + i] *= (float)((Math.Cos(t) + 1.0) / 2.0);
        }
    }

    /// <summary>Upstream tts.py runs <c>drop_invalid_tokens</c> (everything up to a start-of-speech token goes) and then
    /// keeps only real S3 codes (<c>speech_tokens &lt; 6561</c>) before S3Gen.</summary>
    internal static int[] DropInvalidSpeechTokens(List<int> tokens, int startSpeechToken, int codebookSize)
    {
        int start = tokens.IndexOf(startSpeechToken) + 1;
        List<int> kept = new(tokens.Count - start);
        for (int i = start; i < tokens.Count; i++)
        {
            if ((uint)tokens[i] < (uint)codebookSize) kept.Add(tokens[i]);
        }
        return kept.ToArray();
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
