using System.Diagnostics;
using HartsyInference.Audio.Models.CosyVoice;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.Audio.Pipelines;

/// <summary>CosyVoice 2 (`FunAudioLLM/CosyVoice2-0.5B`) non-streaming text-to-speech pipeline. Wires the
/// four-stage path: Qwen2.5-0.5B text→speech-token LM → chunk-aware conditional flow matching
/// (speech-token→mel) → HiFTNet vocoder (mel→24 kHz). For zero-shot cloning the reference clip's mel is
/// run through the S3 tokenizer (prompt speech tokens) and CAM++ (192-d speaker embedding); for
/// preset-voice / precomputed-embedding modes those are passed in directly and the input-side encoders
/// can be omitted.
///
/// <para>Phoneme/text tokenization is the caller's responsibility — <see cref="Synthesize"/> takes Qwen
/// BPE token IDs, matching the token-IDs-in convention of the other HartsyInference audio pipelines.
/// Streaming (the 5:15 text:speech interleave + per-chunk flush) is a follow-up; this pipeline runs the
/// non-streaming "all text, then synthesize" format.</para></summary>
public sealed class CosyVoicePipeline : IDisposable
{
    private readonly CosyVoiceConfig _cfg;
    private readonly CosyVoiceQwenLm _lm;
    private readonly CosyVoiceFlow _flow;
    private readonly HiFTNetVocoder _vocoder;
    private readonly CamPlusSpeakerEncoder? _speakerEncoder;
    private readonly S3Tokenizer? _s3;
    private bool _preloaded;
    private int _disposed;

    public CosyVoicePipeline(CosyVoiceConfig cfg, CosyVoiceQwenLm lm, CosyVoiceFlow flow,
        HiFTNetVocoder vocoder, CamPlusSpeakerEncoder? speakerEncoder = null, S3Tokenizer? s3 = null)
    {
        _cfg = cfg;
        _lm = lm;
        _flow = flow;
        _vocoder = vocoder;
        _speakerEncoder = speakerEncoder;
        _s3 = s3;
    }

    /// <summary>Synthesizes 24 kHz audio for the given Qwen text token IDs.
    /// <list type="bullet">
    ///   <item><b>Precomputed-speaker mode:</b> pass <paramref name="speakerEmbed"/> (<c>[1, 192]</c>);
    ///         leave <paramref name="referenceAudio"/> empty.</item>
    ///   <item><b>Zero-shot mode:</b> pass the raw <paramref name="referenceAudio"/> samples at
    ///         <paramref name="referenceSampleRate"/> + <paramref name="referenceTextTokens"/>. The pipeline
    ///         derives the THREE distinct features each input encoder needs — a 128-bin Whisper log-mel @16 kHz
    ///         for the S3 speech tokenizer, an 80-bin Kaldi fbank (+CMN) @16 kHz for CAM++, and an 80-bin
    ///         matcha mel @24 kHz for the flow's reference conditioning — rather than reusing one mel for all
    ///         three. Requires both input encoders to be attached.</item>
    /// </list></summary>
    public float[] Synthesize(IBackend backend,
        int[] textTokenIds,
        Tensor? speakerEmbed = null,
        ReadOnlySpan<float> referenceAudio = default,
        int referenceSampleRate = 0,
        int[]? referenceTextTokens = null,
        int seed = 0)
    {
        ThrowIfDisposed();
        Stopwatch sw = Stopwatch.StartNew();

        // Full-F32 GEMM across the whole pipeline. The Qwen speech-token LM is autoregressive, and TF32's ~1e-3
        // per-forward error accumulates over the decode loop and flips sampled argmaxes (verified on Zonos: TF32 →
        // babble, F32 → reference parity). Save/restore so we don't disturb the caller's backend state.
        // A layer-split LM runs each stage on its own backend, so every placement stage backend needs it too.
        List<(IBackend Backend, bool Prev)> saved = new(4) { (backend, backend.HighPrecisionGemm) };
        backend.HighPrecisionGemm = true;
        if (_lm.Placement is { IsSingle: false } lmPlacement)
        {
            foreach (LlmStage stage in lmPlacement.Stages)
            {
                IBackend sb = stage.Backend;
                bool seen = false;
                for (int i = 0; i < saved.Count && !seen; i++) seen = ReferenceEquals(saved[i].Backend, sb);
                if (seen) continue;
                saved.Add((sb, sb.HighPrecisionGemm));
                sb.HighPrecisionGemm = true;
            }
        }
        try
        {
            PreloadWeights(backend);
            return SynthesizeCore(backend, textTokenIds, speakerEmbed, referenceAudio, referenceSampleRate,
                referenceTextTokens, seed, sw);
        }
        finally
        {
            for (int i = saved.Count - 1; i >= 0; i--) saved[i].Backend.HighPrecisionGemm = saved[i].Prev;
        }
    }

    /// <summary>Bulk-uploads every component's weights to the device once (idempotent — the GPU weight cache keys by
    /// tensor reference). Without this each Linear/Conv re-uploads its weight per call: the pre-preload profile was
    /// ~62k <c>H2D_MISS</c> transfers (~3.7 s) dominating the run. No-op on backends without a weight cache. The LM
    /// gets a per-stage asymmetric preload when <see cref="CosyVoiceQwenLm.Placement"/> is a layer-split (mirrors
    /// <c>YuePipeline.PreloadStage1</c>) — preloading the full set on every stage would replicate instead of pool.</summary>
    private void PreloadWeights(IBackend backend)
    {
        if (_preloaded) return;
        if (_lm.Placement is { IsSingle: false } placement)
        {
            for (int s = 0; s < placement.Stages.Count; s++)
            {
                LlmStage stage = placement.Stages[s];
                // Canonical set only (no redundant fused/split views) — matches YuE's resident-preload choice.
                stage.Backend.PreloadWeights(_lm.EnumerateStageWeights(
                    stage.StartLayer, stage.EndLayer, s == 0, s == placement.Stages.Count - 1,
                    includeRedundantSplits: false));
            }
            backend.PreloadWeights(EnumerateNonLmWeights());
        }
        else
        {
            backend.PreloadWeights(EnumerateWeights());
        }
        _preloaded = true;
    }

    private IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _lm.EnumerateWeights()) yield return t;
        foreach (Tensor t in EnumerateNonLmWeights()) yield return t;
    }

    private IEnumerable<Tensor> EnumerateNonLmWeights()
    {
        foreach (Tensor t in _flow.EnumerateWeights()) yield return t;
        foreach (Tensor t in _vocoder.EnumerateWeights()) yield return t;
        if (_speakerEncoder is not null)
            foreach (Tensor t in _speakerEncoder.EnumerateWeights()) yield return t;
        if (_s3 is not null)
            foreach (Tensor t in _s3.EnumerateWeights()) yield return t;
    }

    private float[] SynthesizeCore(IBackend backend,
        int[] textTokenIds,
        Tensor? speakerEmbed,
        ReadOnlySpan<float> referenceAudio,
        int referenceSampleRate,
        int[]? referenceTextTokens,
        int seed,
        Stopwatch sw)
    {
        int[] promptSpeechTokens = [];
        Tensor? promptMel = null;
        Tensor spk;
        bool ownsSpk = false;

        if (!referenceAudio.IsEmpty)
        {
            if (referenceSampleRate <= 0)
                throw new ArgumentException("referenceSampleRate must be set when referenceAudio is provided.", nameof(referenceSampleRate));
            if (_s3 is null || _speakerEncoder is null)
                throw new InvalidOperationException("Zero-shot mode requires both an S3Tokenizer and a CamPlusSpeakerEncoder.");

            // Each input encoder was trained on a different acoustic feature; compute them independently.
            float[] audio16k = S3GenReference.Resample(referenceAudio, referenceSampleRate, 16_000);
            float[] audio24k = S3GenReference.Resample(referenceAudio, referenceSampleRate, 24_000);

            promptSpeechTokens = S3GenReference.SpeechTokens(backend, _s3, audio16k);
            spk = S3GenReference.SpeakerEmbedding(backend, _speakerEncoder, audio16k);
            ownsSpk = true;
            promptMel = S3GenReference.FlowMel(audio24k);                             // [1, 80, T]

            Logs.Info($"CosyVoice: reference → {promptSpeechTokens.Length} prompt speech tokens + speaker embedding.");
        }
        else if (speakerEmbed is not null)
        {
            spk = speakerEmbed;
        }
        else
        {
            throw new ArgumentException("Provide either speakerEmbed (precomputed) or referenceAudio (zero-shot).");
        }

        int[] refText = referenceTextTokens ?? [];
        List<int> speechTokens = _lm.GenerateSpeechTokens(backend, textTokenIds, refText, promptSpeechTokens, seed: seed);
        Logs.Info($"CosyVoice: LM emitted {speechTokens.Count} speech tokens in {sw.ElapsedMilliseconds}ms.");

        Tensor mel = _flow.Inference(backend, speechTokens.ToArray(), promptSpeechTokens, promptMel, spk, seed);
        if (ownsSpk) spk.Dispose();
        promptMel?.Dispose();

        float[] audio = _vocoder.Forward(backend, mel);
        mel.Dispose();

        sw.Stop();
        Logs.Info($"CosyVoice synthesis complete: {audio.Length} samples ({audio.Length / (double)_cfg.SampleRate:F2}s) in {sw.ElapsedMilliseconds}ms.");
        return audio;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _lm.Dispose();
        _flow.Dispose();
        _vocoder.Dispose();
        _speakerEncoder?.Dispose();
        _s3?.Dispose();
        GC.SuppressFinalize(this);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed != 0) throw new ObjectDisposedException(nameof(CosyVoicePipeline));
    }
}
