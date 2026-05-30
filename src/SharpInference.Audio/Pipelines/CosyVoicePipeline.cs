using System.Diagnostics;
using SharpInference.Audio.Models.CosyVoice;
using SharpInference.Core.Backends;
using SharpInference.Core.Logging;
using SharpInference.Core.Tensors;

namespace SharpInference.Audio.Pipelines;

/// <summary>CosyVoice 2 (`FunAudioLLM/CosyVoice2-0.5B`) non-streaming text-to-speech pipeline. Wires the
/// four-stage path: Qwen2.5-0.5B text→speech-token LM → chunk-aware conditional flow matching
/// (speech-token→mel) → HiFTNet vocoder (mel→24 kHz). For zero-shot cloning the reference clip's mel is
/// run through the S3 tokenizer (prompt speech tokens) and CAM++ (192-d speaker embedding); for
/// preset-voice / precomputed-embedding modes those are passed in directly and the input-side encoders
/// can be omitted.
///
/// <para>Phoneme/text tokenization is the caller's responsibility — <see cref="Synthesize"/> takes Qwen
/// BPE token IDs, matching the token-IDs-in convention of the other SharpInference audio pipelines.
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
    ///         leave <paramref name="referenceMel"/> null.</item>
    ///   <item><b>Zero-shot mode:</b> pass <paramref name="referenceMel"/> (<c>[1, 80, T]</c>) +
    ///         <paramref name="referenceTextTokens"/>; the pipeline derives the prompt speech tokens
    ///         (S3) and speaker embedding (CAM++). Requires the encoders to be attached.</item>
    /// </list></summary>
    public float[] Synthesize(IBackend backend,
        int[] textTokenIds,
        Tensor? speakerEmbed = null,
        Tensor? referenceMel = null,
        int[]? referenceTextTokens = null,
        int seed = 0)
    {
        ThrowIfDisposed();
        Stopwatch sw = Stopwatch.StartNew();

        int[] promptSpeechTokens = [];
        Tensor? promptMel = referenceMel;
        Tensor spk;
        bool ownsSpk = false;

        if (referenceMel is not null)
        {
            if (_s3 is null || _speakerEncoder is null)
                throw new InvalidOperationException("Zero-shot mode requires both an S3Tokenizer and a CamPlusSpeakerEncoder.");
            promptSpeechTokens = _s3.Forward(backend, referenceMel);
            spk = _speakerEncoder.Forward(backend, referenceMel);
            ownsSpk = true;
            Logs.Info($"CosyVoice: reference → {promptSpeechTokens.Length} prompt speech tokens + speaker embedding.");
        }
        else if (speakerEmbed is not null)
        {
            spk = speakerEmbed;
        }
        else
        {
            throw new ArgumentException("Provide either speakerEmbed (precomputed) or referenceMel (zero-shot).");
        }

        int[] refText = referenceTextTokens ?? [];
        List<int> speechTokens = _lm.GenerateSpeechTokens(backend, textTokenIds, refText, promptSpeechTokens, seed: seed);
        Logs.Info($"CosyVoice: LM emitted {speechTokens.Count} speech tokens in {sw.ElapsedMilliseconds}ms.");

        Tensor mel = _flow.Inference(backend, speechTokens.ToArray(), promptSpeechTokens, promptMel, spk, seed);
        if (ownsSpk) spk.Dispose();

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
