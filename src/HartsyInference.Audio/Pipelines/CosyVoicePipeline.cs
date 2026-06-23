using System.Diagnostics;
using HartsyInference.Audio.Io;
using HartsyInference.Audio.Models.CosyVoice;
using HartsyInference.Audio.Preprocessing;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;

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
            float[] audio16k = ResampleTo(referenceAudio, referenceSampleRate, 16_000);
            float[] audio24k = ResampleTo(referenceAudio, referenceSampleRate, 24_000);

            // S3 speech tokenizer: Whisper-style 128-bin log-mel @16 kHz (center=True → reflect-pad n_fft/2).
            MelSpectrogramExtractor s3Mel = new(MelSpectrogramExtractor.WhisperConfig(128));
            float[] a16Centered = ReflectPad(audio16k, s3Mel.Configuration.NFft / 2);
            using (Tensor s3Feat = ChannelMajorTensor(s3Mel.Compute(a16Centered)))   // [1, 128, T]
                promptSpeechTokens = _s3.Forward(backend, s3Feat);

            // CAM++ speaker encoder: Kaldi fbank @16 kHz (snip_edges, no padding) + cepstral mean normalization.
            KaldiFbankExtractor fbank = new(16_000, 80);
            using (Tensor spkFeat = TimeMajorTensorWithCmn(fbank.Compute(audio16k)))  // [1, T, 80]
            {
                spk = _speakerEncoder.Forward(backend, spkFeat);
                ownsSpk = true;
            }

            // Flow reference conditioning: matcha 80-bin mel @24 kHz (center=False → reflect-pad (n_fft-hop)/2).
            MelSpectrogramExtractor flowMelExt = new(MelSpectrogramExtractor.CosyVoice2FlowConfig());
            MelSpectrogramExtractor.Config fc = flowMelExt.Configuration;
            float[] a24Centered = ReflectPad(audio24k, (fc.NFft - fc.HopLength) / 2);
            promptMel = ChannelMajorTensor(flowMelExt.Compute(a24Centered));          // [1, 80, T]

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

    private static float[] ResampleTo(ReadOnlySpan<float> audio, int inRate, int outRate)
        => inRate == outRate ? audio.ToArray() : Resampler.Create(inRate, outRate).Resample(audio);

    /// <summary>PyTorch <c>reflect</c> padding (reflects around the edge sample without repeating it), used to
    /// emulate the reference STFT's center padding.</summary>
    private static float[] ReflectPad(ReadOnlySpan<float> x, int pad)
    {
        if (pad <= 0) return x.ToArray();
        int n = x.Length;
        if (n <= pad) return x.ToArray(); // too short to reflect; skip padding rather than throw
        float[] y = new float[n + 2 * pad];
        for (int i = 0; i < pad; i++) y[i] = x[pad - i];
        for (int i = 0; i < n; i++) y[pad + i] = x[i];
        for (int i = 0; i < pad; i++) y[pad + n + i] = x[n - 2 - i];
        return y;
    }

    /// <summary>Wraps a <c>[channels, frames]</c> feature as a channel-major tensor <c>[1, channels, frames]</c>.</summary>
    private static Tensor ChannelMajorTensor(float[,] feat)
    {
        int c = feat.GetLength(0), f = feat.GetLength(1);
        Tensor t = new(new TensorShape(1, c, f), DType.F32);
        Span<float> d = t.AsSpan<float>();
        for (int i = 0; i < c; i++)
            for (int j = 0; j < f; j++)
                d[i * f + j] = feat[i, j];
        return t;
    }

    /// <summary>Wraps a <c>[frames, bins]</c> fbank as a time-major tensor <c>[1, frames, bins]</c> after
    /// per-bin cepstral mean normalization (subtract each bin's mean over time), as CosyVoice does before CAM++.</summary>
    private static Tensor TimeMajorTensorWithCmn(float[,] fbank)
    {
        int frames = fbank.GetLength(0), bins = fbank.GetLength(1);
        Tensor t = new(new TensorShape(1, frames, bins), DType.F32);
        Span<float> d = t.AsSpan<float>();
        for (int m = 0; m < bins; m++)
        {
            float mean = 0f;
            for (int x = 0; x < frames; x++) mean += fbank[x, m];
            mean /= Math.Max(frames, 1);
            for (int x = 0; x < frames; x++) d[x * bins + m] = fbank[x, m] - mean;
        }
        return t;
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
