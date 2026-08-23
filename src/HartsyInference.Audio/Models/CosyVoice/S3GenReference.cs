using HartsyInference.Audio.Io;
using HartsyInference.Audio.Preprocessing;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.CosyVoice;

/// <summary>Reference-clip front-ends for S3Gen-style conditioning (CosyVoice 2's zero-shot frontend and Chatterbox's <c>embed_ref</c>). Each input encoder was trained on a DIFFERENT acoustic feature — a 128-bin Whisper log-mel @16 kHz for the S3 speech tokenizer, an 80-bin Kaldi fbank (+CMN) @16 kHz for CAM++, and an 80-bin matcha mel @24 kHz for the flow's reference conditioning — so they are computed independently rather than reusing one mel. Shared by <see cref="Pipelines.CosyVoicePipeline"/> and <see cref="Pipelines.ChatterboxPipeline"/>.</summary>
public static class S3GenReference
{
    /// <summary>Polyphase-resamples <paramref name="audio"/>; pass-through copy when the rates match.</summary>
    public static float[] Resample(ReadOnlySpan<float> audio, int inRate, int outRate)
        => inRate == outRate ? audio.ToArray() : Resampler.Create(inRate, outRate).Resample(audio);

    /// <summary>Tokenizes a 16 kHz reference into 25 Hz S3 speech tokens via the Whisper-style 128-bin log-mel front-end (center=True → reflect-pad n_fft/2).</summary>
    public static int[] SpeechTokens(IBackend backend, S3Tokenizer s3, ReadOnlySpan<float> audio16k)
    {
        MelSpectrogramExtractor s3Mel = new(MelSpectrogramExtractor.WhisperConfig(128));
        float[] centered = ReflectPad(audio16k, s3Mel.Configuration.NFft / 2);
        using Tensor feat = ChannelMajorTensor(s3Mel.Compute(centered));            // [1, 128, T]
        return s3.Forward(backend, feat);
    }

    /// <summary>Derives the CAM++ x-vector <c>[1, 192]</c> from a 16 kHz reference via Kaldi fbank (snip_edges, no padding) + per-bin cepstral mean normalization.</summary>
    public static Tensor SpeakerEmbedding(IBackend backend, CamPlusSpeakerEncoder speakerEncoder, ReadOnlySpan<float> audio16k)
    {
        KaldiFbankExtractor fbank = new(16_000, 80);
        using Tensor feat = TimeMajorTensorWithCmn(fbank.Compute(audio16k));        // [1, T, 80]
        return speakerEncoder.Forward(backend, feat);
    }

    /// <summary>Computes the flow's reference-conditioning mel <c>[1, 80, T]</c> from a 24 kHz reference (matcha mel, center=False → reflect-pad (n_fft-hop)/2).</summary>
    public static Tensor FlowMel(ReadOnlySpan<float> audio24k)
    {
        MelSpectrogramExtractor flowMelExt = new(MelSpectrogramExtractor.CosyVoice2FlowConfig());
        MelSpectrogramExtractor.Config fc = flowMelExt.Configuration;
        float[] centered = ReflectPad(audio24k, (fc.NFft - fc.HopLength) / 2);
        return ChannelMajorTensor(flowMelExt.Compute(centered));
    }

    /// <summary>Reflect padding that leaves a reference shorter than the pad width unpadded — the flow mel
    /// tolerates a short reference, but periodically reflecting one would fabricate content.</summary>
    private static float[] ReflectPad(ReadOnlySpan<float> x, int pad)
        => pad <= 0 || x.Length <= pad ? x.ToArray() : SignalPadding.Reflect(x, pad);

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

    /// <summary>Wraps a <c>[frames, bins]</c> fbank as a time-major tensor <c>[1, frames, bins]</c> after per-bin cepstral mean normalization (subtract each bin's mean over time), as CosyVoice does before CAM++.</summary>
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
}
