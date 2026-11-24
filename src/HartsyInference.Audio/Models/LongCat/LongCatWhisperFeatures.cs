using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Audio.Preprocessing;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.LongCat;

/// <summary>Avatar 1.5/1.6 speech features (<c>get_audio_embedding_whisper</c>): log-mel per 30 s chunk, the Whisper encoder
/// with all 33 hidden states, pooled into 5 blocks, cut to <c>2 * video_length</c> encoder frames and resampled from 50 Hz
/// to the video rate. Loudness normalisation (on for 1.5, off for 1.6) is the caller's step.</summary>
public static class LongCatWhisperFeatures
{
    /// <summary>Samples per feature-extractor call (<c>MEL_CHUNK = 750 * 640</c>): 30 s at 16 kHz, so every call pads to one full window.</summary>
    public const int MelChunkSamples = 750 * 640;

    /// <summary>Mel frames per encoder call (<c>ENC_CHUNK</c>): one Whisper window.</summary>
    public const int EncoderChunkFrames = 3000;

    private const int PoolTileFrames = 125;

    /// <summary><c>[videoFrames, 5, channels]</c> features of a 16 kHz mono clip, <c>videoFrames = int(seconds * fps)</c>.</summary>
    /// <param name="mel">A <see cref="MelSpectrogramExtractor.WhisperConfig"/> extractor with the encoder's mel-bin count.</param>
    public static float[] Extract(IBackend backend, WhisperEncoder encoder, MelSpectrogramExtractor mel, ReadOnlySpan<float> speech16k,
        double fps = 25)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(encoder);
        ArgumentNullException.ThrowIfNull(mel);
        WhisperConfig cfg = encoder.Config;
        if (cfg.EncoderLayers + 1 != LongCatAudioFeatures.WhisperStateCount)
            throw new ArgumentException($"LongCat pools 33 hidden states; this encoder has {cfg.EncoderLayers + 1}.", nameof(encoder));
        if (mel.Configuration.SampleRate != 16_000 || mel.Configuration.NMels != cfg.NumMelBins)
            throw new ArgumentException("The mel extractor must be a 16 kHz Whisper preset with the encoder's mel-bin count.", nameof(mel));
        if (!(fps > 0)) throw new ArgumentOutOfRangeException(nameof(fps), fps, "Frame rate must be positive.");

        int videoFrames = LongCatAudioFeatures.VideoFrames(speech16k.Length, 16_000, fps);
        if (videoFrames < 1) throw new ArgumentException("The clip is shorter than one video frame.", nameof(speech16k));
        int channels = cfg.HiddenSize;
        int needed = 2 * videoFrames;
        int blockValues = LongCatAudioFeatures.WhisperBlockCount * channels;
        if ((long)needed * blockValues > int.MaxValue)
            throw new ArgumentException(
                $"A {speech16k.Length / 16_000.0 / 60:0.#} minute clip needs more than 2^31 pooled values; embed it in parts.",
                nameof(speech16k));
        float[] pooled = new float[needed * blockValues];
        float[] tile = new float[PoolTileFrames * LongCatAudioFeatures.WhisperStateCount * channels];
        float[] melChunk = new float[cfg.NumMelBins * EncoderChunkFrames];

        int done = 0;
        for (int start = 0; start < speech16k.Length && done < needed; start += MelChunkSamples)
        {
            ReadOnlySpan<float> chunk = speech16k.Slice(start, Math.Min(MelChunkSamples, speech16k.Length - start));
            mel.ComputeZeroPadded(chunk, MelChunkSamples, melChunk);
            Tensor melTensor = new(new TensorShape(1, cfg.NumMelBins, EncoderChunkFrames), DType.F32);
            melChunk.CopyTo(melTensor.AsSpan<float>());
            Tensor[] states = encoder.ForwardAllHiddenStates(backend, melTensor);
            melTensor.Dispose();
            try
            {
                int frames = (int)states[0].Shape[1];
                int take = Math.Min(frames, needed - done);
                for (int t0 = 0; t0 < take; t0 += PoolTileFrames)
                {
                    int rows = Math.Min(PoolTileFrames, take - t0);
                    for (int s = 0; s < states.Length; s++)
                    {
                        ReadOnlySpan<float> state = states[s].AsReadOnlySpan<float>();
                        for (int t = 0; t < rows; t++)
                            state.Slice((t0 + t) * channels, channels).CopyTo(tile.AsSpan((t * states.Length + s) * channels, channels));
                    }
                    LongCatAudioFeatures.PoolWhisperStates(tile.AsSpan(0, rows * states.Length * channels), rows, channels,
                        pooled.AsSpan((done + t0) * blockValues, rows * blockValues));
                }
                done += take;
            }
            finally
            {
                foreach (Tensor state in states) state.Dispose();
            }
        }
        return LongCatAudioFeatures.ResampleFrames(
            pooled.AsSpan(0, done * blockValues), done, LongCatAudioFeatures.WhisperBlockCount, channels, videoFrames);
    }
}
