using HartsyInference.Audio.Preprocessing;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.QwenOmni;

/// <summary>Qwen2.5-Omni audio front-end math: the Whisper log-mel with zero padding to the 300 s window and the mel/conv/token length formulas.</summary>
public sealed class QwenOmniProcessor
{
    private readonly QwenOmniConfig _cfg;
    private readonly MelSpectrogramExtractor _mel;

    /// <summary>Builds the 128-bin Whisper extractor for <paramref name="cfg"/>.</summary>
    public QwenOmniProcessor(QwenOmniConfig cfg)
    {
        ArgumentNullException.ThrowIfNull(cfg);
        _cfg = cfg;
        _mel = new MelSpectrogramExtractor(MelSpectrogramExtractor.WhisperConfig(cfg.NumMelBins));
    }

    /// <summary>Valid mel frames of an utterance: the feature-extractor attention mask sampled at the hop, <c>ceil(samples / hop)</c>.</summary>
    public static int MelFrames(int samples, int hopLength)
    {
        if (samples <= 0) throw new ArgumentOutOfRangeException(nameof(samples), "Audio must hold at least one sample.");
        return (samples + hopLength - 1) / hopLength;
    }

    /// <summary>Frames after the stride-2 conv stem, <c>(L - 1) / 2 + 1</c>.</summary>
    public static int ConvFrames(int melFrames) => (melFrames - 1) / 2 + 1;

    /// <summary>Audio embeddings (and <c>&lt;|AUDIO|&gt;</c> tokens) after the 2x average pool, <c>(conv - 2) // 2 + 1</c> with floor division, which is <c>conv / 2</c>.</summary>
    public static int AudioTokens(int melFrames) => ConvFrames(melFrames) / 2;

    /// <summary>Audio tokens for <paramref name="samples"/> samples at the config hop.</summary>
    public int AudioTokensForSamples(int samples) => AudioTokens(MelFrames(samples, _cfg.HopLength));

    /// <summary>Frames of the padded mel returned by <see cref="ComputeMel"/>.</summary>
    public int PaddedFrames => _mel.OutputFrames(_cfg.MaxSamples);

    /// <summary>Computes the log-mel zero-padded to the 300 s window into <paramref name="mel"/> <c>[NumMelBins, PaddedFrames]</c> and returns the valid frame count (columns past it are padding).</summary>
    public int ComputeMel(ReadOnlySpan<float> audio16k, Tensor mel)
    {
        int frames = MelFrames(audio16k.Length, _cfg.HopLength);
        if (audio16k.Length > _cfg.MaxSamples)
        {
            throw new ArgumentException($"Audio of {audio16k.Length} samples exceeds the {_cfg.MaxSamples}-sample window.", nameof(audio16k));
        }
        if (mel.DType != DType.F32 || mel.Shape.Rank != 2 || (int)mel.Shape[0] != _cfg.NumMelBins || (int)mel.Shape[1] != PaddedFrames)
        {
            throw new ArgumentException($"mel must be F32 [{_cfg.NumMelBins}, {PaddedFrames}]; got {mel.Shape}.", nameof(mel));
        }
        unsafe
        {
            _mel.ComputeZeroPadded(audio16k, _cfg.MaxSamples, new Span<float>(mel.DataPointer, (int)mel.Shape.ElementCount));
        }
        return frames;
    }

    /// <summary>Overwrites the rows of <paramref name="embeds"/> <c>[ids.Length, dim]</c> at every <paramref name="audioTokenId"/> position with the next row of <paramref name="audio"/> <c>[N, dim]</c>; the placeholder count must equal N.</summary>
    public static void SpliceAudioRows(ReadOnlySpan<int> ids, int audioTokenId, Span<float> embeds, ReadOnlySpan<float> audio, int dim)
    {
        if (dim <= 0 || embeds.Length != ids.Length * dim || audio.Length % dim != 0)
        {
            throw new ArgumentException("embeds must be [ids.Length, dim] and audio a whole number of dim-wide rows.");
        }
        int rows = audio.Length / dim;
        int next = 0;
        for (int t = 0; t < ids.Length; t++)
        {
            if (ids[t] != audioTokenId) continue;
            if (next >= rows) throw new ArgumentException($"More <|AUDIO|> placeholders than the {rows} audio rows.", nameof(ids));
            audio.Slice(next * dim, dim).CopyTo(embeds.Slice(t * dim, dim));
            next++;
        }
        if (next != rows) throw new ArgumentException($"{next} <|AUDIO|> placeholders for {rows} audio rows.", nameof(ids));
    }
}
