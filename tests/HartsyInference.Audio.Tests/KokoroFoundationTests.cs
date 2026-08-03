using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Models.Kokoro;
using HartsyInference.Core.Tensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Smoke tests for the Kokoro foundation pieces — config, phoneme tokenizer,
/// and voice pack loader — against the actual cached <c>hexgrad/Kokoro-82M</c>
/// checkpoint. These tests skip cleanly when the cache is empty.
///
/// <para>They do not load model weights or run inference; they only verify that the
/// pieces wire to the real on-disk artifacts correctly.</para></summary>
public sealed class KokoroFoundationTests
{
    private readonly ITestOutputHelper _out;
    public KokoroFoundationTests(ITestOutputHelper output) => _out = output;

    private static string KokoroRepo => AudioModelCache.GetRepoDirectory("hexgrad/Kokoro-82M", "tts");
    private static string ConfigPath => Path.Combine(KokoroRepo, "config.json");
    private static string VoicePath => Path.Combine(KokoroRepo, "voices", "af_heart.bin");

    private static bool CacheReady() => File.Exists(ConfigPath) && File.Exists(VoicePath);

    [Fact]
    public void Config_V1_DefaultsAreSensible()
    {
        KokoroConfig cfg = KokoroConfig.V1;
        Assert.Equal(178, cfg.NumTokens);
        Assert.Equal(128, cfg.StyleDim);
        Assert.Equal(512, cfg.HiddenDim);
        Assert.Equal(50, cfg.MaxDuration);
        Assert.Equal(24_000, cfg.SampleRate);
        Assert.Equal(12, cfg.PlBert.NumHiddenLayers);
        Assert.Equal(768, cfg.PlBert.HiddenSize);
        Assert.Equal([10, 6], cfg.IStftNet.UpsampleRates);
        Assert.Equal([20, 12], cfg.IStftNet.UpsampleKernelSizes);
    }

    [Fact]
    public void Tokenizer_LoadsFromCacheAndEncodes()
    {
        if (!CacheReady())
        {
            _out.WriteLine("Kokoro cache missing — skipping.");
            return;
        }
        KokoroPhonemeTokenizer tok = KokoroPhonemeTokenizer.LoadFromConfig(ConfigPath);
        Assert.Equal(178, tok.VocabSize);

        // Known characters from the printed vocab: 'h'=50, 'ɛ'=86, 'l'=54, 'o'=57
        // ' '=16. "hello" → 50, 86, 54, 54, 57; surrounded by [0, ..., 0].
        int[] ids = tok.Encode("hɛllo");
        Assert.Equal([0, 50, 86, 54, 54, 57, 0], ids);
        _out.WriteLine($"Encoded 'hɛllo' → [{string.Join(", ", ids)}]");

        // Unknown chars are silently dropped — verify by mixing IPA + ASCII not in vocab.
        int[] dropped = tok.Encode("hɛllo•");     // '•' is not in vocab
        Assert.Equal(7, dropped.Length);     // same as without it
    }

    [Fact]
    public unsafe void VoicePack_LoadsAndYieldsSplitStyleVectors()
    {
        if (!CacheReady())
        {
            _out.WriteLine("Kokoro cache missing — skipping.");
            return;
        }

        using KokoroVoicePack pack = KokoroVoicePack.LoadFromFile(VoicePath);
        _out.WriteLine($"Voice pack '{pack.Name}' has {pack.NumBuckets} buckets.");
        Assert.True(pack.NumBuckets > 100, $"Voice pack too small ({pack.NumBuckets} buckets).");

        // Select the style for a typical token count, split into decoder + predictor halves.
        using Tensor style = pack.GetStyle(tokenCount: 30);
        Assert.Equal(new TensorShape(1, 256), style.Shape);

        // Style values should not be all zero — they're trained vectors.
        float* sp = (float*)style.DataPointer;
        float maxAbs = 0f;
        for (int i = 0; i < 256; i++) maxAbs = MathF.Max(maxAbs, MathF.Abs(sp[i]));
        Assert.True(maxAbs > 0.01f, $"Voice pack row appears empty (maxAbs={maxAbs}).");
        _out.WriteLine($"Style row maxAbs = {maxAbs:F4}");

        (Tensor dec, Tensor pred) = KokoroVoicePack.SplitStyle(style);
        try
        {
            Assert.Equal(new TensorShape(1, 128), dec.Shape);
            Assert.Equal(new TensorShape(1, 128), pred.Shape);

            // Decoder + predictor halves should be DIFFERENT — the two halves serve
            // different roles and Kokoro voice packs train them independently.
            float* dp = (float*)dec.DataPointer;
            float* pp = (float*)pred.DataPointer;
            int distinct = 0;
            for (int i = 0; i < 128; i++) if (MathF.Abs(dp[i] - pp[i]) > 1e-6f) distinct++;
            Assert.True(distinct > 50, $"Decoder/predictor halves too similar ({distinct} differ).");
        }
        finally { dec.Dispose(); pred.Dispose(); }
    }

    [Fact]
    public void VoicePack_OutOfRangeTokenCountClamps()
    {
        if (!CacheReady())
        {
            _out.WriteLine("Kokoro cache missing — skipping.");
            return;
        }
        using KokoroVoicePack pack = KokoroVoicePack.LoadFromFile(VoicePath);

        // tokenCount=0 → idx clamped to 0 (a valid row).
        using Tensor first = pack.GetStyle(tokenCount: 0);
        Assert.Equal(new TensorShape(1, 256), first.Shape);

        // tokenCount way past the bucket count → idx clamped to NumBuckets-1.
        using Tensor last = pack.GetStyle(tokenCount: 10_000);
        Assert.Equal(new TensorShape(1, 256), last.Shape);
    }
}
