using HartsyInference.Audio.Models.VibeVoice;
using HartsyInference.Core.Tensors;
using HartsyInference.Core.Backends.Cpu;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests.VibeVoice;

/// <summary>Regression net for the shared VibeVoice acoustic tokenizer (1.5B, 7B, Realtime and Haru all run it): a tiny random
/// codec decoded and encoded whole and frame by frame must agree, and its output is pinned so a later change to the shared
/// convolution or streaming-cache code cannot move it unnoticed.</summary>
public sealed unsafe class VibeVoiceCodecStreamingTests(ITestOutputHelper output)
{
    private const string Prefix = "acoustic_tokenizer";
    private const int Frames = 6;

    private static VibeVoiceAcousticTokenizerModel Load()
    {
        VibeVoiceAcousticTokenizerModel model = new(VibeVoiceSyntheticWeights.TinyConfig, Prefix);
        model.LoadWeights(VibeVoiceSyntheticWeights.Create(VibeVoiceSyntheticWeights.TinyConfig, Prefix, seed: 11));
        return model;
    }

    private static Tensor Fill(int seed, params long[] dims)
    {
        Tensor t = new(new TensorShape(dims), DType.F32);
        Random rng = new(seed);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++)
        {
            p[i] = (float)(rng.NextDouble() * 2 - 1);
        }
        return t;
    }

    private static float[] Copy(Tensor t) => new ReadOnlySpan<float>((void*)t.DataPointer, (int)t.ElementCount).ToArray();

    private static float MaxAbsDiff(float[] a, float[] b)
    {
        Assert.Equal(a.Length, b.Length);
        float worst = 0;
        for (int i = 0; i < a.Length; i++)
        {
            worst = MathF.Max(worst, MathF.Abs(a[i] - b[i]));
        }
        return worst;
    }

    [Fact]
    public void Decode_FrameByFrameStreaming_MatchesWholeSequence()
    {
        VibeVoiceAcousticTokenizerModel model = Load();
        using CpuBackend backend = new();
        using Tensor latents = Fill(5, 1, Frames, VibeVoiceSyntheticWeights.TinyConfig.VaeDim);
        using Tensor whole = model.Decode(backend, latents, 1);
        float[] want = Copy(whole);
        Assert.Equal(Frames * VibeVoiceSyntheticWeights.TinyHop, want.Length);

        using VibeVoiceTokenizerStreamingCache cache = new();
        List<float> got = [];
        int dim = VibeVoiceSyntheticWeights.TinyConfig.VaeDim;
        for (int f = 0; f < Frames; f++)
        {
            using Tensor one = new(new TensorShape(1, 1, dim), DType.F32);
            Buffer.MemoryCopy((float*)latents.DataPointer + f * dim, (void*)one.DataPointer, dim * 4, dim * 4);
            using Tensor chunk = model.Decode(backend, one, 1, cache, [0]);
            got.AddRange(Copy(chunk));
        }
        float worst = MaxAbsDiff(want, [.. got]);
        output.WriteLine($"decode whole vs streaming max |diff| = {worst:E3}");
        Assert.True(worst < 1e-4f, $"max |diff| = {worst}");
    }

    [Fact]
    public void Encode_FrameByFrameStreaming_MatchesWholeClip()
    {
        VibeVoiceAcousticTokenizerModel model = Load();
        using CpuBackend backend = new();
        int hop = VibeVoiceSyntheticWeights.TinyHop;
        using Tensor pcm = Fill(6, 1, 1, Frames * hop);
        using Tensor whole = model.Encode(backend, pcm, 1, Frames * hop);
        float[] want = Copy(whole);

        using VibeVoiceTokenizerStreamingCache cache = new();
        List<float> got = [];
        for (int f = 0; f < Frames; f++)
        {
            using Tensor one = new(new TensorShape(1, 1, hop), DType.F32);
            Buffer.MemoryCopy((float*)pcm.DataPointer + f * hop, (void*)one.DataPointer, hop * 4, hop * 4);
            using Tensor chunk = model.Encode(backend, one, 1, hop, cache, [0]);
            got.AddRange(Copy(chunk));
        }
        float worst = MaxAbsDiff(want, [.. got]);
        output.WriteLine($"encode whole vs streaming max |diff| = {worst:E3}");
        Assert.True(worst < 1e-4f, $"max |diff| = {worst}");
    }

    [Fact]
    public void Decode_FixedLatents_MatchesPinnedGolden()
    {
        VibeVoiceAcousticTokenizerModel model = Load();
        using CpuBackend backend = new();
        using Tensor latents = Fill(5, 1, Frames, VibeVoiceSyntheticWeights.TinyConfig.VaeDim);
        using Tensor pcm = model.Decode(backend, latents, 1);
        float[] audio = Copy(pcm);
        double sum = 0;
        double energy = 0;
        foreach (float v in audio)
        {
            sum += v;
            energy += (double)v * v;
        }
        output.WriteLine($"decode: n={audio.Length} sum={sum:R} energy={energy:R} first={audio[0]:R} last={audio[^1]:R}");
        // Captured on unmodified origin/main (58f248e00), CPU F32. A shared-code change must keep these within 1e-3 relative.
        Assert.Equal(576, audio.Length);
        Assert.Equal(21.970818347326713, sum, 1e-3 * 21.97);
        Assert.Equal(8.205365254869216, energy, 1e-3 * 8.205);
        Assert.Equal(0.013818156f, audio[0], 1e-4f);
        Assert.Equal(0.30657238f, audio[^1], 1e-4f);
    }
}
