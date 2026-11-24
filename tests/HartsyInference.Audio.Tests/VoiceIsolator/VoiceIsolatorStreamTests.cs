using HartsyInference.Audio.Models.VoiceIsolator;
using HartsyInference.Core.Backends.Cpu;
using Xunit;

namespace HartsyInference.Audio.Tests.VoiceIsolator;

public sealed class VoiceIsolatorStreamTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "voice-isolator-stream-" + Guid.NewGuid().ToString("N"));
    private readonly VoiceIsolatorNetwork _network;
    private readonly CpuBackend _backend = new();

    public VoiceIsolatorStreamTests()
    {
        Directory.CreateDirectory(_dir);
        _network = VoiceIsolatorNetwork.Load(TinyVoiceIsolatorCheckpoint.Write(_dir, causal: true, seed: 21));
    }

    public void Dispose()
    {
        _network.Dispose();
        _backend.Dispose();
        Directory.Delete(_dir, recursive: true);
    }

    [Theory]
    [InlineData(160)]
    [InlineData(1)]
    [InlineData(777)]
    public void APlainStreamEqualsTheWholeClipForward(int piece)
    {
        float[] audio = Signal(2_000, 8);
        float[] whole = _network.Isolate(_backend, audio);
        float[] streamed = Run(audio, piece, windowFrames: null, crossfadeFrames: null, cacheFrames: null);
        Assert.Equal(whole.Length, streamed.Length);
        Assert.True(MaxDiff(whole, streamed) < 2e-5f, $"max abs difference {MaxDiff(whole, streamed)}");
    }

    [Fact]
    public void ASmallCacheCompactsItsKeysAndStillMatches()
    {
        float[] audio = Signal(3_200, 3);
        float[] whole = _network.Isolate(_backend, audio);
        float[] streamed = Run(audio, 400, windowFrames: null, crossfadeFrames: null, cacheFrames: 6);
        Assert.True(MaxDiff(whole, streamed) < 5e-5f, $"max abs difference {MaxDiff(whole, streamed)}");
    }

    [Fact]
    public void WindowedLanesSwapAtEveryBoundaryAndDoNotDependOnThePushes()
    {
        float[] audio = Signal(4_800, 5);
        float[] a = Run(audio, 16, 40, 12, 40, out VoiceIsolatorStream? first);
        Assert.Equal(new[] { 40, 80, 120, 160, 200 }, first!.Boundaries.Take(5).ToArray());
        float[] b = Run(audio, 500, 40, 12, 40, out _);
        float[] c = Run(audio, 1, 40, 12, 40, out _);
        Assert.Equal(audio.Length, a.Length);
        Assert.True(MaxDiff(a, b) < 2e-5f, $"16 vs 500: {MaxDiff(a, b)}");
        Assert.True(MaxDiff(a, c) < 2e-5f, $"16 vs 1: {MaxDiff(a, c)}");
        float[] plain = Run(audio, 16, null, null, 40);
        int identicalUntil = (40 - 12 - 2) * TinyVoiceIsolatorCheckpoint.Hop - TinyVoiceIsolatorCheckpoint.NFft;
        Assert.True(MaxDiff(a.AsSpan(0, identicalUntil).ToArray(), plain.AsSpan(0, identicalUntil).ToArray()) < 2e-5f,
            "before the first crossfade the windowed stream is the plain stream");
        Assert.True(MaxDiff(a, plain) > 1e-6f, "the lanes must change the output once they blend");
    }

    [Fact]
    public void LatencyIsTheWindowPlusTheLookahead()
    {
        using VoiceIsolatorStream stream = new(_network, _backend);
        Assert.Equal(TinyVoiceIsolatorCheckpoint.NFft + TinyVoiceIsolatorCheckpoint.Hop * 2, stream.LatencySamples);
        float[] first = stream.Push(new float[TinyVoiceIsolatorCheckpoint.NFft + TinyVoiceIsolatorCheckpoint.Hop]);
        Assert.True(first.Length <= TinyVoiceIsolatorCheckpoint.Hop, "nothing beyond the look-ahead may be released early");
    }

    private float[] Run(float[] audio, int piece, int? windowFrames, int? crossfadeFrames, int? cacheFrames) =>
        Run(audio, piece, windowFrames, crossfadeFrames, cacheFrames, out _);

    private float[] Run(float[] audio, int piece, int? windowFrames, int? crossfadeFrames, int? cacheFrames, out VoiceIsolatorStream? stream)
    {
        stream = new VoiceIsolatorStream(_network, _backend, blockFrames: 4, cacheFrames, windowFrames, crossfadeFrames);
        List<float> output = [];
        for (int i = 0; i < audio.Length; i += piece)
        {
            output.AddRange(stream.Push(audio.AsSpan(i, Math.Min(piece, audio.Length - i))));
        }
        output.AddRange(stream.Flush());
        return [.. output];
    }

    private static float MaxDiff(float[] a, float[] b)
    {
        float max = 0f;
        for (int i = 0; i < a.Length; i++)
        {
            max = Math.Max(max, Math.Abs(a[i] - b[i]));
        }
        return max;
    }

    private static float[] Signal(int n, int seed)
    {
        Random rng = new(seed);
        float[] x = new float[n];
        for (int i = 0; i < n; i++)
        {
            x[i] = 0.3f * MathF.Sin(i * 0.05f) + 0.1f * ((float)rng.NextDouble() - 0.5f);
        }
        return x;
    }
}
