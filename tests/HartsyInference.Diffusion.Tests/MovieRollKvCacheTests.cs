using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The MovieRoll streaming KV cache on the CPU backend: appends, the HieraKV sink trim, the recompute trigger, soft clear and replace.
/// The reference's <c>KVCacheManager</c> has no reference dump of its own; these tests pin the same bookkeeping by hand (chunk frames, token
/// counts, which tokens survive a trim).</summary>
public sealed class MovieRollKvCacheTests
{
    private const int Heads = 2, HeadDim = 3, Height = 2, Width = 2, TokensPerFrame = Height * Width;

    private static Tensor Chunk(int frames, float value, int heads = Heads)
    {
        Tensor t = new(new TensorShape([1L, frames * TokensPerFrame, heads, HeadDim]), DType.F32);
        t.AsSpan<float>().Fill(value);
        return t;
    }

    private static MovieRollKvCache Initialised(params string[] layers)
    {
        MovieRollKvCache cache = new();
        cache.Init(layers, Height, Width);
        return cache;
    }

    [Fact]
    public void UpdateAppendsChunksAndTracksTheirFrames()
    {
        using MovieRollKvCache cache = Initialised("block0");
        IBackend backend = new CpuBackend();
        using Tensor k1 = Chunk(2, 1f);
        using Tensor v1 = Chunk(2, 1f);
        using Tensor k2 = Chunk(1, 2f);
        using Tensor v2 = Chunk(1, 2f);
        cache.Update(backend, "block0", k1, v1, 0, 2);
        cache.Update(backend, "block0", k2, v2, 2, 3);

        (Tensor keys, _) = cache.Get("block0")!.Value;
        Assert.Equal(3 * TokensPerFrame, keys.Shape[1]);
        ReadOnlySpan<float> s = keys.AsReadOnlySpan<float>();
        Assert.Equal(1f, s[0]);
        Assert.Equal(2f, s[2 * TokensPerFrame * Heads * HeadDim]);
        Assert.Equal(new[] { 2, 1 }, cache.CachedChunkSizes("block0").ToArray());
        Assert.Equal(new[] { (0, 2), (2, 3) }, cache.FrameRanges("block0").ToArray());
    }

    [Fact]
    public void TrimToSinkKeepsOnlyTheSinkChunks()
    {
        using MovieRollKvCache cache = Initialised("block0");
        IBackend backend = new CpuBackend();
        for (int c = 0; c < 3; c++)
        {
            using Tensor k = Chunk(1, c + 1f);
            using Tensor v = Chunk(1, c + 1f);
            cache.Update(backend, "block0", k, v, c, c + 1);
        }

        cache.TrimToSink(backend, "block0", 1);

        (Tensor keys, Tensor values) = cache.Get("block0")!.Value;
        Assert.Equal(TokensPerFrame, keys.Shape[1]);
        Assert.Equal(TokensPerFrame, values.Shape[1]);
        Assert.Equal(1f, keys.AsReadOnlySpan<float>()[0]);
        Assert.Equal(new[] { (0, 1) }, cache.FrameRanges("block0").ToArray());
        Assert.True(cache.ShouldRecompute(1));
    }

    [Fact]
    public void TrimToSinkLeavesALayerWithNoMoreChunksAlone()
    {
        using MovieRollKvCache cache = Initialised("block0");
        IBackend backend = new CpuBackend();
        using Tensor k = Chunk(1, 5f);
        using Tensor v = Chunk(1, 5f);
        cache.Update(backend, "block0", k, v, 0, 1);

        cache.TrimToSink(backend, "block0", 1);

        Assert.Equal(new[] { 1 }, cache.CachedChunkSizes("block0").ToArray());
        Assert.Equal(TokensPerFrame, cache.Get("block0")!.Value.Keys.Shape[1]);
    }

    [Fact]
    public void UpdateRejectsATokenCountThatDoesNotMatchItsFrames()
    {
        using MovieRollKvCache cache = Initialised("block0");
        IBackend backend = new CpuBackend();
        using Tensor k = Chunk(2, 1f);
        using Tensor v = Chunk(2, 1f);
        Assert.Throws<ArgumentException>(() => cache.Update(backend, "block0", k, v, 0, 1));
    }

    [Fact]
    public void UpdateRejectsAHeadCountChangeBetweenChunks()
    {
        using MovieRollKvCache cache = Initialised("block0");
        IBackend backend = new CpuBackend();
        using Tensor k1 = Chunk(1, 1f);
        using Tensor v1 = Chunk(1, 1f);
        cache.Update(backend, "block0", k1, v1, 0, 1);
        using Tensor k2 = Chunk(1, 2f, heads: Heads + 1);
        using Tensor v2 = Chunk(1, 2f, heads: Heads + 1);
        Assert.Throws<ArgumentException>(() => cache.Update(backend, "block0", k2, v2, 1, 2));
    }

    [Fact]
    public void UseBeforeInitThrows()
    {
        using MovieRollKvCache cache = new();
        IBackend backend = new CpuBackend();
        using Tensor k = Chunk(1, 1f);
        using Tensor v = Chunk(1, 1f);
        Assert.Throws<InvalidOperationException>(() => cache.Update(backend, "block0", k, v, 0, 1));
    }

    [Fact]
    public void ShouldRecomputeCountsTheFirstLayerOnly()
    {
        using MovieRollKvCache cache = Initialised("first", "second");
        IBackend backend = new CpuBackend();
        using Tensor k = Chunk(1, 1f);
        using Tensor v = Chunk(1, 1f);
        cache.Update(backend, "first", k, v, 0, 1);
        cache.Update(backend, "second", k, v, 0, 1);
        cache.Update(backend, "second", k, v, 1, 2);

        Assert.True(cache.ShouldRecompute(1));
        Assert.False(cache.ShouldRecompute(2));
        Assert.Equal("first", cache.FirstLayer);
    }

    [Fact]
    public void SoftClearKeepsTheLayersAndTheGrid()
    {
        using MovieRollKvCache cache = Initialised("block0");
        IBackend backend = new CpuBackend();
        using Tensor k = Chunk(1, 1f);
        using Tensor v = Chunk(1, 1f);
        cache.Update(backend, "block0", k, v, 0, 1);

        cache.SoftClear();

        Assert.True(cache.IsInit);
        Assert.Null(cache.Get("block0"));
        Assert.Empty(cache.FrameRanges("block0"));
        cache.Update(backend, "block0", k, v, 0, 1);
        Assert.Equal(new[] { 1 }, cache.CachedChunkSizes("block0").ToArray());
    }

    [Fact]
    public void ReplaceLayerCopiesTheArraysAndSetsTheFrames()
    {
        using MovieRollKvCache cache = Initialised("block0");
        IBackend backend = new CpuBackend();
        using Tensor keys = Chunk(2, 7f);
        using Tensor values = Chunk(2, 7f);

        cache.ReplaceLayer(backend, "block0", keys, values, [(0, 2)]);
        keys.AsSpan<float>().Fill(9f);

        (Tensor stored, _) = cache.Get("block0")!.Value;
        Assert.Equal(7f, stored.AsReadOnlySpan<float>()[0]);
        Assert.Equal(new[] { 2 }, cache.CachedChunkSizes("block0").ToArray());
    }

    [Fact]
    public void ReplaceLayerRejectsFramesThatDoNotMatchTheTokens()
    {
        using MovieRollKvCache cache = Initialised("block0");
        IBackend backend = new CpuBackend();
        using Tensor keys = Chunk(2, 1f);
        using Tensor values = Chunk(2, 1f);
        Assert.Throws<ArgumentException>(() => cache.ReplaceLayer(backend, "block0", keys, values, [(0, 1)]));
    }

    [Fact]
    public void DisposeIsIdempotent()
    {
        MovieRollKvCache cache = Initialised("block0");
        cache.Dispose();
        cache.Dispose();
    }
}
