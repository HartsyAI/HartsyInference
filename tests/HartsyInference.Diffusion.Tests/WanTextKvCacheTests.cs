using HartsyInference.Core.Backends;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Diffusion.Models.Denoisers.UNetBlocks;
using HartsyInference.Tests.Common;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Wan's text cross-attention K/V cache: with it, cond and uncond alternating over the steps reuse their K/V and
/// reproduce the uncached forwards bit for bit, and the transformer's scope is the cache's whole lifetime.</summary>
public sealed unsafe class WanTextKvCacheTests
{
    private const int ContextTokens = 5;

    private static WanVideoConfig TinyConfig => new()
    {
        NumHeads = 2, HeadDim = 12, InChannels = 48, OutChannels = 48,
        TextDim = 16, FreqDim = 16, FfnDim = 32, NumLayers = 2, PatchSize = (1, 2, 2),
        VaeSpatialCompression = 16, VaeTemporalCompression = 4,
        NumInferenceSteps = 3, GuidanceScale = 5, FlowShift = 5,
    };

    [Fact]
    public void TheBlockReusesEachContextsKvAndMatchesTheUncachedBlock()
    {
        WanVideoConfig cfg = TinyConfig;
        WanVideoBlock block = new(cfg, crossAttnNorm: true);
        block.LoadWeights(WanSyntheticWeights.BuildTransformer(cfg), "blocks.0");
        using CpuBackend backend = new();
        WanRope rope = new(cfg.HeadDim, cfg.RopeTheta, cfg.RopeMaxSeqLen);
        (Tensor cos, Tensor sin) = rope.BuildCosSin(2, 2, 2);
        const int s = 8;
        Tensor[] contexts = [Rand([ContextTokens, cfg.InnerDim], 1), Rand([ContextTokens, cfg.InnerDim], 2)];
        CountingKv kv = new(new ConstantKvCache(contextsPerBackend: 2, pinned: true));

        for (int step = 0; step < 3; step++)
        {
            Tensor hidden = Rand([s, cfg.InnerDim], 10 + step);
            Tensor temb = Rand([1, 6, cfg.InnerDim], 20 + step);
            foreach (Tensor context in contexts)
            {
                float[] expected = Read(block.Forward(backend, hidden, context, temb, rope, cos, sin, s));
                float[] actual = Read(block.Forward(backend, hidden, context, temb, rope, cos, sin, s, textKv: kv));
                Assert.Equal(expected, actual);
            }
        }

        Assert.Equal(2, kv.Stores);
        Assert.Equal(4, kv.Hits);
    }

    [Fact]
    public void AContextRebuiltInsideTheScopeIsProjectedAgainNotServedTheOldKv()
    {
        WanVideoConfig cfg = TinyConfig;
        WanVideoBlock block = new(cfg, crossAttnNorm: true);
        block.LoadWeights(WanSyntheticWeights.BuildTransformer(cfg), "blocks.0");
        using CpuBackend backend = new();
        WanRope rope = new(cfg.HeadDim, cfg.RopeTheta, cfg.RopeMaxSeqLen);
        (Tensor cos, Tensor sin) = rope.BuildCosSin(2, 2, 2);
        const int s = 8;
        Tensor hidden = Rand([s, cfg.InnerDim], 10), temb = Rand([1, 6, cfg.InnerDim], 20);
        CountingKv kv = new(new ConstantKvCache(contextsPerBackend: 2, pinned: true));
        block.Forward(backend, hidden, Rand([ContextTokens, cfg.InnerDim], 1), temb, rope, cos, sin, s, textKv: kv);

        Tensor rebuilt = Rand([ContextTokens, cfg.InnerDim], 3);
        float[] expected = Read(block.Forward(backend, hidden, rebuilt, temb, rope, cos, sin, s));
        float[] actual = Read(block.Forward(backend, hidden, rebuilt, temb, rope, cos, sin, s, textKv: kv));

        Assert.Equal(expected, actual);
        Assert.Equal(2, kv.Stores);
        Assert.Equal(0, kv.Hits);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void TransformerForwardsInsideTheScopeMatchTheUncachedForwardsBitForBit(bool perFrameTimesteps)
    {
        WanVideoConfig cfg = TinyConfig;
        WanVideoTransformer transformer = new(cfg);
        transformer.LoadWeights(WanSyntheticWeights.BuildTransformer(cfg));
        using CpuBackend backend = new();
        Tensor latent = Rand([1, cfg.InChannels, 3, 4, 4], 7);
        Tensor cond = Rand([ContextTokens, cfg.TextDim], 1), uncond = Rand([ContextTokens, cfg.TextDim], 2);
        float[][] timesteps = perFrameTimesteps ? [[0f, 900f, 900f], [0f, 500f, 500f], [0f, 100f, 100f]] : [[900f], [500f], [100f]];

        List<float[]> expected = Steps(transformer, backend, latent, cond, uncond, timesteps);
        List<float[]> cached;
        using (transformer.CacheTextCrossAttentionKv())
            cached = Steps(transformer, backend, latent, cond, uncond, timesteps);

        Assert.Equal(expected, cached);
    }

    [Fact]
    public void TheScopeIsExclusiveAndReopensOnceDisposed()
    {
        WanVideoTransformer transformer = new(TinyConfig);
        transformer.LoadWeights(WanSyntheticWeights.BuildTransformer(TinyConfig));

        IDisposable scope = transformer.CacheTextCrossAttentionKv();
        Assert.Throws<InvalidOperationException>(() => transformer.CacheTextCrossAttentionKv());
        scope.Dispose();
        scope.Dispose();
        transformer.CacheTextCrossAttentionKv().Dispose();
    }

    private static List<float[]> Steps(WanVideoTransformer transformer, IBackend backend, Tensor latent, Tensor cond, Tensor uncond,
        float[][] timesteps)
    {
        List<float[]> outputs = [];
        foreach (float[] t in timesteps)
        {
            outputs.Add(Read(transformer.Forward(backend, latent, cond, t)));
            outputs.Add(Read(transformer.Forward(backend, latent, uncond, t)));
        }
        return outputs;
    }

    /// <summary>Counts the hits and stores of the cache it wraps.</summary>
    private sealed class CountingKv(ICachedKv inner) : ICachedKv
    {
        public int Hits { get; private set; }

        public int Stores { get; private set; }

        public bool TryGet(IBackend backend, Tensor kvInput, out Tensor keyMh, out Tensor valueMh)
        {
            bool hit = inner.TryGet(backend, kvInput, out keyMh, out valueMh);
            if (hit) Hits++;
            return hit;
        }

        public bool TryStore(IBackend backend, Tensor kvInput, Tensor keyMh, Tensor valueMh)
        {
            bool stored = inner.TryStore(backend, kvInput, keyMh, valueMh);
            if (stored) Stores++;
            return stored;
        }
    }

    private static float[] Read(Tensor t) => new ReadOnlySpan<float>(t.DataPointer, checked((int)t.ElementCount)).ToArray();

    private static Tensor Rand(long[] shape, int seed)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        float* p = (float*)t.DataPointer;
        Random rng = new(seed);
        for (long i = 0; i < t.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }
}
