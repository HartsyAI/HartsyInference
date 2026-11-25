using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Rope;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The MovieRoll inference-regime window of block 0 on the tiny DiT checkpoint, against the <c>win.*</c> tensors that
/// <c>tests/python-reference/movieroll_reference/dump_hy15_tiny.py</c> saves from MovieRoll's <c>forward_inference</c> and
/// <c>parallel_attention</c> (torch attention). Window A is encoded at packed frames 0..2 and its K and V are the cache; window C is
/// denoised at frames 3..5 over that cache. A second cache, captured for window B at frames 6..8, is re-packed to 0..2 for the last check,
/// so a cache that stored post-RoPE keys would fail there. The window geometry is 3 frames of 2×3 tokens.</summary>
public sealed class MovieRollWindowTests
{
    private const float Tolerance = 1e-5f;
    private const int Hidden = 32;
    private const int Heads = 2;
    private const int HeadDim = Hidden / Heads;
    private const int Blocks = 2;
    private const int Frames = 3;
    private const int Height = 2;
    private const int Width = 3;
    private const int PerFrame = Height * Width;
    private const int WindowTokens = Frames * PerFrame;
    private const int FullHistoryBlocks = 1;   // block 0 full history, block 1 sink-recent (the tiny model has two blocks)

    private static HunyuanVideo15Config TinyConfig() => new()
    {
        NumBlocks = Blocks, HiddenSize = Hidden, NumHeads = 2, LatentChannels = 4, TextDim = 24, ByT5Dim = 16, RopeAxesDim = [4, 6, 6],
    };

    [Fact]
    public void EncodedWindowKeysAndValuesMatchTheReference()
    {
        using Fixture fixture = new();
        using Tensor frameTemb = fixture.FrameTemb();
        using Tensor txtTemb = fixture.TxtTemb();
        using TokenSegmentLayout layout = fixture.Layout();
        using MovieRollWindowOutput encoded = fixture.Block.ForwardWindow(fixture.Backend, fixture.Ref("win.in.img_a"), null,
            txtTemb, frameTemb, layout, fixture.Rope, Rows(0, Frames), null, null, returnKv: true);

        Assert.Null(encoded.Text);
        AssertValues(fixture.Ref("win.encode.img"), encoded.Image, "encoded image");
        AssertValues(fixture.Ref("win.encode.k"), encoded.Keys!, "encoded keys");
        AssertValues(fixture.Ref("win.encode.v"), encoded.Values!, "encoded values");
    }

    [Fact]
    public void EncodeOverANonEmptyCacheMatchesTheReference()
    {
        using Fixture fixture = new();
        using Tensor frameTemb = fixture.FrameTemb();
        using Tensor txtTemb = fixture.TxtTemb();
        using TokenSegmentLayout layout = fixture.Layout();
        using Tensor keys = Copy(fixture.Ref("win.encode.k"), new TensorShape(1, WindowTokens, Heads, HeadDim));
        using Tensor values = Copy(fixture.Ref("win.encode.v"), new TensorShape(1, WindowTokens, Heads, HeadDim));
        using MovieRollWindowOutput encoded = fixture.Block.ForwardWindow(fixture.Backend, fixture.Ref("win.in.img_c"), null,
            txtTemb, frameTemb, layout, fixture.Rope, Concat(Rows(0, Frames), Rows(Frames, Frames)), keys, values, returnKv: true);

        AssertValues(fixture.Ref("win.encode_cached.img"), encoded.Image, "encoded image over cache");
        AssertValues(fixture.Ref("win.encode_cached.k"), encoded.Keys!, "encoded keys over cache");
        AssertValues(fixture.Ref("win.encode_cached.v"), encoded.Values!, "encoded values over cache");
    }

    [Fact]
    public void DenoisedWindowOverItsCacheMatchesTheReference()
    {
        using Fixture fixture = new();
        using MovieRollWindowOutput denoised = fixture.Denoise(fixture.Ref("win.encode.k"), fixture.Ref("win.encode.v"));

        AssertValues(fixture.Ref("win.denoise.img"), denoised.Image, "denoised image");
        AssertValues(fixture.Ref("win.denoise.txt"), denoised.Text!, "denoised text");
    }

    [Fact]
    public void CachedHistoryRotatesAtItsPackedPositionNotItsCapturedOne()
    {
        using Fixture fixture = new();
        using MovieRollWindowOutput denoised = fixture.Denoise(fixture.Ref("win.encode_b.k"), fixture.Ref("win.encode_b.v"));

        AssertValues(fixture.Ref("win.repos.img"), denoised.Image, "re-packed image");
        AssertValues(fixture.Ref("win.repos.txt"), denoised.Text!, "re-packed text");
    }

    [Fact]
    public void RotateRowsMatchesTheReferenceTables()
    {
        using Fixture fixture = new();
        using Tensor queries = Filled(WindowTokens, 1f);
        fixture.Rope.RotateRows(fixture.Backend, queries, Rows(Frames, Frames), WindowTokens, 2);
        AssertValues(fixture.Ref("win.table.rot_ones_q"), queries, "rotated queries");

        using Tensor keys = Filled(2 * WindowTokens, 1f);
        fixture.Rope.RotateRows(fixture.Backend, keys, Concat(Rows(0, Frames), Rows(Frames, Frames)), 2 * WindowTokens, 2);
        AssertValues(fixture.Ref("win.table.rot_ones_k"), keys, "rotated keys");
    }

    [Fact]
    public void EmptyCacheAtGridPositionsIsThePerFrameForward()
    {
        using Fixture fixture = new();
        using Tensor frameTemb = fixture.FrameTemb();
        using Tensor txtTemb = fixture.TxtTemb();
        using TokenSegmentLayout layout = fixture.Layout();
        Tensor image = fixture.Ref("win.in.img_c");
        Tensor text = fixture.Ref("win.in.txt");
        using MovieRollWindowOutput windowed = fixture.Block.ForwardWindow(fixture.Backend, image, text, txtTemb, frameTemb, layout,
            fixture.Rope, Rows(0, Frames), null, null, returnKv: false);
        (Tensor perFrameImage, Tensor perFrameText) = fixture.Block.ForwardPerFrame(fixture.Backend, image, text, txtTemb, frameTemb,
            layout, fixture.Rope, Height, Width, Frames);
        using (perFrameImage)
        using (perFrameText)
        {
            AssertValues(perFrameImage, windowed.Image, "window image at grid positions");
            AssertValues(perFrameText, windowed.Text!, "window text at grid positions");
        }
    }

    [Fact]
    public void PositionsMustCoverTheCacheAndTheWindow()
    {
        using Fixture fixture = new();
        using Tensor frameTemb = fixture.FrameTemb();
        using Tensor txtTemb = fixture.TxtTemb();
        using Tensor cacheKeys = Copy(fixture.Ref("win.encode.k"), new TensorShape(1, WindowTokens, Heads, HeadDim));
        using Tensor cacheValues = Copy(fixture.Ref("win.encode.v"), new TensorShape(1, WindowTokens, Heads, HeadDim));
        Assert.Throws<ArgumentException>(() => fixture.Block.ForwardWindow(fixture.Backend, fixture.Ref("win.in.img_c"), fixture.Ref("win.in.txt"),
            txtTemb, frameTemb, fixture.Layout(), fixture.Rope, Rows(3, Frames), cacheKeys, cacheValues, returnKv: false));
    }

    [Fact]
    public void FourWindowStreamingRolloutMatchesTheReference()
    {
        using Fixture fixture = new();
        using MovieRollKvCache cache = new(fixture.Backend);
        cache.Init(fixture.Dit.StreamLayerNames(), Height, Width);
        float[] timesteps = [700f, 537f, 300f];
        Tensor qwen = fixture.Ref("in.qwen");
        Tensor byt5 = fixture.Ref("in.byt5");
        float windowTimestep = fixture.Ref("in.timestep").AsReadOnlySpan<float>()[0];

        for (int window = 0; window < 4; window++)
        {
            Tensor latent = fixture.Ref($"win.stream.latent{window}");
            using Tensor velocity = fixture.Dit.StreamDenoise(fixture.Backend, latent, timesteps, windowTimestep, qwen, byt5, cache, FullHistoryBlocks);
            AssertValues(fixture.Ref($"win.stream.velocity{window}"), velocity, $"window {window} velocity");
            fixture.Dit.StreamEncode(fixture.Backend, latent, timesteps, cache, FullHistoryBlocks, startFrame: Frames * window);
        }
    }

    [Fact]
    public void RecomputeRebuildsEveryLayerFromTheSelectedChunks()
    {
        using Fixture fixture = new();
        using MovieRollKvCache cache = new(fixture.Backend);
        cache.Init(fixture.Dit.StreamLayerNames(), Height, Width);

        int cached = fixture.Dit.StreamRecompute(fixture.Backend, fixture.Ref("rec.latent"), [1, 2, 2], cache, FullHistoryBlocks);

        Assert.Equal(5, cached);
        string full = HunyuanVideo15Dit.StreamLayerName(0);
        Assert.Equal(new[] { (0, 1), (1, 3), (3, 5) }, cache.FrameRanges(full).ToArray());
        (Tensor Keys, Tensor Values)? fullCache = cache.Get(full);
        AssertValues(fixture.Ref("rec.full0.k"), fullCache!.Value.Keys, "full-history keys after recompute");
        AssertValues(fixture.Ref("rec.full0.v"), fullCache.Value.Values, "full-history values after recompute");

        string sinkRecent = HunyuanVideo15Dit.StreamLayerName(1);
        Assert.Equal(new[] { (0, 1), (3, 5) }, cache.FrameRanges(sinkRecent).ToArray());
        (Tensor Keys, Tensor Values)? srCache = cache.Get(sinkRecent);
        AssertValues(fixture.Ref("rec.sink1.k"), srCache!.Value.Keys, "sink-recent keys after recompute");
        AssertValues(fixture.Ref("rec.sink1.v"), srCache.Value.Values, "sink-recent values after recompute");
    }

    [Fact]
    public void StreamRefusesALatentThatDoesNotMatchTheCacheGrid()
    {
        using Fixture fixture = new();
        using MovieRollKvCache cache = new(fixture.Backend);
        cache.Init(fixture.Dit.StreamLayerNames(), Height + 1, Width);

        Assert.Throws<ArgumentException>(() => fixture.Dit.StreamEncode(fixture.Backend, fixture.Ref("win.stream.latent0"), [700f, 537f, 300f],
            cache, FullHistoryBlocks, startFrame: 0));
    }

    [Fact]
    public void CacheRefusesAChunkThatStartsBeforeTheLastCachedChunkEnds()
    {
        using MovieRollKvCache cache = new(new CpuBackend());
        cache.Init(["layer"], Height, Width);
        using Tensor first = new(new TensorShape(1, 2 * PerFrame, Heads, HeadDim), DType.F32);
        using Tensor overlap = new(new TensorShape(1, PerFrame, Heads, HeadDim), DType.F32);
        cache.Update("layer", first, first, 0, 2);

        Assert.Throws<ArgumentOutOfRangeException>(() => cache.Update("layer", overlap, overlap, 1, 2));
    }

    [Fact]
    public void RecomputeWithEveryBlockFullHistoryReturnsTheClipAndPacksEveryLayer()
    {
        using Fixture fixture = new();
        using MovieRollKvCache cache = new(fixture.Backend);
        cache.Init(fixture.Dit.StreamLayerNames(), Height, Width);

        int cached = fixture.Dit.StreamRecompute(fixture.Backend, fixture.Ref("rec.latent"), [1, 2, 2], cache, fixture.Dit.BlockCount);

        Assert.Equal(5, cached);
        foreach (string layer in fixture.Dit.StreamLayerNames())
            Assert.Equal(new[] { (0, 1), (1, 3), (3, 5) }, cache.FrameRanges(layer).ToArray());
    }

    [Fact]
    public void RecomputeOfASingleChunkKeepsOneRangePerLayer()
    {
        using Fixture fixture = new();
        using MovieRollKvCache cache = new(fixture.Backend);
        cache.Init(fixture.Dit.StreamLayerNames(), Height, Width);

        int cached = fixture.Dit.StreamRecompute(fixture.Backend, fixture.Ref("rec.latent"), [5], cache, FullHistoryBlocks);

        Assert.Equal(5, cached);
        foreach (string layer in fixture.Dit.StreamLayerNames())
            Assert.Equal(new[] { (0, 5) }, cache.FrameRanges(layer).ToArray());
    }

    [Fact]
    public void RecomputeRefusesChunksThatDoNotCoverTheLatent()
    {
        using Fixture fixture = new();
        using MovieRollKvCache cache = new(fixture.Backend);
        cache.Init(fixture.Dit.StreamLayerNames(), Height, Width);
        Tensor latent = fixture.Ref("rec.latent");

        Assert.Throws<ArgumentException>(() => fixture.Dit.StreamRecompute(fixture.Backend, latent, [1, 2], cache, FullHistoryBlocks));
        Assert.Throws<ArgumentException>(() => fixture.Dit.StreamRecompute(fixture.Backend, latent, [0, 5], cache, FullHistoryBlocks));
    }

    [Fact]
    public void CacheRefusesAWriteOverItsByteBudgetAndKeepsItsState()
    {
        const long OneFrameKeyAndValue = 2L * PerFrame * Heads * HeadDim * sizeof(float);
        using MovieRollKvCache cache = new(new CpuBackend(), OneFrameKeyAndValue);
        cache.Init(["layer"], Height, Width);
        using Tensor first = new(new TensorShape(1, PerFrame, Heads, HeadDim), DType.F32);
        using Tensor second = new(new TensorShape(1, PerFrame, Heads, HeadDim), DType.F32);
        cache.Update("layer", first, first, 0, 1);

        Assert.Throws<InvalidOperationException>(() => cache.Update("layer", second, second, 1, 2));
        Assert.Equal(new[] { (0, 1) }, cache.FrameRanges("layer").ToArray());
        Assert.Equal(OneFrameKeyAndValue, cache.BytesHeld);
    }

    [Fact]
    public void BytesPerFrameIsTheAuditEstimateAt480p()
    {
        // 480p latent grid 30 × 52 = 1,560 tokens per frame; 54 blocks; hidden 2,048; F32 K and V.
        Assert.Equal(1_380_188_160L, MovieRollKvCache.BytesPerFrame(54, 30, 52, 2048));
    }

    [Fact]
    public void RecomputeRefusedByTheBudgetLeavesTheCacheAsItWas()
    {
        using Fixture fixture = new();
        const long OneFrameKeyAndValue = 2L * PerFrame * Heads * HeadDim * sizeof(float);
        using MovieRollKvCache cache = new(fixture.Backend, OneFrameKeyAndValue);
        cache.Init(fixture.Dit.StreamLayerNames(), Height, Width);
        using Tensor first = new(new TensorShape(1, PerFrame, Heads, HeadDim), DType.F32);
        cache.Update(HunyuanVideo15Dit.StreamLayerName(0), first, first, 0, 1);

        Assert.Throws<InvalidOperationException>(() => fixture.Dit.StreamRecompute(fixture.Backend, fixture.Ref("rec.latent"), new[] { 1, 2, 2 },
            cache, FullHistoryBlocks));
        Assert.Equal(new[] { (0, 1) }, cache.FrameRanges(HunyuanVideo15Dit.StreamLayerName(0)).ToArray());
    }

    /// <summary>Rows of rope-axis coordinates (frame, row, column) for the packed window of <paramref name="frames"/> frames from
    /// <paramref name="firstFrame"/>, in the token order the window uses.</summary>
    private static float[] Rows(int firstFrame, int frames)
    {
        float[] rows = new float[frames * PerFrame * 3];
        int token = 0;
        for (int f = 0; f < frames; f++)
            for (int h = 0; h < Height; h++)
                for (int w = 0; w < Width; w++, token++)
                {
                    rows[token * 3] = firstFrame + f;
                    rows[token * 3 + 1] = h;
                    rows[token * 3 + 2] = w;
                }
        return rows;
    }

    private static float[] Concat(float[] head, float[] tail) => [.. head, .. tail];

    private static Tensor Filled(int tokens, float value)
    {
        Tensor tensor = new(new TensorShape(1, tokens, Hidden), DType.F32);
        tensor.AsSpan<float>().Fill(value);
        return tensor;
    }

    private static Tensor Copy(Tensor source, TensorShape shape)
    {
        Tensor copy = new(shape, DType.F32);
        source.AsReadOnlySpan<float>().CopyTo(copy.AsSpan<float>());
        return copy;
    }

    private static void AssertValues(Tensor expected, Tensor actual, string label)
    {
        ReadOnlySpan<float> e = expected.AsReadOnlySpan<float>();
        ReadOnlySpan<float> a = actual.AsReadOnlySpan<float>();
        Assert.Equal(e.Length, a.Length);
        Assert.NotEqual(0, e.Length);
        float max = 0f;
        for (int i = 0; i < e.Length; i++) max = Math.Max(max, Math.Abs(e[i] - a[i]));
        Assert.True(max <= Tolerance, $"{label}: max abs error {max:E3} exceeds {Tolerance:E0}.");
    }

    private sealed class Fixture : IDisposable
    {
        public Fixture()
        {
            Loader = new SafeTensorsLoader();
            Loader.Load(Path.Combine(AppContext.BaseDirectory, "Fixtures", "hy15_dit_tiny.safetensors"));
            Weights = Loader.GetAllTensors()
                .Where(pair => !pair.Key.StartsWith("ref.", StringComparison.Ordinal) && !pair.Key.StartsWith("in.", StringComparison.Ordinal)
                    && !pair.Key.StartsWith("win.", StringComparison.Ordinal) && !pair.Key.StartsWith("rec.", StringComparison.Ordinal))
                .ToDictionary(pair => pair.Key, pair => pair.Value);
            Backend = new CpuBackend();
            Dit = new HunyuanVideo15Dit(TinyConfig());
            Dit.LoadWeights(HunyuanVideo15CheckpointConverter.Convert(Weights, Blocks, Hidden));
            Block = (HunyuanImageBlock)Dit.GetBlock(0);
            Rope = new HunyuanImageRope([4, 6, 6], 256f);
        }

        public SafeTensorsLoader Loader { get; }
        public Dictionary<string, Tensor> Weights { get; }
        public IBackend Backend { get; }
        public HunyuanVideo15Dit Dit { get; }
        public HunyuanImageBlock Block { get; }
        public HunyuanImageRope Rope { get; }

        public Tensor Ref(string name) => Loader.GetTensor(name);

        /// <summary>The window's per-frame timestep embedding, <c>[frames, hidden]</c>.</summary>
        public Tensor FrameTemb() => Copy(Ref("win.in.vec"), new TensorShape(Frames, Hidden));

        /// <summary>The window's text timestep embedding, <c>[1, hidden]</c>.</summary>
        public Tensor TxtTemb() => Copy(Ref("win.in.vec_txt"), new TensorShape(1, Hidden));

        public TokenSegmentLayout Layout() =>
            new(Enumerable.Repeat(PerFrame, Frames).ToArray(), Enumerable.Range(0, Frames).ToArray());

        /// <summary>Window C denoised at frames 3..5 over <paramref name="cacheKeys"/> and <paramref name="cacheValues"/> (packed at 0..2).</summary>
        public MovieRollWindowOutput Denoise(Tensor cacheKeys, Tensor cacheValues)
        {
            using Tensor frameTemb = FrameTemb();
            using Tensor txtTemb = TxtTemb();
            using TokenSegmentLayout layout = Layout();
            using Tensor keys = Copy(cacheKeys, new TensorShape(1, WindowTokens, Heads, HeadDim));
            using Tensor values = Copy(cacheValues, new TensorShape(1, WindowTokens, Heads, HeadDim));
            return Block.ForwardWindow(Backend, Ref("win.in.img_c"), Ref("win.in.txt"), txtTemb, frameTemb, layout, Rope,
                Concat(Rows(0, Frames), Rows(Frames, Frames)), keys, values, returnKv: false);
        }

        public void Dispose()
        {
            Rope.Dispose();
            Dit.Dispose();
            Backend.Dispose();
            Loader.Dispose();
        }
    }
}
