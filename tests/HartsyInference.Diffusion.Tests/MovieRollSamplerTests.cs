using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Pipelines.MovieRoll;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The rolling sampler on the tiny DiT checkpoint. These are invariant tests, not reference parity: the rolling loop has no Python
/// transcription yet. They check the output layout, that classifier-free guidance at scale 1 reproduces the unguided run, and that the recompute
/// and distilled paths complete with a consistent cache.</summary>
public sealed class MovieRollSamplerTests
{
    private const int Channels = 4;
    private const int Height = 2;
    private const int Width = 3;
    private const int Blocks = 2;
    private const int Hidden = 32;

    private static HunyuanVideo15Config TinyConfig() => new()
    {
        NumBlocks = Blocks, HiddenSize = Hidden, NumHeads = 2, LatentChannels = Channels, TextDim = 24, ByT5Dim = 16, RopeAxesDim = [4, 6, 6],
    };

    [Fact]
    public void SampleEmitsOneFramePerChunkInOrder()
    {
        using Fixture fixture = new();
        using MovieRollKvCache cache = fixture.NewCache();
        using Tensor output = fixture.Sample(cache, Options(numChunks: 3), fixture.Schedule(), cfg: null, seed: 7);

        Assert.Equal(new TensorShape(new long[] { 1, Channels, 3, Height, Width }), output.Shape);
    }

    [Fact]
    public void GuidanceAtScaleOneReproducesTheUnguidedRun()
    {
        using Fixture fixture = new();
        Tensor unguided;
        Tensor guided;
        using (MovieRollKvCache cache = fixture.NewCache())
            unguided = fixture.Sample(cache, Options(numChunks: 3), fixture.Schedule(), cfg: null, seed: 11);
        using (MovieRollKvCache cache = fixture.NewCache())
            guided = fixture.Sample(cache, Options(numChunks: 3), fixture.Schedule(), new MovieRollCfg(1f, 1f), seed: 11);

        using (unguided)
        using (guided)
            AssertClose(unguided, guided, 1e-4f);
    }

    [Fact]
    public void RecomputeFiresAndTheRunCompletes()
    {
        using Fixture fixture = new();
        using MovieRollKvCache cache = fixture.NewCache();
        using Tensor output = fixture.Sample(cache, Options(numChunks: 5, recompute: true, threshold: 3), fixture.Schedule(), cfg: null, seed: 13);

        Assert.Equal(5, output.Shape[2]);
        IReadOnlyList<(int Start, int End)> ranges = cache.FrameRanges(HunyuanVideo15Dit.StreamLayerName(0));
        Assert.Equal(3, ranges.Count);
        Assert.Equal(3, ranges[ranges.Count - 1].End);
    }

    [Fact]
    public void DistilledLadderRunsOneStepPerChunkWithoutGuidance()
    {
        using Fixture fixture = new();
        using MovieRollKvCache cache = fixture.NewCache();
        MovieRollSchedule distilled = MovieRollSchedule.Distilled(new[] { 1000, 800, 600, 400, 200 }, 5, 5.0);
        using Tensor output = fixture.Sample(cache, Options(numChunks: 2), distilled, cfg: null, seed: 17);

        Assert.Equal(2, output.Shape[2]);
    }

    private static MovieRollSamplerOptions Options(int numChunks, bool recompute = false, int threshold = 18) => new()
    {
        ChunkFrames = 1,
        NumChunks = numChunks,
        FullHistoryBlocks = 1,
        ContextNoise = 0.05f,
        KvRecomputeEnabled = recompute,
        KvRecomputeThreshold = threshold,
        KvRecomputeRecentK = 1,
    };

    private static void AssertClose(Tensor expected, Tensor actual, float tolerance)
    {
        Assert.Equal(expected.Shape, actual.Shape);
        ReadOnlySpan<float> e = expected.AsReadOnlySpan<float>();
        ReadOnlySpan<float> a = actual.AsReadOnlySpan<float>();
        float max = 0f;
        for (int i = 0; i < e.Length; i++) max = Math.Max(max, Math.Abs(e[i] - a[i]));
        Assert.True(max <= tolerance, $"max abs error {max:E3} exceeds {tolerance:E0}.");
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
        }

        public SafeTensorsLoader Loader { get; }
        public Dictionary<string, Tensor> Weights { get; }
        public CpuBackend Backend { get; }
        public HunyuanVideo15Dit Dit { get; }

        public MovieRollKvCache NewCache()
        {
            MovieRollKvCache cache = new(Backend);
            cache.Init(Dit.StreamLayerNames(), Height, Width);
            return cache;
        }

        public MovieRollSchedule Schedule() => MovieRollSchedule.Base(4, 2, 5.0);

        /// <summary>One run with the tiny prompt. The negative prompt is a scaled copy, so guidance acts on a different unconditional branch.</summary>
        public Tensor Sample(MovieRollKvCache cache, MovieRollSamplerOptions options, MovieRollSchedule schedule, MovieRollCfg? cfg, int seed)
        {
            Tensor qwen = Loader.GetTensor("in.qwen");
            Tensor byt5 = Loader.GetTensor("in.byt5");
            MovieRollSamplerOptions run = new()
            {
                ChunkFrames = options.ChunkFrames, TailChunkFrames = options.TailChunkFrames, NumChunks = options.NumChunks,
                FullHistoryBlocks = options.FullHistoryBlocks, ContextNoise = options.ContextNoise,
                KvRecomputeEnabled = options.KvRecomputeEnabled, KvRecomputeThreshold = options.KvRecomputeThreshold,
                KvRecomputeRecentK = options.KvRecomputeRecentK, Cfg = cfg,
            };
            MovieRollSampler sampler = new(Backend, Dit, schedule);
            Func<int, Tensor> noise = SeededNoise(seed);
            using Tensor negative = Scaled(qwen, 0.5f);
            return sampler.Sample(cache, run, qwen, byt5, cfg is null ? null : negative, null, noise);
        }

        private Tensor Scaled(Tensor source, float factor)
        {
            Tensor output = new(source.Shape, DType.F32);
            Backend.Scale(output, source, factor);
            return output;
        }

        /// <summary>Gaussian noise of shape [1, C, frames, H, W] from a seeded generator, so two runs with one seed draw the same sequence.</summary>
        private static Func<int, Tensor> SeededNoise(int seed)
        {
            Random random = new(seed);
            return frames =>
            {
                Tensor noise = new(new TensorShape(new long[] { 1, Channels, frames, Height, Width }), DType.F32);
                Span<float> values = noise.AsSpan<float>();
                for (int i = 0; i < values.Length; i++)
                {
                    double u1 = 1.0 - random.NextDouble();
                    double u2 = random.NextDouble();
                    values[i] = (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
                }
                return noise;
            };
        }

        public void Dispose()
        {
            Dit.Dispose();
            Backend.Dispose();
            Loader.Dispose();
        }
    }
}
