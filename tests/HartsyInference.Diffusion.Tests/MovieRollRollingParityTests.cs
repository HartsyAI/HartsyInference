using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Pipelines.MovieRoll;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>MovieRollSampler against the rolling-loop transcription (tests/python-reference/movieroll_reference/dump_rolling_tiny.py, MovieRoll 0b5f73e).
/// The transcription saves every noise draw in call order; the sampler replays them through its noise callback, so the two runs see the same
/// chunks, context noise and recompute inputs. The outputs are compared element-wise.</summary>
public sealed class MovieRollRollingParityTests
{
    private const int Channels = 4;
    private const int Height = 2;
    private const int Width = 3;
    private const int Blocks = 2;
    private const int Hidden = 32;
    private const float Tolerance = 1e-4f;

    [Fact]
    public void BaseChainWithCfgAndRecomputeMatchesTheTranscription()
    {
        using Fixture fixture = new();
        MovieRollSamplerOptions options = new()
        {
            ChunkFrames = 1, NumChunks = 5, FullHistoryBlocks = 1, ContextNoise = 0.05f,
            KvRecomputeEnabled = true, KvRecomputeThreshold = 3, KvRecomputeRecentK = 1, Cfg = new MovieRollCfg(1f, 4f),
        };
        fixture.AssertRun("base", options, MovieRollSchedule.Base(4, 2, 5.0), negative: true);
    }

    [Fact]
    public void BaseChainWithTwoFrameChunksMatchesTheTranscription()
    {
        using Fixture fixture = new();
        MovieRollSamplerOptions options = new()
        {
            ChunkFrames = 2, NumChunks = 3, FullHistoryBlocks = 1, ContextNoise = 0.05f,
            KvRecomputeEnabled = false, Cfg = new MovieRollCfg(1f, 4f),
        };
        fixture.AssertRun("base_chunk2", options, MovieRollSchedule.Base(4, 2, 5.0), negative: true);
    }

    [Fact]
    public void DistilledLadderWithRecomputeMatchesTheTranscription()
    {
        using Fixture fixture = new();
        MovieRollSamplerOptions options = new()
        {
            ChunkFrames = 1, NumChunks = 7, FullHistoryBlocks = 1, ContextNoise = 0.15f,
            KvRecomputeEnabled = true, KvRecomputeThreshold = 3, KvRecomputeRecentK = 1, Cfg = null,
        };
        fixture.AssertRun("distilled", options, MovieRollSchedule.Distilled(new[] { 1000, 800, 600, 400, 200 }, 5, 5.0), negative: false);
    }

    private static HunyuanVideo15Config TinyConfig() => new()
    {
        NumBlocks = Blocks, HiddenSize = Hidden, NumHeads = 2, LatentChannels = Channels, TextDim = 24, ByT5Dim = 16, RopeAxesDim = [4, 6, 6],
    };

    private static void AssertClose(Tensor expected, Tensor actual, float tolerance, string name)
    {
        Assert.Equal(expected.Shape, actual.Shape);
        ReadOnlySpan<float> e = expected.AsReadOnlySpan<float>();
        ReadOnlySpan<float> a = actual.AsReadOnlySpan<float>();
        float max = 0f;
        for (int i = 0; i < e.Length; i++) max = Math.Max(max, Math.Abs(e[i] - a[i]));
        Assert.True(max <= tolerance, $"{name}: max abs error {max:E3} exceeds {tolerance:E0}.");
    }

    private sealed class Fixture : IDisposable
    {
        public Fixture()
        {
            Weights = new SafeTensorsLoader();
            Weights.Load(Path.Combine(AppContext.BaseDirectory, "Fixtures", "hy15_dit_tiny.safetensors"));
            Reference = new SafeTensorsLoader();
            Reference.Load(Path.Combine(AppContext.BaseDirectory, "Fixtures", "rolling_tiny.safetensors"));
            Backend = new CpuBackend();
            Dit = new HunyuanVideo15Dit(TinyConfig());
            Dit.LoadWeights(HunyuanVideo15CheckpointConverter.Convert(
                Weights.GetAllTensors()
                    .Where(pair => !pair.Key.StartsWith("ref.", StringComparison.Ordinal) && !pair.Key.StartsWith("in.", StringComparison.Ordinal)
                        && !pair.Key.StartsWith("win.", StringComparison.Ordinal) && !pair.Key.StartsWith("rec.", StringComparison.Ordinal))
                    .ToDictionary(pair => pair.Key, pair => pair.Value),
                Blocks, Hidden));
        }

        public SafeTensorsLoader Weights { get; }
        public SafeTensorsLoader Reference { get; }
        public CpuBackend Backend { get; }
        public HunyuanVideo15Dit Dit { get; }

        /// <summary>Runs the sampler with the transcription's prompts and replays its noise draws; fails if the run draws a different sequence.</summary>
        public void AssertRun(string name, MovieRollSamplerOptions options, MovieRollSchedule schedule, bool negative)
        {
            List<Tensor> draws = Draws(name);
            int next = 0;
            Func<int, Tensor> noise = frames =>
            {
                Assert.True(next < draws.Count, $"{name}: the sampler drew more noise than the transcription ({draws.Count}).");
                Tensor source = draws[next++];
                Assert.Equal(frames, (int)source.Shape[2]);
                Tensor copy = new(source.Shape, DType.F32);
                source.AsReadOnlySpan<float>().CopyTo(copy.AsSpan<float>());
                return copy;
            };

            using MovieRollKvCache cache = new(Backend);
            cache.Init(Dit.StreamLayerNames(), Height, Width);
            Tensor qwen = Weights.GetTensor("in.qwen");
            Tensor byt5 = Weights.GetTensor("in.byt5");
            Tensor? negativeQwen = negative ? Reference.GetTensor("in.neg_qwen") : null;
            Tensor? negativeByt5 = negative ? Reference.GetTensor("in.neg_byt5") : null;
            MovieRollSampler sampler = new(Backend, Dit, schedule);
            using Tensor output = sampler.Sample(cache, options, qwen, byt5, negativeQwen, negativeByt5, noise);

            Assert.Equal(draws.Count, next);
            AssertClose(Reference.GetTensor($"{name}.out"), output, Tolerance, name);
        }

        private List<Tensor> Draws(string name)
        {
            string prefix = $"{name}.noise.";
            return Reference.GetAllTensors()
                .Where(pair => pair.Key.StartsWith(prefix, StringComparison.Ordinal))
                .OrderBy(pair => pair.Key, StringComparer.Ordinal)
                .Select(pair => pair.Value)
                .ToList();
        }

        public void Dispose()
        {
            Dit.Dispose();
            Backend.Dispose();
            Reference.Dispose();
            Weights.Dispose();
        }
    }
}
