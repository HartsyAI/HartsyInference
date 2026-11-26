using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines.MovieRoll;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The MovieRoll pipeline on the tiny DiT. <c>SyntheticSmoke</c>: the checks are wiring (the pipeline's latents equal the sampler's on the same
/// noise, and a rerun with the same seed is identical), not parity. Parity for the sampler loop is in <see cref="MovieRollRollingParityTests"/>. The tiny
/// VAE fixture has two latent channels and this DiT has four, so <see cref="MovieRollPipeline.Generate"/>'s decode is not exercised here; the VAE tests cover it.</summary>
[Trait("Category", "SyntheticSmoke")]
public sealed class MovieRollPipelineTests
{
    private const int Channels = 4;
    private const int Height = 2;
    private const int Width = 3;
    private const int Blocks = 2;
    private const int Hidden = 32;

    private static readonly MovieRollSamplerOptions Options = new()
    {
        ChunkFrames = 1, FirstChunkFrames = 2, NumChunks = 5, FullHistoryBlocks = 1, ContextNoise = 0.05f,
        KvRecomputeEnabled = true, KvRecomputeThreshold = 3, KvRecomputeRecentK = 1, Cfg = new MovieRollCfg(1f, 4f),
    };

    [Fact]
    public void SampleLatentsMatchesTheSamplerOnTheSameNoise()
    {
        using Fixture fixture = new();
        using MovieRollPipeline pipeline = fixture.Pipeline();
        using Tensor latents = pipeline.SampleLatents(Options, fixture.Qwen, fixture.ByT5, fixture.NegativeQwen, fixture.NegativeByt5, Height, Width,
            SeededNoise(7));

        Assert.Equal(new TensorShape([1L, Channels, 6, Height, Width]), latents.Shape);
        using Tensor direct = fixture.DirectSample(SeededNoise(7));
        Assert.Equal(direct.Shape, latents.Shape);
        ReadOnlySpan<float> a = latents.AsReadOnlySpan<float>();
        ReadOnlySpan<float> b = direct.AsReadOnlySpan<float>();
        for (int i = 0; i < a.Length; i++) Assert.Equal(b[i], a[i]);
    }

    [Fact]
    public void SameSeedGivesTheSameLatents()
    {
        using Fixture fixture = new();
        using MovieRollPipeline pipeline = fixture.Pipeline();
        using Tensor first = pipeline.SampleLatents(Options, fixture.Qwen, fixture.ByT5, fixture.NegativeQwen, fixture.NegativeByt5, Height, Width,
            SeededNoise(11));
        using Tensor second = pipeline.SampleLatents(Options, fixture.Qwen, fixture.ByT5, fixture.NegativeQwen, fixture.NegativeByt5, Height, Width,
            SeededNoise(11));
        ReadOnlySpan<float> a = first.AsReadOnlySpan<float>();
        ReadOnlySpan<float> b = second.AsReadOnlySpan<float>();
        for (int i = 0; i < a.Length; i++) Assert.Equal(b[i], a[i]);
    }

    /// <summary>Draws the same noise sequence for a seed: each call takes the next seed, so the sampler's draws are reproducible.</summary>
    private static Func<int, Tensor> SeededNoise(long seed)
    {
        long next = seed;
        return frames => TorchCompatibleNoise.Randn(new TensorShape([1L, Channels, frames, Height, Width]), next++);
    }

    private static HunyuanVideo15Config TinyConfig() => new()
    {
        NumBlocks = Blocks, HiddenSize = Hidden, NumHeads = 2, LatentChannels = Channels, TextDim = 24, ByT5Dim = 16, RopeAxesDim = [4, 6, 6],
    };

    private sealed class Fixture : IDisposable
    {
        public Fixture()
        {
            Weights = new SafeTensorsLoader();
            Weights.Load(Path.Combine(AppContext.BaseDirectory, "Fixtures", "hy15_dit_tiny.safetensors"));
            Backend = new CpuBackend();
            Dit = new HunyuanVideo15Dit(TinyConfig());
            Dit.LoadWeights(HunyuanVideo15CheckpointConverter.Convert(
                Weights.GetAllTensors()
                    .Where(pair => !pair.Key.StartsWith("ref.", StringComparison.Ordinal) && !pair.Key.StartsWith("in.", StringComparison.Ordinal)
                        && !pair.Key.StartsWith("win.", StringComparison.Ordinal) && !pair.Key.StartsWith("rec.", StringComparison.Ordinal))
                    .ToDictionary(pair => pair.Key, pair => pair.Value),
                Blocks, Hidden));
            Qwen = Weights.GetTensor("in.qwen");
            ByT5 = Weights.GetTensor("in.byt5");
            // The prompt features are the fixture's; the negative prompt is a fixed random pair, as in the transcription's CFG run.
            Random random = new(77);
            NegativeQwen = RandomTensor(random, Qwen);
            NegativeByt5 = RandomTensor(random, ByT5);
        }

        public SafeTensorsLoader Weights { get; }
        public CpuBackend Backend { get; }
        public HunyuanVideo15Dit Dit { get; }
        public Tensor Qwen { get; }
        public Tensor ByT5 { get; }
        public Tensor NegativeQwen { get; }
        public Tensor NegativeByt5 { get; }

        public MovieRollPipeline Pipeline() => new(Backend, Dit, new HunyuanVideo15Vae(HunyuanVideo15VaeConfig.Default),
            MovieRollSchedule.Base(4, 2, 5.0));

        /// <summary>The sampler on its own with the same inputs, to compare the pipeline's latents against.</summary>
        public Tensor DirectSample(Func<int, Tensor> noise)
        {
            using MovieRollKvCache cache = new(Backend);
            cache.Init(Dit.StreamLayerNames(), Height, Width);
            MovieRollSampler sampler = new(Backend, Dit, MovieRollSchedule.Base(4, 2, 5.0));
            return sampler.Sample(cache, Options, Qwen, ByT5, NegativeQwen, NegativeByt5, noise);
        }

        private static Tensor RandomTensor(Random random, Tensor like)
        {
            Tensor tensor = new(like.Shape, DType.F32);
            Span<float> values = tensor.AsSpan<float>();
            for (int i = 0; i < values.Length; i++) values[i] = (float)(random.NextDouble() * 2 - 1);
            return tensor;
        }

        public void Dispose()
        {
            Dit.Dispose();
            Backend.Dispose();
            Weights.Dispose();
        }
    }
}
