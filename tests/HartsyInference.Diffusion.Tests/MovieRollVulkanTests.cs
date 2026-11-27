using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines.MovieRoll;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.Tests.Shared;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.Diffusion.Models.Denoisers;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The MovieRoll decode on a device, through <see cref="MovieRollPipeline.Generate"/>: the tiny 4-channel VAE is placed for the decode and
/// released after it, and the frames must match the CPU run. The bound is not set yet: the test prints the max error and asserts only that the
/// sampling and the decode read nothing back from the device before the frames are compared. Runs only through the GPU session
/// (Category=GpuIntegration); it is skipped where no Vulkan device opens.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class MovieRollVulkanTests(ITestOutputHelper output)
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

    [VulkanDeviceFact]
    public void Vulkan_PipelineDecode_ReadsNothingBackAndPrintsTheMaxErrorAgainstCpu()
    {
        using Fixture cpu = new(new CpuBackend());
        using Tensor expected = cpu.Generate();

        using Fixture device = new(GpuDevices.OpenVulkan());
        device.Backend.ResetD2hSyncCount();
        using Tensor actual = device.Generate();
        long reads = device.Backend.GetD2hSyncCount();
        output.WriteLine($"D2H_SYNCS MovieRollVulkanDecode: {reads}");

        Assert.Equal(expected.Shape, actual.Shape);
        ReadOnlySpan<float> e = expected.AsReadOnlySpan<float>();
        ReadOnlySpan<float> a = actual.AsReadOnlySpan<float>();
        float max = 0f;
        int at = 0;
        for (int i = 0; i < e.Length; i++)
        {
            float d = Math.Abs(e[i] - a[i]);
            if (float.IsNaN(d)) { max = float.NaN; at = i; break; }
            if (d > max) { max = d; at = i; }
        }
        output.WriteLine($"MAX_ERR MovieRollVulkanDecode frames: {max:E3} at flat index {at} of {e.Length} (cpu {e[at]:G9}, device {a[at]:G9})");
        output.WriteLine($"SWITCHES MovieRollVulkanDecode: HighPrecisionGemm={device.Backend.HighPrecisionGemm}");

        Assert.Equal(0, reads);
    }

    private sealed class Fixture : IDisposable
    {
        public Fixture(IBackend backend)
        {
            Backend = backend;
            Weights = new SafeTensorsLoader();
            Weights.Load(Path.Combine(AppContext.BaseDirectory, "Fixtures", "hy15_dit_tiny.safetensors"));
            Dit = new HunyuanVideo15Dit(new HunyuanVideo15Config
            {
                NumBlocks = Blocks, HiddenSize = Hidden, NumHeads = 2, LatentChannels = Channels, TextDim = 24, ByT5Dim = 16, RopeAxesDim = [4, 6, 6],
            });
            Dit.LoadWeights(HunyuanVideo15CheckpointConverter.Convert(
                Weights.GetAllTensors()
                    .Where(pair => !pair.Key.StartsWith("ref.", StringComparison.Ordinal) && !pair.Key.StartsWith("in.", StringComparison.Ordinal)
                        && !pair.Key.StartsWith("win.", StringComparison.Ordinal) && !pair.Key.StartsWith("rec.", StringComparison.Ordinal))
                    .ToDictionary(pair => pair.Key, pair => pair.Value),
                Blocks, Hidden));

            VaeWeights = new SafeTensorsLoader();
            VaeWeights.Load(Path.Combine(AppContext.BaseDirectory, "Fixtures", "hy15_vae_tiny_z4.safetensors"));
            Vae = new HunyuanVideo15Vae(new HunyuanVideo15VaeConfig
            {
                LatentChannels = Channels, BlockOutChannels = [4, 8, 16, 32, 32], LayersPerBlock = 2,
                SpatialCompression = 16, TemporalCompression = 4, ScalingFactor = 2f,
            });
            Vae.LoadWeights(VaeWeights.GetAllTensors()
                .Where(pair => pair.Key.StartsWith("encoder.", StringComparison.Ordinal) || pair.Key.StartsWith("decoder.", StringComparison.Ordinal))
                .ToDictionary(pair => pair.Key, pair => pair.Value));

            Qwen = Weights.GetTensor("in.qwen");
            ByT5 = Weights.GetTensor("in.byt5");
            Random random = new(77);
            NegativeQwen = RandomTensor(random, Qwen);
            NegativeByt5 = RandomTensor(random, ByT5);
        }

        public IBackend Backend { get; }
        public SafeTensorsLoader Weights { get; }
        public SafeTensorsLoader VaeWeights { get; }
        public HunyuanVideo15Dit Dit { get; }
        public HunyuanVideo15Vae Vae { get; }
        public Tensor Qwen { get; }
        public Tensor ByT5 { get; }
        public Tensor NegativeQwen { get; }
        public Tensor NegativeByt5 { get; }

        /// <summary>The frames of one clip through the whole pipeline (sampler, then decode). The caller owns the returned frames.</summary>
        public Tensor Generate()
        {
            using MovieRollPipeline pipeline = new(Backend, Dit, Vae, MovieRollSchedule.Base(4, 2, 5.0));
            return pipeline.Generate(Options, Qwen, ByT5, NegativeQwen, NegativeByt5, Height, Width, SeededNoise(7));
        }

        private static Func<int, Tensor> SeededNoise(long seed)
        {
            long next = seed;
            return frames => TorchCompatibleNoise.Randn(new TensorShape([1L, Channels, frames, Height, Width]), next++);
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
            NegativeQwen.Dispose();
            NegativeByt5.Dispose();
            Vae.Dispose();
            Dit.Dispose();
            Backend.Dispose();
            VaeWeights.Dispose();
            Weights.Dispose();
        }
    }
}
