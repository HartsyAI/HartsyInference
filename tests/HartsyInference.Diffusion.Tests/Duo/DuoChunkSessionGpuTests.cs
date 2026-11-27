using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Duo;
using HartsyInference.Tests.Shared;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests.Duo;

/// <summary>The Duo chunk session on CUDA against the CPU, on the same tiny synthetic model: two blocks of a prefilled session, same inputs,
/// same weights. Synthetic weights only; no released weights and no upstream code are run. Reported as skipped, never passed, when no CUDA
/// device opens. Prints the largest per-block error on each run.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class DuoChunkSessionGpuTests
{
    private readonly ITestOutputHelper _output;

    public DuoChunkSessionGpuTests(ITestOutputHelper output)
    {
        _output = output;
    }

    private const int Hidden = 8, Heads = 2, HeadDim = 96, Inner = Heads * HeadDim, Ffn = 16, TextDim = 5, Layers = 2;
    private const int VideoIn = 96, AudioIn = 32, TimeEmbedDim = 6, TimeEmbedHidden = 8;
    private const int Shared = 3, BodyRows = 2, RefRows = 4, VideoRows = 8, AudioRows = 6;
    /// <summary>F32 on the device against F32 on the CPU, for a model this small. A wider gap points at a kernel, not rounding.</summary>
    private const float Tolerance = 1e-3f;

    [CudaDeviceFact]
    public void CudaBlocksMatchTheCpuBlocks()
    {
        float[] cpuVideo0, cpuAudio0, cpuVideo1, cpuAudio1;
        using (MiniMaxH3Transformer transformer = Transformer())
        {
            (cpuVideo0, cpuAudio0, cpuVideo1, cpuAudio1) = Rollout(transformer, TestBackends.Cpu);
        }

        using CudaBackend cuda = new(0, PtxDirectory());
        float[] gpuVideo0, gpuAudio0, gpuVideo1, gpuAudio1;
        using (MiniMaxH3Transformer transformer = Transformer())
        {
            (gpuVideo0, gpuAudio0, gpuVideo1, gpuAudio1) = Rollout(transformer, cuda);
        }

        AssertClose(cpuVideo0, gpuVideo0, "block 0 video");
        AssertClose(cpuAudio0, gpuAudio0, "block 0 audio");
        AssertClose(cpuVideo1, gpuVideo1, "block 1 video");
        AssertClose(cpuAudio1, gpuAudio1, "block 1 audio");
    }

    /// <summary>The device readbacks of one block do not grow with its step count: the denoise reads nothing per step, and the block reads its
    /// video and audio once at the end. The count is taken with the backend's device-to-host sync counter.</summary>
    [CudaDeviceFact]
    public void ABlockReadsBackOnceRegardlessOfItsStepCount()
    {
        using CudaBackend cuda = new(0, PtxDirectory());
        long twoSteps = ReadsForOneBlock(cuda, steps: 2);
        long sixSteps = ReadsForOneBlock(cuda, steps: 6);

        Assert.Equal(twoSteps, sixSteps);
        Assert.True(twoSteps <= 2, $"A block read back {twoSteps} times; the design is one readback each for video and audio.");
    }

    private static long ReadsForOneBlock(CudaBackend cuda, int steps)
    {
        using MiniMaxH3Transformer transformer = Transformer();
        using DuoStreamCache cache = new(cuda, blocks: Layers, width: Inner, capacityRows: 64);
        DuoChunkSession session = new(transformer, cuda, Sample(), DuoRopePolicy.Old, cache);
        session.Prefill(Shared0(), Values(30, RefRows * VideoIn));
        cuda.ResetD2hSyncCount();
        session.GenerateBlock(0, Body(0), Noise(40, VideoRows), Values(41, AudioRows * AudioIn), steps, 1.0);
        return cuda.GetD2hSyncCount();
    }

    /// <summary>Prefills and generates two blocks. The cache is on <paramref name="backend"/>, so a CUDA run keeps its history on the device.</summary>
    private static (float[] Video0, float[] Audio0, float[] Video1, float[] Audio1) Rollout(MiniMaxH3Transformer transformer, IBackend backend)
    {
        using DuoStreamCache cache = new(backend, blocks: Layers, width: Inner, capacityRows: 64);
        DuoChunkSession session = new(transformer, backend, Sample(), DuoRopePolicy.Old, cache);
        session.Prefill(Shared0(), Values(30, RefRows * VideoIn));
        (float[] v0, float[] a0) = session.GenerateBlock(0, Body(0), Noise(40, VideoRows), Values(41, AudioRows * AudioIn), 2, 1.0);
        (float[] v1, float[] a1) = session.GenerateBlock(1, Body(1), Noise(50, VideoRows), Values(51, AudioRows * AudioIn), 2, 1.0);
        return (v0, a0, v1, a1);
    }

    private void AssertClose(float[] expected, float[] actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        float worst = 0f;
        for (int i = 0; i < expected.Length; i++)
        {
            worst = MathF.Max(worst, MathF.Abs(expected[i] - actual[i]));
        }
        _output.WriteLine($"{what}: max |CPU - CUDA| = {worst:E3} (tolerance {Tolerance:E0})");
        Assert.True(worst <= Tolerance, $"{what}: CUDA differs from the CPU by {worst}, above {Tolerance}.");
    }

    private static string PtxDirectory()
    {
        for (DirectoryInfo? dir = new(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            string candidate = Path.Combine(dir.FullName, "src", "HartsyInference.Cuda", "Ptx");
            if (Directory.Exists(candidate))
            {
                return candidate;
            }
        }
        throw new DirectoryNotFoundException("Could not find src/HartsyInference.Cuda/Ptx above the test binary.");
    }

    private static DuoSampleInput Sample() => new()
    {
        SampleId = 3,
        SharedDescriptionRows = 2,
        SharedCueRows = 1,
        Reference = new DuoGrid(1, 4, 4),
        Blocks =
        [
            new DuoBlockInput { StartMs = 0.0, BodyTextRows = BodyRows, Video = new DuoGrid(2, 4, 4), AudioSteps = 3, FrameStartsMs = [0.0, 100.0] },
            new DuoBlockInput { StartMs = 708.0, BodyTextRows = BodyRows, Video = new DuoGrid(2, 4, 4), AudioSteps = 3, FrameStartsMs = [708.0, 808.0] },
        ],
        CleanBlocks = 0,
        NoisyBlocks = [0],
    };

    private static float[] Shared0() => Values(2, Shared * TextDim);

    private static float[] Body(int block) => Values(10 + block, BodyRows * TextDim);

    private static float[] Noise(int seed, int rows) => Values(seed, rows * VideoIn);

    private static MiniMaxH3Transformer Transformer()
    {
        MiniMaxH3Transformer transformer = new(new MiniMaxH3Config
        {
            HiddenSize = Hidden,
            NumLayers = Layers,
            TokenRefinerNumLayers = 1,
            NumAttentionHeads = Heads,
            AttentionHeadDim = HeadDim,
            FfnHiddenSize = Ffn,
            LatentsDim = 24,
            HasAudio = true,
            TextDim = TextDim,
            OutputHeadCount = 4,
            TimeEmbedHiddenSize = TimeEmbedHidden,
            TimeEmbedDim = TimeEmbedDim,
        });
        Dictionary<string, Tensor> weights = new()
        {
            ["video_patch_proj.weight"] = Weight(Hidden, VideoIn, 10, 1f),
            ["audio_patch_proj.weight"] = Weight(Hidden, AudioIn, 11, 1f),
            ["condition_proj.weight"] = Weight(Hidden, TextDim, 12, 1f),
            ["time_embedder.proj_in.weight"] = Weight(TimeEmbedHidden, 256, 13, 1f),
            ["time_embedder.proj_out.weight"] = Weight(TimeEmbedDim, TimeEmbedHidden, 14, 1f),
            ["token_refiner.blocks.0.norm1.weight"] = Ones(Hidden),
            ["token_refiner.blocks.0.norm2.weight"] = Ones(Hidden),
            ["token_refiner.blocks.0.attn.qkv_proj.weight"] = Weight(3 * Inner, Hidden, 15, 1f),
            ["token_refiner.blocks.0.attn.q_norm.weight"] = Ones(HeadDim),
            ["token_refiner.blocks.0.attn.k_norm.weight"] = Ones(HeadDim),
            ["token_refiner.blocks.0.attn.out_proj.weight"] = Weight(Hidden, Inner, 16, 1f),
            ["token_refiner.blocks.0.mlp.fc1.weight"] = Weight(2 * Ffn, Hidden, 17, 1f),
            ["token_refiner.blocks.0.mlp.fc2.weight"] = Weight(Hidden, Ffn, 18, 1f),
            ["token_refiner.final_norm.weight"] = Ones(Hidden),
            ["final_layer.norm.weight"] = Ones(Hidden),
            ["final_layer.adaln_proj.linear.weight"] = Weight(2 * Hidden, TimeEmbedDim, 24, 0.5f),
            ["final_layer.video_out.weight"] = Weight(VideoIn, Hidden, 25, 1f),
            ["final_layer.audio_out.weight"] = Weight(AudioIn, Hidden, 26, 1f),
        };
        for (int layer = 0; layer < Layers; layer++)
        {
            int s = 100 * layer;
            string p = $"blocks.{layer}";
            weights[$"{p}.norm1.weight"] = Ones(Hidden);
            weights[$"{p}.norm2.weight"] = Ones(Hidden);
            weights[$"{p}.adaln_proj.linear.weight"] = Weight(6 * Hidden * 3, TimeEmbedDim, 19 + s, 0.5f);
            weights[$"{p}.attn.qkv_proj.weight"] = Weight(3 * Inner, Hidden, 20 + s, 1f);
            weights[$"{p}.attn.q_norm.weight"] = Ones(HeadDim);
            weights[$"{p}.attn.k_norm.weight"] = Ones(HeadDim);
            weights[$"{p}.attn.out_proj.weight"] = Weight(Hidden, Inner, 21 + s, 1f);
            weights[$"{p}.mlp.fc1.weight"] = Weight(2 * Ffn, Hidden, 22 + s, 1f);
            weights[$"{p}.mlp.fc2.weight"] = Weight(Hidden, Ffn, 23 + s, 1f);
        }
        transformer.LoadWeights(weights);
        return transformer;
    }

    private static Tensor Weight(int rows, int cols, int seed, float scale) =>
        MakeTensor(new TensorShape(rows, cols), Values(seed, rows * cols, scale));

    private static Tensor MakeTensor(TensorShape shape, float[] values)
    {
        Tensor tensor = new(shape, DType.F32);
        values.AsSpan().CopyTo(tensor.AsSpan<float>());
        return tensor;
    }

    private static Tensor Ones(int count)
    {
        float[] data = new float[count];
        Array.Fill(data, 1f);
        return MakeTensor(new TensorShape(count), data);
    }

    /// <summary>Hash-based values in [-scale, scale], the same generator the CPU chunk session tests use.</summary>
    private static float[] Values(int seed, int count, float scale = 1f)
    {
        float[] data = new float[count];
        for (int i = 0; i < count; i++)
        {
            double h = Math.Sin(seed * 12.9898 + i * 78.233) * 43758.5453;
            double frac = h - Math.Floor(h);
            data[i] = scale * (float)(2.0 * frac - 1.0);
        }
        return data;
    }
}
