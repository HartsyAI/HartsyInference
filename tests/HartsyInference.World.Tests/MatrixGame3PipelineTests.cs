using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.World.Pipelines;
using HartsyInference.Tests.Common;

namespace HartsyInference.World.Tests;

/// <summary>End-to-end structural test of the Matrix-Game 3.0 segment loop on CPU: tiny-config
/// <see cref="MatrixGame3Transformer"/> + the reused Wan2.2 VAE decoder, two segments with bootstrap (seed image →
/// past/memory), FOV memory retrieval, FlowUniPC denoising, and per-segment decode. Numerics vs the reference
/// checkpoint are validation-pending.</summary>
public unsafe class MatrixGame3PipelineTests
{
    private readonly ITestOutputHelper _output;
    public MatrixGame3PipelineTests(ITestOutputHelper output) => _output = output;

    // SyntheticSmoke: full segment rollout on synthetic weights; can hard-crash the CPU test host
    // (native heap corruption) until Matrix-Game 3 is CPU-validated. Runs on the GPU lane. See CODE_STYLE.
    [Trait("Category", "SyntheticSmoke")]
    [Fact]
    public void Generate_TwoSegments_ProducesContinuousRollout()
    {
        CpuBackend backend = new();
        MatrixGame3Config cfg = new()
        {
            NumHeads = 2, HeadDim = 12, InChannels = 48, OutChannels = 48,
            TextDim = 16, FreqDim = 16, FfnDim = 32, NumLayers = 2,
            ActionStreamDim = 128, ActionHeads = 2, ActionHiddenSize = 8,   // ActionModule headDim 64 = rope [8,28,28]
            MemorySlots = 2, PastFrames = 2, FirstSegmentLatents = 3, SegmentLatents = 2,
            PluckerPatchDim = 6 * 32 * 32,
            ActionBlocks = [1],
            StepsBase = 2,
        };
        using MatrixGame3Transformer transformer = new(cfg);
        transformer.LoadWeights(MatrixGame3SyntheticWeights.Build(cfg));

        int[] dimMult = [1, 2, 4, 4];
        bool[] tUp = [false, true, true];
        Wan22VaeDecoder vae = new(dim: 8, zDim: 48, dimMult: dimMult, numResBlocks: 2, temperalUpsample: tUp);
        vae.LoadWeights(LanceSyntheticWeights.BuildVae(8, 48, dimMult, 2, tUp));

        MatrixGame3Pipeline pipeline = new(backend, transformer, vae, cfg);

        const int width = 32, height = 32, segments = 2;
        int requiredActions = pipeline.RequiredActionFrames(segments);
        Assert.Equal(cfg.FirstSegmentLatents * 4 + cfg.SegmentLatents * 4, requiredActions);

        float[][] keyboard = new float[requiredActions][];
        float[][] mouse = new float[requiredActions][];
        for (int i = 0; i < requiredActions; i++)
        {
            keyboard[i] = [1, 0, 0, 0, 0, 0];                      // hold W
            mouse[i] = [0.05f * ((i % 5) - 2), 0f];                // gentle yaw wiggles
        }

        Tensor promptEmbeds = RandRows(3, cfg.TextDim, seed: 1);
        Tensor negEmbeds = RandRows(2, cfg.TextDim, seed: 2);
        Tensor seedLatent = Rand5d(1, 48, 1, 2, 2, seed: 3);       // 32×32 → 2×2 latent

        (byte[][] frames, int w, int h, int seed) = pipeline.GenerateFromEmbeddings(
            promptEmbeds, negEmbeds, seedLatent, keyboard, mouse, width, height, segments,
            steps: 2, guidanceScale: 5f, seed: 42,
            p => _output.WriteLine($"step {p.Step}/{p.TotalSteps}"));

        Assert.Equal(pipeline.TotalFrames(segments), frames.Length);   // 9 + 5 under per-segment decode
        Assert.Equal(width, w);
        Assert.Equal(height, h);
        foreach (byte[] f in frames) Assert.Equal(width * height * 3, f.Length);
        _ = seed;
    }

    [Fact]
    public void Generate_RejectsShortActionPlanAndBadSeedLatent()
    {
        CpuBackend backend = new();
        MatrixGame3Config cfg = new()
        {
            NumHeads = 2, HeadDim = 12, InChannels = 48, OutChannels = 48,
            TextDim = 16, FreqDim = 16, FfnDim = 32, NumLayers = 1,
            ActionStreamDim = 128, ActionHeads = 2, ActionHiddenSize = 8,
            MemorySlots = 2, PastFrames = 2, FirstSegmentLatents = 3, SegmentLatents = 2,
            PluckerPatchDim = 6 * 32 * 32, ActionBlocks = [],
        };
        using MatrixGame3Transformer transformer = new(cfg);
        transformer.LoadWeights(MatrixGame3SyntheticWeights.Build(cfg));
        Wan22VaeDecoder vae = new(dim: 8, zDim: 48, dimMult: [1, 2, 4, 4], numResBlocks: 2, temperalUpsample: [false, true, true]);
        vae.LoadWeights(LanceSyntheticWeights.BuildVae(8, 48, [1, 2, 4, 4], 2, [false, true, true]));
        MatrixGame3Pipeline pipeline = new(backend, transformer, vae, cfg);

        Tensor prompt = RandRows(2, cfg.TextDim, seed: 5);
        Tensor neg = RandRows(2, cfg.TextDim, seed: 6);
        Tensor seedLatent = Rand5d(1, 48, 1, 2, 2, seed: 7);
        float[][] tooFew = [[0, 0, 0, 0, 0, 0]];
        float[][] mouse1 = [[0, 0]];

        Assert.Throws<ArgumentException>(() => pipeline.GenerateFromEmbeddings(
            prompt, neg, seedLatent, tooFew, mouse1, 32, 32, numSegments: 1));

        Tensor badSeed = Rand5d(1, 48, 1, 4, 4, seed: 8);
        float[][] kbd = Enumerable.Repeat(new float[] { 0, 0, 0, 0, 0, 0 }, 12).ToArray();
        float[][] mouse = Enumerable.Repeat(new float[] { 0, 0 }, 12).ToArray();
        Assert.Throws<ArgumentException>(() => pipeline.GenerateFromEmbeddings(
            prompt, neg, badSeed, kbd, mouse, 32, 32, numSegments: 1));
    }

    private static Tensor RandRows(int rows, int cols, int seed)
    {
        Tensor t = new Tensor(new TensorShape(rows, cols), DType.F32);
        float* p = (float*)t.DataPointer;
        Random rng = new(seed);
        for (long i = 0; i < t.Shape.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }

    private static Tensor Rand5d(int b, int c, int t, int h, int w, int seed)
    {
        Tensor x = new Tensor(new TensorShape([(long)b, c, t, h, w]), DType.F32);
        float* p = (float*)x.DataPointer;
        Random rng = new(seed);
        for (long i = 0; i < x.Shape.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 2 - 1);
        return x;
    }
}
