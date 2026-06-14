using Xunit;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Tests.Common;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Tiny-config CPU tests for the Matrix-Game 3.0 DiT — the memory-augmented sequence (mem ‖ past ‖ current
/// with per-frame timesteps + historical RoPE indices), the trailing-frames readout, and the ActionModule/Plücker
/// hooks. Numerics vs the reference checkpoint are validation-pending.</summary>
public unsafe class MatrixGame3TransformerTests
{
    private static MatrixGame3Config TinyConfig => new()
    {
        NumHeads = 2, HeadDim = 12, InChannels = 8, OutChannels = 8,
        TextDim = 16, FreqDim = 16, FfnDim = 32, NumLayers = 2,
        ActionStreamDim = 128, ActionHeads = 2, ActionHiddenSize = 8,   // headDim 64 = rope [8,28,28]
        MemorySlots = 2, PastFrames = 2, FirstSegmentLatents = 3, SegmentLatents = 2,
        PluckerPatchDim = 6 * 32 * 32,
        ActionBlocks = [1],
    };

    [Fact]
    public void Forward_MemoryAugmentedSequence_ReadsOutTrailingFrames()
    {
        CpuBackend backend = new();
        MatrixGame3Config cfg = TinyConfig;
        using MatrixGame3Transformer transformer = new(cfg);
        transformer.LoadWeights(MatrixGame3SyntheticWeights.Build(cfg));

        int mem = 2, past = 2, cur = 3, tTotal = mem + past + cur;
        Tensor latent = Rand5d(1, cfg.InChannels, tTotal, 4, 4, seed: 41);
        Tensor encoder = RandRows(3, cfg.TextDim, seed: 42);
        Tensor mouse = RandRows(20, 2, seed: 43);
        Tensor keyboard = RandRows(20, 6, seed: 44);

        float[] frameTs = new float[tTotal];
        for (int i = mem + past; i < tTotal; i++) frameTs[i] = 500f;
        int[] ropeIdx = [0, 9, 17, 18, 19, 20, 21];   // memory keeps historical positions

        Tensor outVel = transformer.Forward(backend, latent, encoder, frameTs, ropeIdx, mem, cur, mouse, keyboard, pluckerTokens: null);

        Assert.Equal(5, outVel.Shape.Rank);
        Assert.Equal(cfg.OutChannels, (int)outVel.Shape[1]);
        Assert.Equal(cur, (int)outVel.Shape[2]);     // memory + past dropped from the readout
        Assert.Equal(4, (int)outVel.Shape[3]);
        Assert.Equal(4, (int)outVel.Shape[4]);
        float* p = (float*)outVel.DataPointer;
        for (long i = 0; i < outVel.Shape.ElementCount; i++) Assert.True(float.IsFinite(p[i]), $"non-finite at {i}");
    }

    [Fact]
    public void Forward_ActionsAndPlucker_ChangeTheOutput()
    {
        CpuBackend backend = new();
        MatrixGame3Config cfg = TinyConfig;
        using MatrixGame3Transformer transformer = new(cfg);
        transformer.LoadWeights(MatrixGame3SyntheticWeights.Build(cfg));

        int mem = 2, past = 2, cur = 3, tTotal = mem + past + cur;
        Tensor latent = Rand5d(1, cfg.InChannels, tTotal, 4, 4, seed: 51);
        Tensor encoder = RandRows(3, cfg.TextDim, seed: 52);
        float[] frameTs = new float[tTotal];
        for (int i = mem + past; i < tTotal; i++) frameTs[i] = 500f;
        int[] ropeIdx = [0, 4, 8, 9, 10, 11, 12];

        Tensor mouseA = RandRows(20, 2, seed: 53);
        Tensor mouseB = RandRows(20, 2, seed: 99);   // different mouse trajectory
        Tensor keyboard = RandRows(20, 6, seed: 54);

        Tensor outA = transformer.Forward(backend, latent, encoder, frameTs, ropeIdx, mem, cur, mouseA, keyboard, null);
        Tensor outB = transformer.Forward(backend, latent, encoder, frameTs, ropeIdx, mem, cur, mouseB, keyboard, null);
        Assert.True(MaxDiff(outA, outB) > 1e-6f, "different mouse input must change the prediction");

        // Plücker tokens shift the patch embeddings.
        int tokens = tTotal * 2 * 2;
        Tensor plucker = RandRows(tokens, cfg.PluckerPatchDim, seed: 55);
        Tensor outC = transformer.Forward(backend, latent, encoder, frameTs, ropeIdx, mem, cur, mouseA, keyboard, plucker);
        Assert.True(MaxDiff(outA, outC) > 1e-6f, "plucker conditioning must change the prediction");
    }

    private static float MaxDiff(Tensor a, Tensor b)
    {
        float* ap = (float*)a.DataPointer;
        float* bp = (float*)b.DataPointer;
        float max = 0;
        for (long i = 0; i < a.Shape.ElementCount; i++) max = MathF.Max(max, MathF.Abs(ap[i] - bp[i]));
        return max;
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
