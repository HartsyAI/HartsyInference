using Xunit;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Tests.Common;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Tiny-config CPU tests for the Matrix-Game 2.0 stack: the DMD few-step scheduler's analytic exactness,
/// the block-causal + local-window mask semantics, the Wan2.1 VAE round trip (channel-halving decoder), and the
/// causal DiT forward (CLIP context / action sensitivity). Numerics validation-pending.</summary>
public unsafe class MatrixGame2ModelTests
{
    private static MatrixGame2Config TinyConfig => new()
    {
        NumHeads = 2, HeadDim = 12, NumLayers = 2, FfnDim = 32,
        InChannels = 36, OutChannels = 16, ClipContextDim = 16, FreqDim = 16,
        ActionStreamDim = 128, ActionHeads = 2, ActionHiddenSize = 8,   // ActionModule headDim 64 = rope [8,28,28]
        KeyboardDim = 4, ActionBlocks = [1], LocalAttnSize = 2, NumFramePerBlock = 3,
        DenoisingStepList = [1000, 500],
    };

    [Fact]
    public void DmdScheduler_WarpsSigmasAndIsAnalyticallyExact()
    {
        FlowMatchDmdScheduler sched = FlowMatchDmdScheduler.MatrixGame2ThreeStep();
        Assert.Equal(3, sched.NumSteps);
        Assert.Equal(1.0f, sched.Sigma(0), 5);                          // t=1000 → σ=1 (shift fixes 1)
        float t1 = 666f / 1000f;
        Assert.Equal(5f * t1 / (1f + 4f * t1), sched.Sigma(1), 5);      // warped grid

        // x_t = (1−σ)x₀ + σ·ε with v = ε − x₀ ⇒ ToX0 must land exactly on x₀; AddNoise re-noises exactly.
        const float x0 = 1.5f, eps = -0.75f;
        Tensor sample = new Tensor(new TensorShape(1), DType.F32);
        Tensor v = new Tensor(new TensorShape(1), DType.F32);
        float sigma = sched.Sigma(1);
        *(float*)sample.DataPointer = (1 - sigma) * x0 + sigma * eps;
        *(float*)v.DataPointer = eps - x0;
        sched.ToX0(sample, v, 1);
        Assert.Equal(x0, *(float*)sample.DataPointer, 4);

        Tensor fresh = new Tensor(new TensorShape(1), DType.F32);
        *(float*)fresh.DataPointer = eps;
        sched.AddNoise(sample, fresh, 2);
        float sigma2 = sched.Sigma(2);
        Assert.Equal((1 - sigma2) * x0 + sigma2 * eps, *(float*)sample.DataPointer, 4);
        sample.Dispose(); v.Dispose(); fresh.Dispose();
    }

    [Fact]
    public void BlockCausalMask_EnforcesCausalityAndWindow()
    {
        // 3 blocks of 1 frame each, window 2, 1 token per frame.
        int[] frames = [0, 1, 2, 3, 4, 5];
        Tensor mask = MatrixGame2Transformer.BuildBlockCausalMask(frames, tokensPerFrame: 1, localAttnSize: 2, framesPerBlock: 3);
        float* mp = (float*)mask.DataPointer;
        int s = frames.Length;
        float At(int i, int j) => mp[(long)i * s + j];

        Assert.Equal(0f, At(0, 2));      // bidirectional inside block 0
        Assert.Equal(0f, At(2, 0));
        Assert.True(At(0, 3) < -1e8f);   // block 0 cannot see block 1
        Assert.Equal(0f, At(3, 4));      // bidirectional inside block 1
        Assert.True(At(3, 0) < -1e8f);   // window 2 below block start (frame 3): frames ≥ 1 only
        Assert.Equal(0f, At(3, 1));
        Assert.Equal(0f, At(3, 2));
        mask.Dispose();
    }

    [Fact]
    public void Wan21Vae_EncodeDecode_RoundTripsShapes()
    {
        CpuBackend backend = new();
        int[] dimMult = [1, 2, 4, 4];
        Wan21VaeDecoder dec = new(dim: 8, zDim: 16, dimMult: dimMult, numResBlocks: 2, temperalUpsample: [true, true, false]);
        dec.LoadWeights(MatrixGame2SyntheticWeights.BuildVae21Decoder(8, 16, dimMult, 2, [true, true, false]));
        Wan21VaeEncoder enc = new(dim: 8, zDim: 16, dimMult: dimMult, numResBlocks: 2, temperalDownsample: [false, true, true]);
        enc.LoadWeights(MatrixGame2SyntheticWeights.BuildVae21Encoder(8, 16, dimMult, 2, [false, true, true]));

        Tensor rgb = Rand5d(1, 3, 1, 32, 32, seed: 5);
        Tensor latent = enc.EncodeFrame(backend, rgb);
        Assert.Equal(16, (int)latent.Shape[1]);
        Assert.Equal(4, (int)latent.Shape[3]);     // 32 / 8
        Assert.Equal(4, (int)latent.Shape[4]);

        Tensor decoded = dec.Decode(backend, latent);
        Assert.Equal(3, (int)decoded.Shape[1]);
        Assert.Equal(1, (int)decoded.Shape[2]);
        Assert.Equal(32, (int)decoded.Shape[3]);
        Assert.Equal(32, (int)decoded.Shape[4]);

        // Multi-frame streaming decode: T latent frames → (T−1)·4 + 1 RGB frames.
        Tensor clip = Rand5d(1, 16, 3, 4, 4, seed: 6);
        Tensor video = dec.Decode(backend, clip);
        Assert.Equal(9, (int)video.Shape[2]);
        float* p = (float*)video.DataPointer;
        for (long i = 0; i < video.Shape.ElementCount; i++) Assert.True(float.IsFinite(p[i]));
    }

    [Fact]
    public void Transformer_Forward_RespondsToContextActionsAndClip()
    {
        CpuBackend backend = new();
        MatrixGame2Config cfg = TinyConfig;
        using MatrixGame2Transformer transformer = new(cfg);
        transformer.LoadWeights(MatrixGame2SyntheticWeights.BuildTransformer(cfg));

        int ctx = 2, block = 3, included = ctx + block;
        Tensor latent = Rand5d(1, 36, included, 4, 4, seed: 11);
        Tensor clipA = RandRows(5, cfg.ClipContextDim, seed: 12);
        Tensor mouse = RandRows(included * 4, 2, seed: 13);
        Tensor keyboard = RandRows(included * 4, 4, seed: 14);
        float[] frameTs = [0f, 0f, 700f, 700f, 700f];
        int[] ropeIdx = [3, 4, 5, 6, 7];

        Tensor v1 = transformer.Forward(backend, latent, clipA, frameTs, ropeIdx, outputFrames: block, mouse, keyboard);
        Assert.Equal(16, (int)v1.Shape[1]);
        Assert.Equal(block, (int)v1.Shape[2]);     // context dropped from the readout
        float* a = (float*)v1.DataPointer;
        for (long i = 0; i < v1.Shape.ElementCount; i++) Assert.True(float.IsFinite(a[i]));

        // CLIP context conditions the prediction (I2V cross-attention).
        Tensor clipB = RandRows(5, cfg.ClipContextDim, seed: 99);
        Tensor v2 = transformer.Forward(backend, latent, clipB, frameTs, ropeIdx, block, mouse, keyboard);
        Assert.True(MaxDiff(v1, v2) > 1e-6f, "different CLIP context must change the prediction");

        // Keyboard actions condition the prediction (ActionModule on block 1).
        Tensor keyboard2 = RandRows(included * 4, 4, seed: 98);
        Tensor v3 = transformer.Forward(backend, latent, clipA, frameTs, ropeIdx, block, mouse, keyboard2);
        Assert.True(MaxDiff(v1, v3) > 1e-6f, "different keyboard input must change the prediction");
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

    private static Tensor Rand5d(int a, int b, int c, int d, int e, int seed)
    {
        Tensor x = new Tensor(new TensorShape([(long)a, b, c, d, e]), DType.F32);
        float* p = (float*)x.DataPointer;
        Random rng = new(seed);
        for (long i = 0; i < x.Shape.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 2 - 1);
        return x;
    }
}
