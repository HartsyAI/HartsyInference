using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.World.ActionEncoders;
using HartsyInference.World.Pipelines;
using HartsyInference.Tests.Common;

namespace HartsyInference.World.Tests;

/// <summary>End-to-end structural test of the Matrix-Game 2.0 autoregressive block loop on CPU: RGB seed → Wan2.1
/// VAE conditioning + channel-concat input → 2-step DMD denoise per block with sliding-window context → streamed
/// Wan2.1 decode. Numerics validation-pending.</summary>
public unsafe class MatrixGame2PipelineTests
{
    private readonly ITestOutputHelper _output;
    public MatrixGame2PipelineTests(ITestOutputHelper output) => _output = output;

    private static (MatrixGame2Transformer Dit, Wan21VaeDecoder Dec, Wan21VaeEncoder Enc, MatrixGame2Config Cfg) BuildTiny()
    {
        MatrixGame2Config cfg = new()
        {
            NumHeads = 2, HeadDim = 12, NumLayers = 2, FfnDim = 32,
            InChannels = 36, OutChannels = 16, ClipContextDim = 16, FreqDim = 16,
            ActionStreamDim = 128, ActionHeads = 2, ActionHiddenSize = 8,
            KeyboardDim = 4, ActionBlocks = [1], LocalAttnSize = 2, NumFramePerBlock = 3,
            DenoisingStepList = [1000, 500],
        };
        MatrixGame2Transformer dit = new(cfg);
        dit.LoadWeights(MatrixGame2SyntheticWeights.BuildTransformer(cfg));
        int[] dimMult = [1, 2, 4, 4];
        Wan21VaeDecoder dec = new(dim: 8, zDim: 16, dimMult: dimMult, numResBlocks: 2, temperalUpsample: [true, true, false]);
        dec.LoadWeights(MatrixGame2SyntheticWeights.BuildVae21Decoder(8, 16, dimMult, 2, [true, true, false]));
        Wan21VaeEncoder enc = new(dim: 8, zDim: 16, dimMult: dimMult, numResBlocks: 2, temperalDownsample: [false, true, true]);
        enc.LoadWeights(MatrixGame2SyntheticWeights.BuildVae21Encoder(8, 16, dimMult, 2, [false, true, true]));
        return (dit, dec, enc, cfg);
    }

    // SyntheticSmoke: full block rollout on synthetic weights; can hard-crash the CPU test host
    // (native heap corruption) until Matrix-Game 2 is CPU-validated. Runs on the GPU lane. See CODE_STYLE.
    [Trait("Category", "SyntheticSmoke")]
    [Fact]
    public void Generate_TwoBlocks_ProducesControllableRollout()
    {
        CpuBackend backend = new();
        (MatrixGame2Transformer dit, Wan21VaeDecoder dec, Wan21VaeEncoder enc, MatrixGame2Config cfg) = BuildTiny();
        MatrixGame2Pipeline pipeline = new(backend, dit, dec, enc, cfg);

        const int width = 32, height = 32, totalLatents = 6;   // 2 blocks of 3
        byte[] seedImage = new byte[width * height * 3];
        new Random(3).NextBytes(seedImage);

        int requiredActions = pipeline.RequiredActionFrames(totalLatents);
        Assert.Equal((totalLatents - 1) * 4 + 1, requiredActions);

        using MatrixGame2ActionEncoder encoder = MatrixGame2ActionEncoder.CreateUniversal();
        float[][] keyboard = new float[requiredActions][];
        float[][] mouse = new float[requiredActions][];
        for (int i = 0; i < requiredActions; i++)
        {
            keyboard[i] = [1, 0, 0, 0];                                      // hold forward
            mouse[i] = [0f, MatrixGame2ActionEncoder.CamValue];              // steady yaw right
        }

        Tensor clipContext = RandRows(5, cfg.ClipContextDim, seed: 7);
        (byte[][] frames, int w, int h, int seed) = pipeline.Generate(seedImage, width, height, clipContext,
            keyboard, mouse, totalLatents, seed: 42,
            p => _output.WriteLine($"block {p.Step}/{p.TotalSteps}"));

        Assert.Equal((totalLatents - 1) * 4 + 1, frames.Length);   // 21 frames
        Assert.Equal(width, w);
        Assert.Equal(height, h);
        foreach (byte[] f in frames) Assert.Equal(width * height * 3, f.Length);
        _ = seed;
    }

    [Fact]
    public void Generate_RejectsBadInputs_AndTempleRunSkipsMouse()
    {
        CpuBackend backend = new();
        (MatrixGame2Transformer dit, Wan21VaeDecoder dec, Wan21VaeEncoder enc, MatrixGame2Config cfg) = BuildTiny();
        MatrixGame2Pipeline pipeline = new(backend, dit, dec, enc, cfg);
        byte[] seedImage = new byte[32 * 32 * 3];
        Tensor clip = RandRows(5, cfg.ClipContextDim, seed: 8);

        Assert.Throws<ArgumentException>(() => pipeline.Generate(seedImage, 32, 32, clip,
            [new float[4]], [new float[2]], totalLatentFrames: 3));         // too few action rows
        Assert.Throws<ArgumentException>(() => pipeline.Generate(seedImage, 32, 32, clip,
            Rows(20, 4), Rows(20, 2), totalLatentFrames: 4));               // not a block multiple
        Assert.Throws<ArgumentException>(() => pipeline.Generate(seedImage, 32, 32, clip,
            Rows(20, 4), mouse: null, totalLatentFrames: 3));               // mouse required for universal

        // CLIP encoding without an encoder fails fast (precomputed contexts remain supported).
        Assert.Throws<InvalidOperationException>(() => pipeline.EncodeClipContext(seedImage, 32, 32));

        // Variant encoders expose the per-domain shapes.
        using MatrixGame2ActionEncoder gta = MatrixGame2ActionEncoder.CreateGtaDrive();
        Assert.Equal(2, gta.KeyboardDim);
        Assert.Equal(2, gta.Streams.Count);
        using MatrixGame2ActionEncoder temple = MatrixGame2ActionEncoder.CreateTempleRun();
        Assert.Equal(7, temple.KeyboardDim);
        Assert.Single(temple.Streams);                                       // keyboard only
        Assert.Throws<ArgumentException>(() => temple.Encode(
            new HartsyInference.Conditioning.ActionInput(new byte[7], 0, 0), "mouse", new float[2]));
    }

    private static float[][] Rows(int count, int dim)
    {
        float[][] rows = new float[count][];
        for (int i = 0; i < count; i++) rows[i] = new float[dim];
        return rows;
    }

    private static Tensor RandRows(int rows, int cols, int seed)
    {
        Tensor t = new Tensor(new TensorShape(rows, cols), DType.F32);
        float* p = (float*)t.DataPointer;
        Random rng = new(seed);
        for (long i = 0; i < t.Shape.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }
}
