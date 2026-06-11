using Xunit;
using Xunit.Abstractions;
using SharpInference.Cpu;
using SharpInference.Diffusion.Models.Denoisers;
using SharpInference.Diffusion.Models.Vae;
using SharpInference.Interactive.ActionEncoders;
using SharpInference.Interactive.Pipelines;
using SharpInference.Tests.Common;

namespace SharpInference.Interactive.Tests;

/// <summary>End-to-end structural test of the Oasis autoregressive Diffusion-Forcing loop on CPU with tiny synthetic
/// weights: RGB prompt in → per-frame DDIM denoise (context held at the stabilization level) → RGB frames out,
/// deterministic for a fixed seed. Numerics vs the reference checkpoint are validation-pending.</summary>
public class OasisPipelineTests
{
    private readonly ITestOutputHelper _output;
    public OasisPipelineTests(ITestOutputHelper output) => _output = output;

    private static (OasisDit Dit, OasisVitVae Vae) BuildTiny()
    {
        OasisDitConfig cfg = new()
        {
            HiddenSize = 32, NumHeads = 2, Depth = 2, InChannels = 16, PatchSize = 2,
            FreqDim = 16, MaxFrames = 8, SpatialRopeDimPerAxis = 8, SpatialRopeMaxFreq = 256,
        };
        OasisDit dit = new(cfg);
        dit.LoadWeights(OasisSyntheticWeights.BuildDit(cfg));
        OasisVitVae vae = new(latentDim: 16, patchSize: 4, dim: 32, heads: 2, encDepth: 2, decDepth: 2);
        vae.LoadWeights(OasisSyntheticWeights.BuildVitVae(16, 4, 32, 2, 2));
        return (dit, vae);
    }

    [Fact]
    public void Generate_AutoregressiveRollout_IsDeterministicPerSeed()
    {
        CpuBackend backend = new();
        (OasisDit dit, OasisVitVae vae) = BuildTiny();
        OasisPipeline pipeline = new(backend, dit, vae);

        const int width = 16, height = 16, totalFrames = 3;
        byte[] prompt = new byte[width * height * 3];
        new Random(2).NextBytes(prompt);

        float[][] actions = new float[totalFrames - 1][];
        byte[] keys = new byte[OasisActionEncoder.KeyCount];
        keys[11] = 1;   // forward
        for (int i = 0; i < actions.Length; i++)
        {
            actions[i] = new float[OasisActionEncoder.ActionDim];
            OasisActionEncoder.BuildRow(keys, cameraX: 0.1f, cameraY: 0f, actions[i]);
        }

        (byte[][] framesA, int w, int h, int seed) = pipeline.Generate(prompt, width, height, actions, totalFrames,
            ddimSteps: 3, seed: 42, onProgress: p => _output.WriteLine($"frame {p.Step}/{p.TotalSteps}"));

        Assert.Equal(totalFrames, framesA.Length);
        Assert.Equal(width, w);
        Assert.Equal(height, h);
        foreach (byte[] f in framesA) Assert.Equal(width * height * 3, f.Length);
        Assert.Equal(42, seed);

        (byte[][] framesB, _, _, _) = pipeline.Generate(prompt, width, height, actions, totalFrames, ddimSteps: 3, seed: 42);
        for (int i = 0; i < totalFrames; i++)
            Assert.Equal(framesA[i], framesB[i]);   // byte-exact replay

        (byte[][] framesC, _, _, _) = pipeline.Generate(prompt, width, height, actions, totalFrames, ddimSteps: 3, seed: 43);
        bool anyDiff = false;
        for (int i = 1; i < totalFrames && !anyDiff; i++)
            anyDiff = !framesC[i].AsSpan().SequenceEqual(framesA[i]);
        Assert.True(anyDiff, "a different seed must change the generated frames");
    }

    [Fact]
    public void Generate_RejectsBadInputs()
    {
        CpuBackend backend = new();
        (OasisDit dit, OasisVitVae vae) = BuildTiny();
        OasisPipeline pipeline = new(backend, dit, vae);
        byte[] prompt = new byte[16 * 16 * 3];

        Assert.Throws<ArgumentException>(() => pipeline.Generate(prompt, 16, 16, [], totalFrames: 3));
        Assert.Throws<ArgumentException>(() => pipeline.Generate(prompt, 15, 16, [new float[25], new float[25]], totalFrames: 3));
    }
}
