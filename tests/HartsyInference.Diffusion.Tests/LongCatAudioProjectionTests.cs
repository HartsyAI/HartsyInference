using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Shape, refusal and all-or-nothing behaviour of the LongCat avatar audio projection on a tiny geometry. The numbers are checked
/// against the reference class in <c>LongCatAudioProjectionParityTests</c>.</summary>
[Trait("Category", "SyntheticSmoke")]
public sealed class LongCatAudioProjectionTests
{
    private const int SeqLen = 5, SeqLenVf = 8, Blocks = 3, Channels = 6, Intermediate = 16, OutputDim = 10, Context = 4;

    [Fact]
    public void ForwardReturnsContextTokensPerFrameOverBothWindows()
    {
        using LongCatAudioProjection projection = NewProjection();
        Dictionary<string, Tensor> weights = Weights(seed: 3);
        try
        {
            projection.LoadWeights(weights);
            using Tensor first = Filled(1, SeqLen * Blocks * Channels, seed: 4);
            using Tensor latter = Filled(3, SeqLenVf * Blocks * Channels, seed: 5);
            using CpuBackend backend = new();

            using Tensor output = projection.Forward(backend, first, latter);

            Assert.Equal(4 * Context, output.Shape[0]);
            Assert.Equal(OutputDim, output.Shape[1]);
            ReadOnlySpan<float> values = output.AsReadOnlySpan<float>();
            for (int i = 0; i < values.Length; i++)
                Assert.True(float.IsFinite(values[i]));
        }
        finally
        {
            DisposeAll(weights);
        }
    }

    [Fact]
    public void ForwardRefusesWhenNoWeightsAreLoaded()
    {
        using LongCatAudioProjection projection = NewProjection();
        using Tensor first = Filled(1, SeqLen * Blocks * Channels, seed: 1);
        using Tensor latter = Filled(3, SeqLenVf * Blocks * Channels, seed: 2);
        using CpuBackend backend = new();

        Assert.Throws<InvalidOperationException>(() => projection.Forward(backend, first, latter));
    }

    [Fact]
    public void ForwardRefusesAWindowOfTheWrongWidth()
    {
        using LongCatAudioProjection projection = NewProjection();
        Dictionary<string, Tensor> weights = Weights(seed: 6);
        try
        {
            projection.LoadWeights(weights);
            using Tensor narrow = Filled(1, SeqLen * Blocks * Channels - 1, seed: 7);
            using Tensor latter = Filled(3, SeqLenVf * Blocks * Channels, seed: 8);
            using CpuBackend backend = new();

            Assert.Throws<ArgumentException>(() => projection.Forward(backend, narrow, latter));
        }
        finally
        {
            DisposeAll(weights);
        }
    }

    [Fact]
    public void ALoadWithAWrongShapeRefusesAndLeavesTheProjectionUnloaded()
    {
        using LongCatAudioProjection projection = NewProjection();
        Dictionary<string, Tensor> weights = Weights(seed: 9);
        try
        {
            // Swap in a wrong-shaped output norm: the load must refuse as a whole.
            weights["audio_proj.norm.weight"].Dispose();
            weights["audio_proj.norm.weight"] = Filled(OutputDim + 1, 0, seed: 10, rank1: true);

            Assert.Throws<InvalidDataException>(() => projection.LoadWeights(weights));

            using Tensor first = Filled(1, SeqLen * Blocks * Channels, seed: 11);
            using Tensor latter = Filled(3, SeqLenVf * Blocks * Channels, seed: 12);
            using CpuBackend backend = new();
            Assert.Throws<InvalidOperationException>(() => projection.Forward(backend, first, latter));
        }
        finally
        {
            DisposeAll(weights);
        }
    }

    private static LongCatAudioProjection NewProjection() =>
        new(SeqLen, SeqLenVf, Blocks, Channels, Intermediate, OutputDim, Context);

    private static Dictionary<string, Tensor> Weights(int seed)
    {
        int inFirst = SeqLen * Blocks * Channels, inLatter = SeqLenVf * Blocks * Channels;
        return new Dictionary<string, Tensor>
        {
            ["audio_proj.proj1.weight"] = Filled(Intermediate, inFirst, seed + 1),
            ["audio_proj.proj1.bias"] = Filled(Intermediate, 0, seed + 2, rank1: true),
            ["audio_proj.proj1_vf.weight"] = Filled(Intermediate, inLatter, seed + 3),
            ["audio_proj.proj1_vf.bias"] = Filled(Intermediate, 0, seed + 4, rank1: true),
            ["audio_proj.proj2.weight"] = Filled(Intermediate, Intermediate, seed + 5),
            ["audio_proj.proj2.bias"] = Filled(Intermediate, 0, seed + 6, rank1: true),
            ["audio_proj.proj3.weight"] = Filled(Context * OutputDim, Intermediate, seed + 7),
            ["audio_proj.proj3.bias"] = Filled(Context * OutputDim, 0, seed + 8, rank1: true),
            ["audio_proj.norm.weight"] = Filled(OutputDim, 0, seed + 9, rank1: true),
            ["audio_proj.norm.bias"] = Filled(OutputDim, 0, seed + 10, rank1: true),
        };
    }

    /// <summary>A rank-2 F32 tensor of <paramref name="rows"/> by <paramref name="cols"/> (or rank 1 of <paramref name="rows"/> when
    /// <paramref name="rank1"/>), filled from a seeded generator.</summary>
    private static Tensor Filled(int rows, int cols, int seed, bool rank1 = false)
    {
        Tensor tensor = rank1 ? new Tensor(new TensorShape(rows), DType.F32) : new Tensor(new TensorShape(rows, cols), DType.F32);
        Random rng = new(seed);
        Span<float> values = tensor.AsSpan<float>();
        for (int i = 0; i < values.Length; i++)
            values[i] = (float)(rng.NextDouble() * 0.4 - 0.2);
        return tensor;
    }

    private static void DisposeAll(Dictionary<string, Tensor> weights)
    {
        foreach (Tensor tensor in weights.Values)
            tensor.Dispose();
    }
}
