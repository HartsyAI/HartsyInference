using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Shape, refusal and conditioning-frame behaviour of the avatar audio cross-attention on a tiny geometry. The numbers are checked against
/// the reference class in <c>LongCatAvatarAudioAttnParityTests</c>.</summary>
[Trait("Category", "SyntheticSmoke")]
public sealed class LongCatAudioCrossAttentionTests
{
    private const int Dim = 16, Heads = 2, Enc = 12, Ctx = 5, Frames = 4, Tokens = 6;

    [Fact]
    public void ConditioningFramesGetZerosAndTheNoisyFramesGetTheAttention()
    {
        using LongCatAudioCrossAttention layer = new(Dim, Enc, Heads);
        Dictionary<string, Tensor> weights = Weights(seed: 3);
        try
        {
            layer.LoadWeights(weights);
            using Tensor x = Filled(Frames * Tokens, Dim, seed: 4);
            using Tensor audio = Filled(Frames * Ctx, Enc, seed: 5);
            using CpuBackend backend = new();

            using Tensor output = layer.Forward(backend, x, audio, Frames, numCond: 1);

            Assert.Equal(Frames * Tokens, output.Shape[0]);
            Assert.Equal(Dim, output.Shape[1]);
            ReadOnlySpan<float> values = output.AsReadOnlySpan<float>();
            for (int i = 0; i < Tokens * Dim; i++)
                Assert.Equal(0f, values[i]);
            Assert.Contains(values[(Tokens * Dim)..].ToArray(), v => v != 0f);
        }
        finally
        {
            DisposeAll(weights);
        }
    }

    [Fact]
    public void ForwardRefusesAnUnloadedLayer()
    {
        using LongCatAudioCrossAttention layer = new(Dim, Enc, Heads);
        using Tensor x = Filled(Frames * Tokens, Dim, seed: 6);
        using Tensor audio = Filled(Frames * Ctx, Enc, seed: 7);
        using CpuBackend backend = new();

        Assert.Throws<InvalidOperationException>(() => layer.Forward(backend, x, audio, Frames, numCond: 0));
    }

    [Fact]
    public void ForwardRefusesAllConditioningAndWrongWidths()
    {
        using LongCatAudioCrossAttention layer = new(Dim, Enc, Heads);
        Dictionary<string, Tensor> weights = Weights(seed: 8);
        try
        {
            layer.LoadWeights(weights);
            using Tensor x = Filled(Frames * Tokens, Dim, seed: 9);
            using Tensor audio = Filled(Frames * Ctx, Enc, seed: 10);
            using Tensor narrow = Filled(Frames * Tokens, Dim - 1, seed: 11);
            using CpuBackend backend = new();

            Assert.Throws<ArgumentOutOfRangeException>(() => layer.Forward(backend, x, audio, Frames, numCond: Frames));
            Assert.Throws<ArgumentException>(() => layer.Forward(backend, narrow, audio, Frames, numCond: 0));
        }
        finally
        {
            DisposeAll(weights);
        }
    }

    [Fact]
    public void ConstructorRefusesAHeadCountThatDoesNotDivideTheWidth()
    {
        Assert.Throws<ArgumentException>(() => new LongCatAudioCrossAttention(Dim, Enc, 3));
    }

    [Fact]
    public void AFailedLoadKeepsThePreviousWeightsAndTheLayerStillRuns()
    {
        using LongCatAudioCrossAttention layer = new(Dim, Enc, Heads);
        Dictionary<string, Tensor> good = Weights(seed: 30);
        Dictionary<string, Tensor> bad = Weights(seed: 31);
        try
        {
            layer.LoadWeights(good);
            // A wrong-shaped k norm: the whole load is refused, and the previous set stays in use.
            bad["attn.k_norm.weight"].Dispose();
            bad["attn.k_norm.weight"] = Filled(Dim / Heads + 1, 0, seed: 32, rank1: true);
            Assert.Throws<InvalidDataException>(() => layer.LoadWeights(bad));

            using Tensor x = Filled(Frames * Tokens, Dim, seed: 33);
            using Tensor audio = Filled(Frames * Ctx, Enc, seed: 34);
            using CpuBackend backend = new();
            using Tensor output = layer.Forward(backend, x, audio, Frames, numCond: 0);
            Assert.Equal(Frames * Tokens, output.Shape[0]);
        }
        finally
        {
            DisposeAll(good);
            DisposeAll(bad);
        }
    }

    [Fact]
    public void ForwardRefusesAWrongAudioWidthAndARowCountThatIsNotAMultipleOfFrames()
    {
        using LongCatAudioCrossAttention layer = new(Dim, Enc, Heads);
        Dictionary<string, Tensor> weights = Weights(seed: 35);
        try
        {
            layer.LoadWeights(weights);
            using Tensor x = Filled(Frames * Tokens, Dim, seed: 36);
            using Tensor narrowAudio = Filled(Frames * Ctx, Enc - 1, seed: 37);
            using Tensor ragged = Filled(7, Enc, seed: 38);
            using CpuBackend backend = new();

            Assert.Throws<ArgumentException>(() => layer.Forward(backend, x, narrowAudio, Frames, numCond: 0));
            ArgumentException error = Assert.Throws<ArgumentException>(() => layer.Forward(backend, x, ragged, Frames, numCond: 0));
            Assert.Equal("audio", error.ParamName);
        }
        finally
        {
            DisposeAll(weights);
        }
    }

    [Fact]
    public void ForwardRefusesANonF32InputAndAfterDispose()
    {
        LongCatAudioCrossAttention layer = new(Dim, Enc, Heads);
        Dictionary<string, Tensor> weights = Weights(seed: 39);
        try
        {
            layer.LoadWeights(weights);
            using Tensor half = new(new TensorShape(Frames * Tokens, Dim), DType.F16);
            using Tensor audio = Filled(Frames * Ctx, Enc, seed: 40);
            using CpuBackend backend = new();

            ArgumentException error = Assert.Throws<ArgumentException>(() => layer.Forward(backend, half, audio, Frames, numCond: 0));
            Assert.Equal("x", error.ParamName);

            layer.Dispose();
            using Tensor x = Filled(Frames * Tokens, Dim, seed: 41);
            Assert.Throws<ObjectDisposedException>(() => layer.Forward(backend, x, audio, Frames, numCond: 0));
        }
        finally
        {
            DisposeAll(weights);
        }
    }

    private static Dictionary<string, Tensor> Weights(int seed) => new()
    {
        ["attn.q_linear.weight"] = Filled(Dim, Dim, seed + 1),
        ["attn.q_linear.bias"] = Filled(Dim, 1, seed + 2, rank1: true),
        ["attn.q_norm.weight"] = Filled(Dim / Heads, 1, seed + 3, rank1: true),
        ["attn.kv_linear.weight"] = Filled(2 * Dim, Enc, seed + 4),
        ["attn.kv_linear.bias"] = Filled(2 * Dim, 1, seed + 5, rank1: true),
        ["attn.k_norm.weight"] = Filled(Dim / Heads, 1, seed + 6, rank1: true),
        ["attn.proj.weight"] = Filled(Dim, Dim, seed + 7),
        ["attn.proj.bias"] = Filled(Dim, 1, seed + 8, rank1: true),
    };

    /// <summary>A seeded F32 tensor: rank 2 for <c>rows × cols</c>, or rank 1 of <paramref name="rows"/> when <paramref name="rank1"/>.</summary>
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
