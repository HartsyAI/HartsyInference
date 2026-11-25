using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Vision.CrossFrame;
using Xunit;

namespace HartsyInference.Vision.Tests.CrossFrame;

/// <summary>Join, Split and RestorePrefix (backend strided copies) against the plain host loops they replaced.</summary>
public sealed class CrossFrameTokensTests : IDisposable
{
    private const int Prefix = 3, Patches = 5, Width = 4;

    private readonly CpuBackend _backend = new();

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(5)]
    public void JoinAveragesThePrefixAndStacksThePatches(int count)
    {
        float[] values = Random(new Random(3), count * (Prefix + Patches) * Width);
        using Tensor frames = Make(values, count, Prefix + Patches, Width);
        using Tensor joint = CrossFrameTokens.Join(_backend, frames, Prefix);

        Assert.Equal(new TensorShape(1, Prefix + count * Patches, Width), joint.Shape);
        ReadOnlySpan<float> actual = joint.AsReadOnlySpan<float>();
        for (int i = 0; i < Prefix * Width; i++)
        {
            float mean = 0;
            for (int t = 0; t < count; t++) mean += values[t * (Prefix + Patches) * Width + i];
            Assert.True(Math.Abs(actual[i] - mean / count) <= 1e-6f, $"prefix value {i}");
        }

        for (int t = 0; t < count; t++)
            for (int i = 0; i < Patches * Width; i++)
                Assert.Equal(values[(t * (Prefix + Patches) + Prefix) * Width + i], actual[(Prefix + t * Patches) * Width + i]);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(3)]
    public void SplitTakesPatchesFromTheJointOutputAndThePrefixFromTheInput(int count)
    {
        Random rng = new(4);
        float[] input = Random(rng, count * (Prefix + Patches) * Width), output = Random(rng, (Prefix + count * Patches) * Width);
        using Tensor frames = Make(input, count, Prefix + Patches, Width);
        using Tensor joint = Make(output, 1, Prefix + count * Patches, Width);
        using Tensor split = CrossFrameTokens.Split(_backend, joint, frames, Prefix);

        Assert.Equal(frames.Shape, split.Shape);
        ReadOnlySpan<float> actual = split.AsReadOnlySpan<float>();
        for (int t = 0; t < count; t++)
        {
            int frame = t * (Prefix + Patches) * Width;
            for (int i = 0; i < Prefix * Width; i++) Assert.Equal(input[frame + i], actual[frame + i]);
            for (int i = 0; i < Patches * Width; i++) Assert.Equal(output[(Prefix + t * Patches) * Width + i], actual[frame + Prefix * Width + i]);
        }
    }

    [Fact]
    public void SplitOfAJoinOfUnchangedFramesReturnsTheFrames()
    {
        float[] values = Random(new Random(5), 2 * (Prefix + Patches) * Width);
        using Tensor frames = Make(values, 2, Prefix + Patches, Width);
        using Tensor joint = CrossFrameTokens.Join(_backend, frames, Prefix);
        using Tensor back = CrossFrameTokens.Split(_backend, joint, frames, Prefix);
        Assert.Equal(values, back.AsReadOnlySpan<float>().ToArray());
    }

    [Fact]
    public void RestorePrefixCopiesOnlyThePrefixRowsOfEveryItem()
    {
        Random rng = new(6);
        float[] target = Random(rng, 3 * (Prefix + Patches) * Width), source = Random(rng, 3 * (Prefix + Patches) * Width);
        using Tensor t = Make(target, 3, Prefix + Patches, Width);
        using Tensor s = Make(source, 3, Prefix + Patches, Width);
        CrossFrameTokens.RestorePrefix(_backend, t, s, Prefix);

        ReadOnlySpan<float> actual = t.AsReadOnlySpan<float>();
        for (int b = 0; b < 3; b++)
            for (int i = 0; i < (Prefix + Patches) * Width; i++)
            {
                int at = b * (Prefix + Patches) * Width + i;
                Assert.Equal(i < Prefix * Width ? source[at] : target[at], actual[at]);
            }
    }

    [Fact]
    public void RestorePrefixRefusesMismatchedTensors()
    {
        using Tensor a = new(new TensorShape(1, 8, 4), DType.F32);
        using Tensor b = new(new TensorShape(1, 9, 4), DType.F32);
        Assert.Throws<ArgumentException>(() => CrossFrameTokens.RestorePrefix(_backend, a, b, 2));
        Assert.Throws<ArgumentException>(() => CrossFrameTokens.RestorePrefix(_backend, a, a, 9));
    }

    public void Dispose() => _backend.Dispose();

    private static float[] Random(Random rng, int count)
    {
        float[] values = new float[count];
        for (int i = 0; i < count; i++) values[i] = (float)(rng.NextDouble() * 2 - 1);
        return values;
    }

    private static Tensor Make(float[] values, params long[] dims)
    {
        Tensor t = new(new TensorShape(dims), DType.F32);
        values.CopyTo(t.AsSpan<float>());
        return t;
    }
}
