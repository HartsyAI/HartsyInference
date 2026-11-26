using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Pipelines.LongCat;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The uncached step's model input: the clip of the conditioning and stepped frames, reshaped. The clip is freed inside
/// LongCatFrameOps.ModelInput; these tests check the values and the shape match the two-step path it replaces.</summary>
public sealed class LongCatModelInputTests
{
    private const int Channels = 4, Hw = 6, Frames = 5, Cond = 2;

    [Theory]
    [InlineData(0)]
    [InlineData(Cond)]
    public void ModelInputMatchesTheClipThenReshape(int cond)
    {
        using CpuBackend backend = new();
        int stepped = Frames - cond;
        float[] steppedValues = Values(1, Channels * stepped * Hw), condValues = Values(2, Channels * cond * Hw);
        using Tensor steppedState = LongCatFrameOps.FromHost(steppedValues, new TensorShape(Channels * stepped, Hw));
        using Tensor condState = cond > 0 ? LongCatFrameOps.FromHost(condValues, new TensorShape(Channels * cond, Hw)) : null!;
        TensorShape shape = new([1, Channels, Frames, 1, Hw]);

        using Tensor expected = LongCatFrameOps.Reshape(backend, LongCatFrameOps.Clip(backend, steppedState, cond > 0 ? condState : null, Channels, cond, stepped, Hw), shape);
        using Tensor actual = LongCatFrameOps.ModelInput(backend, steppedState, cond > 0 ? condState : null, Channels, cond, stepped, Hw, shape);

        Assert.Equal(shape, actual.Shape);
        Assert.Equal(expected.AsReadOnlySpan<float>().ToArray(), actual.AsReadOnlySpan<float>().ToArray());
    }

    private static float[] Values(int seed, int count)
    {
        Random rng = new(seed);
        float[] values = new float[count];
        for (int i = 0; i < count; i++)
            values[i] = (float)(rng.NextDouble() * 2 - 1);
        return values;
    }
}
