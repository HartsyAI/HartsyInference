using HartsyInference.LLM.DeepSeekV41;
using Xunit;

namespace HartsyInference.LLM.Tests.DeepSeekV41;

public sealed class DeepSeekV41GroupedProjectionTests
{
    [Fact]
    public void Each_Group_Uses_Only_Its_Own_Weights()
    {
        // two groups, rank 2, groupDim 3: group 0 sums the inputs, group 1 negates the first input
        float[] weight = [1, 1, 1, 0, 0, 0, -1, 0, 0, 0, 0, 0];
        float[] x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 1, 1];
        float[] dest = new float[2 * 2 * 2];
        DeepSeekV41GroupedProjection.Apply(x, weight, 2, 2, 2, 3, dest);
        Assert.Equal([6f, 0f, -4f, 0f, 24f, 0f, -1f, 0f], dest);
    }

    [Fact]
    public void Rejects_Mismatched_Sizes()
    {
        Assert.Throws<ArgumentException>(() => DeepSeekV41GroupedProjection.Apply(new float[5], new float[12], 2, 2, 2, 3, new float[8]));
        Assert.Throws<ArgumentException>(() => DeepSeekV41GroupedProjection.Apply(new float[12], new float[11], 2, 2, 2, 3, new float[8]));
        Assert.Throws<ArgumentException>(() => DeepSeekV41GroupedProjection.Apply(new float[12], new float[12], 2, 2, 2, 3, new float[7]));
    }
}
