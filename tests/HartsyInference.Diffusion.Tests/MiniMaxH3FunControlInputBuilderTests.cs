using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Locks the published Fun ControlNet input contract before the branch projection consumes it.</summary>
public unsafe class MiniMaxH3FunControlInputBuilderTests
{
    [Fact]
    public void PureControlZeroPadsVisibilityAndMaskedSourceChannels()
    {
        using Tensor control = ChannelTensor(24, 1, 2, 2, channel => channel + 1f);
        using Tensor rows = MiniMaxH3FunControlInputBuilder.Build(control, null, null);

        Assert.Equal(new TensorShape(1, 196), rows.Shape);
        float* pointer = (float*)rows.DataPointer;
        for (int channel = 0; channel < 24; channel++)
        {
            for (int pixel = 0; pixel < 4; pixel++)
            {
                Assert.Equal(channel + 1f, pointer[channel * 4 + pixel]);
            }
        }
        for (int offset = 24 * 4; offset < 196; offset++)
        {
            Assert.Equal(0f, pointer[offset]);
        }
    }

    [Fact]
    public void InpaintUsesControlVisibilityMaskedSourceChannelOrder()
    {
        using Tensor control = ChannelTensor(24, 1, 2, 2, channel => 100f + channel);
        using Tensor visibility = ChannelTensor(1, 1, 2, 2, _ => 0.25f);
        using Tensor source = ChannelTensor(24, 1, 2, 2, channel => 200f + channel);
        using Tensor rows = MiniMaxH3FunControlInputBuilder.Build(control, visibility, source);

        float* pointer = (float*)rows.DataPointer;
        Assert.Equal(100f, pointer[0]);
        Assert.Equal(123f, pointer[23 * 4]);
        for (int pixel = 0; pixel < 4; pixel++)
        {
            Assert.Equal(0.25f, pointer[24 * 4 + pixel]);
        }
        Assert.Equal(200f, pointer[25 * 4]);
        Assert.Equal(223f, pointer[48 * 4]);
    }

    [Fact]
    public void OddAxesCircularPadLikeTheBasePatchifier()
    {
        using Tensor control = ChannelTensor(24, 1, 1, 1, channel => channel + 0.5f);
        using Tensor rows = MiniMaxH3FunControlInputBuilder.Build(control, null, null);

        float* pointer = (float*)rows.DataPointer;
        for (int pixel = 0; pixel < 4; pixel++)
        {
            Assert.Equal(0.5f, pointer[pixel]);
        }
    }

    [Fact]
    public void ControlWindowIsInclusiveAndZeroStrengthIsAnExactBypass()
    {
        using Tensor rows = new Tensor(new TensorShape(1, 196), DType.F32);
        MiniMaxH3FunControlCondition active = new MiniMaxH3FunControlCondition
        {
            ModelIndex = 0,
            ControlRows = rows,
            Strength = 1f,
            Start = 0.25f,
            End = 0.75f,
        };
        Assert.False(active.IsActive(0, 5));
        Assert.True(active.IsActive(1, 5));
        Assert.True(active.IsActive(3, 5));
        Assert.False(active.IsActive(4, 5));

        MiniMaxH3FunControlCondition bypass = active with { Strength = 0f };
        Assert.False(bypass.IsActive(2, 5));
    }

    private static Tensor ChannelTensor(int channels, int frames, int height, int width,
        Func<int, float> value)
    {
        Tensor tensor = new Tensor(
            new TensorShape([1L, channels, frames, height, width]), DType.F32);
        float* pointer = (float*)tensor.DataPointer;
        int spatial = frames * height * width;
        for (int channel = 0; channel < channels; channel++)
        {
            for (int index = 0; index < spatial; index++)
            {
                pointer[channel * spatial + index] = value(channel);
            }
        }
        return tensor;
    }
}
