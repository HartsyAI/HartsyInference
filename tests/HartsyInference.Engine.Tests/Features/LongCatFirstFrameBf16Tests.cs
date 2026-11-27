using HartsyInference.Engine.Recipes.Video;
using Xunit;

namespace HartsyInference.Engine.Tests.Features;

/// <summary>The first-frame clip is rounded to bf16 before the VAE encode, as the reference does (pipeline_longcat_video_avatar.py:1152).</summary>
[Trait("Category", "SyntheticSmoke")]
public sealed class LongCatFirstFrameBf16Tests
{
    [Fact]
    public void RoundsToTheNearestBfloat16WithTiesToEven()
    {
        float[] values = [0.1f, 1f + 1f / 256f, 1f + 3f / 256f, -0.5f, 0f];
        LongCatRecipeSupport.RoundToBf16(values);
        Assert.Equal(0.10009765625f, values[0]);   // 0x3DCCCCCD rounds up to 0x3DCD
        Assert.Equal(1f, values[1]);               // a tie with an even upper half stays
        Assert.Equal(1.015625f, values[2]);        // a tie with an odd upper half rounds up
        Assert.Equal(-0.5f, values[3]);
        Assert.Equal(0f, values[4]);
    }
}
