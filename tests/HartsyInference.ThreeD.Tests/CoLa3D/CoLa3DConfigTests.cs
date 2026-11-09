using HartsyInference.ThreeD.Models.CoLa3D;
using Xunit;

namespace HartsyInference.ThreeD.Tests.CoLa3D;

public sealed class CoLa3DConfigTests
{
    [Fact]
    public void FreeQ20Config_IsValidAndDerivesTheCheckpointShapes()
    {
        CoLa3DConfig config = CoLa3DTestSupport.FreeQ20();
        config.Validate();

        Assert.Equal(64, config.HeadDim);
        Assert.Equal(3072, config.MlpHidden);
    }

    [Theory]
    [InlineData(768, 10)]
    [InlineData(16, 3)]
    public void Validate_RejectsHiddenDimsNotDivisibleByHeads(int hidden, int heads)
    {
        CoLa3DConfig config = CoLa3DTestSupport.Tiny(CoLa3DConfig.QuerySourceKind.Fixed) with { HiddenDim = hidden, NumHeads = heads };
        Assert.Throws<ArgumentException>(() => config.Validate());
    }

    [Theory]
    [InlineData(8, 2)]
    [InlineData(10, 2)]
    public void Validate_RejectsHeadDimsBelowSixOrOdd(int hidden, int heads)
    {
        CoLa3DConfig config = CoLa3DTestSupport.Tiny(CoLa3DConfig.QuerySourceKind.Fixed) with { HiddenDim = hidden, NumHeads = heads };
        Assert.Throws<ArgumentException>(() => config.Validate());
    }

    [Fact]
    public void Validate_RejectsZeroSlotsAndBadMaskWidths()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Validated(CoLa3DTestSupport.FreeQ20() with { MaxParts = 0 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => Validated(CoLa3DTestSupport.MaskQ20() with { MaskEncoderHidden = 6 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => Validated(CoLa3DTestSupport.MaskQ20() with { MaskPePerAxis = 3 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => Validated(CoLa3DTestSupport.FreeQ20() with { MlpRatio = double.NaN }));
    }

    private static CoLa3DConfig Validated(CoLa3DConfig config)
    {
        config.Validate();
        return config;
    }
}
