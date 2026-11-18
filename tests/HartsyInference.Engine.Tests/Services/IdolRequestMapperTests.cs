using HartsyInference.Diffusion.Requests;
using HartsyInference.Engine.Planning;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.ThreeD.Pipelines;
using HartsyInference.ThreeD.Preprocessing;
using Xunit;

namespace HartsyInference.Engine.Tests.Services;

public sealed class IdolRequestMapperTests
{
    private static ImageData Image() => new() { Rgb = new byte[4 * 4 * 3], Width = 4, Height = 4 };

    private static MeshRequest Request() => new() { Image = Image() };

    [Fact]
    public void ADefaultRequestTakesTheModelsDefaults()
    {
        IdolOptions options = IdolRequestMapper.ToOptions(Request() with { Seed = 7 });

        Assert.Equal(0, options.Steps);
        Assert.Null(options.Guidance);
        Assert.Equal(0, options.Splats);
        Assert.Equal(IdolFraming.Object, options.Framing);
        Assert.Equal(7, options.Seed);
    }

    [Fact]
    public void TheFieldsMapThrough()
    {
        IdolOptions options = IdolRequestMapper.ToOptions(Request() with { Steps = 12, Guidance = 0, SplatCount = 65536, Framing = MeshFraming.Photo, Seed = 3 });

        Assert.Equal(12, options.Steps);
        Assert.Equal(0f, options.Guidance);
        Assert.Equal(65536, options.Splats);
        Assert.Equal(IdolFraming.Photo, options.Framing);
    }

    [Theory]
    [InlineData(32767)]
    [InlineData(32769)]
    [InlineData(1048608)]
    [InlineData(40001)]
    public void ASplatCountOutOfRangeOrOffTheMultipleIsRejected(int splats)
    {
        ArgumentOutOfRangeException error = Assert.Throws<ArgumentOutOfRangeException>(() => IdolRequestMapper.Validate(Request() with { SplatCount = splats }));
        Assert.Equal(nameof(MeshRequest.SplatCount), MeshFieldTag.FieldOf(error));
    }

    [Theory]
    [InlineData(-0.5)]
    [InlineData(21)]
    public void AGuidanceOutOfRangeIsRejected(double guidance)
        => Assert.Equal(nameof(MeshRequest.Guidance), MeshFieldTag.FieldOf(Assert.Throws<ArgumentOutOfRangeException>(() => IdolRequestMapper.Validate(Request() with { Guidance = guidance }))));

    [Theory]
    [InlineData(-1)]
    [InlineData(1001)]
    [InlineData(100000)]
    public void AStepCountOutsideZeroToAThousandIsRejected(int steps)
        => Assert.Equal(nameof(MeshRequest.Steps), MeshFieldTag.FieldOf(Assert.Throws<ArgumentOutOfRangeException>(() => IdolRequestMapper.Validate(Request() with { Steps = steps }))));

    [Fact]
    public void AGridAFieldOfViewAndANegativeStepCountAreRejected()
    {
        Assert.Equal(nameof(MeshRequest.GridResolution), MeshFieldTag.FieldOf(Assert.Throws<NotSupportedException>(() => IdolRequestMapper.Validate(Request() with { GridResolution = 256 }))));
        Assert.Equal(nameof(MeshRequest.FieldOfViewRadians), MeshFieldTag.FieldOf(Assert.Throws<NotSupportedException>(() => IdolRequestMapper.Validate(Request() with { FieldOfViewRadians = 0.5 }))));
        Assert.Equal(nameof(MeshRequest.Steps), MeshFieldTag.FieldOf(Assert.Throws<ArgumentOutOfRangeException>(() => IdolRequestMapper.Validate(Request() with { Steps = -1 }))));
        Assert.Throws<NotSupportedException>(() => IdolRequestMapper.Validate(new MeshRequest()));
    }

    [Fact]
    public void OtherModelsRejectIdolsFields()
    {
        Assert.Equal(nameof(MeshRequest.Guidance), MeshFieldTag.FieldOf(Assert.Throws<NotSupportedException>(() => IdolRequestMapper.RejectForOtherModels(Request() with { Guidance = 3 }, "trellis2"))));
        Assert.Equal(nameof(MeshRequest.SplatCount), MeshFieldTag.FieldOf(Assert.Throws<NotSupportedException>(() => IdolRequestMapper.RejectForOtherModels(Request() with { SplatCount = 65536 }, "trellis2"))));
        Assert.Equal(nameof(MeshRequest.Framing), MeshFieldTag.FieldOf(Assert.Throws<NotSupportedException>(() => IdolRequestMapper.RejectForOtherModels(Request() with { Framing = MeshFraming.Photo }, "trellis2"))));
        IdolRequestMapper.RejectForOtherModels(Request() with { Guidance = 3, SplatCount = 65536, Framing = MeshFraming.Photo }, MeshModelIds.Idol);
        IdolRequestMapper.RejectForOtherModels(Request(), "trellis2");
    }

    [Fact]
    public void TheProfileDescribesIdolsRangesAndTheCatalogListsItsFiles()
    {
        MeshModelProfile profile = MeshModelProfiles.Find(MeshModelIds.Idol)!;
        MeshFieldSpec guidance = profile.Fields.Single(f => f.Name == nameof(MeshRequest.Guidance));
        MeshFieldSpec splats = profile.Fields.Single(f => f.Name == nameof(MeshRequest.SplatCount));

        Assert.Equal(MeshFieldAccess.Used, guidance.Access);
        Assert.Equal("3", guidance.Default);
        Assert.Equal(MeshFieldAccess.Used, splats.Access);
        Assert.Equal(32768, splats.Min);
        Assert.Equal(1048576, splats.Max);
        Assert.Equal(MeshOutputKinds.Splats, profile.Outputs);

        string[] files = [.. HartsyInference.Engine.Registry.ModelCatalog.Find(MeshModelIds.Idol)!.Assets!.Select(a => a.FileName)];
        foreach (string expected in IdolImageTo3DPipeline.RequiredFiles)
            Assert.Contains(expected.Split('/')[^1], files.Select(f => f.Split('/')[^1]));
    }
}
