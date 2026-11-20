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
    public void IdolTurboIsTheSameRequestWithTheTurboFlagAndNoGuidance()
    {
        IdolOptions options = IdolRequestMapper.ToOptions(Request() with { Seed = 7, SplatCount = 65536 }, MeshModelIds.IdolTurbo);

        Assert.True(options.Turbo);
        Assert.Equal(0, options.Steps);   // the pipeline takes the model's 5
        Assert.False(IdolRequestMapper.ToOptions(Request()).Turbo);
        Assert.Equal(nameof(MeshRequest.Guidance), MeshFieldTag.FieldOf(Assert.Throws<NotSupportedException>(() => IdolRequestMapper.Validate(Request() with { Guidance = 0 }, MeshModelIds.IdolTurbo))));
        IdolRequestMapper.RejectForOtherModels(Request() with { SplatCount = 65536, Framing = MeshFraming.Photo }, MeshModelIds.IdolTurbo);
        Assert.Equal(nameof(MeshRequest.Guidance), MeshFieldTag.FieldOf(Assert.Throws<NotSupportedException>(() => IdolRequestMapper.RejectForOtherModels(Request() with { Guidance = 3 }, "trellis2"))));
    }

    [Fact]
    public void TheTurboProfileAndCatalogEntryShareIdolsFilesAndAddTheTwoLoras()
    {
        MeshModelProfile profile = MeshModelProfiles.Find(MeshModelIds.IdolTurbo)!;
        Assert.Equal(MeshOutputKinds.Splats, profile.Outputs);
        Assert.Equal(MeshFieldAccess.Rejected, profile.Fields.Single(f => f.Name == nameof(MeshRequest.Guidance)).Access);
        Assert.Equal(MeshFieldAccess.Used, profile.Fields.Single(f => f.Name == nameof(MeshRequest.SplatCount)).Access);
        Assert.Equal("5", profile.Fields.Single(f => f.Name == nameof(MeshRequest.Steps)).Default);

        string[] turbo = [.. HartsyInference.Engine.Registry.ModelCatalog.Find(MeshModelIds.IdolTurbo)!.Assets!.Select(a => a.TargetName ?? a.FileName)];
        string[] idol = [.. HartsyInference.Engine.Registry.ModelCatalog.Find(MeshModelIds.Idol)!.Assets!.Select(a => a.TargetName ?? a.FileName)];
        Assert.All(idol, file => Assert.Contains(file, turbo));
        Assert.Contains(IdolImageTo3DPipeline.TurboSlowFile, turbo);
        Assert.Contains(IdolImageTo3DPipeline.TurboFastFile, turbo);
        Assert.Empty(IdolImageTo3DPipeline.MissingFiles(Path.Combine(Path.GetTempPath(), "no-such-idol"), turbo: false).Except(IdolImageTo3DPipeline.RequiredFiles));
        Assert.Equal(2, IdolImageTo3DPipeline.MissingFiles(Path.Combine(Path.GetTempPath(), "no-such-idol"), turbo: true).Except(IdolImageTo3DPipeline.RequiredFiles).Count());
    }

    [Theory]
    [InlineData(MeshModelIds.BudgetIdol, false)]
    [InlineData(MeshModelIds.BudgetIdolTurbo, true)]
    public void ABudget3DRequestCarriesTheBudgetAndTheVariantFlags(string id, bool turbo)
    {
        IdolOptions options = IdolRequestMapper.ToOptions(Request() with { TokenBudget = 0.25, Seed = 3 }, id);

        Assert.True(options.Budget3D);
        Assert.Equal(turbo, options.Turbo);
        Assert.Equal(0.25, options.TokenBudget);
        Assert.Equal(0.0, IdolRequestMapper.ToOptions(Request(), id).TokenBudget);   // null takes the pipeline's 1
        Assert.False(IdolRequestMapper.ToOptions(Request(), MeshModelIds.Idol).Budget3D);
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(0.05)]
    [InlineData(1.01)]
    [InlineData(double.NaN)]
    public void ATokenBudgetOutsideTheRangeIsRejected(double budget)
        => Assert.Equal(nameof(MeshRequest.TokenBudget), MeshFieldTag.FieldOf(Assert.Throws<ArgumentOutOfRangeException>(() => IdolRequestMapper.Validate(Request() with { TokenBudget = budget }, MeshModelIds.BudgetIdol))));

    [Theory]
    [InlineData(MeshModelIds.Idol)]
    [InlineData(MeshModelIds.IdolTurbo)]
    [InlineData("trellis2")]
    public void OnlyTheBudget3DModelsTakeATokenBudget(string id)
    {
        Assert.Equal(nameof(MeshRequest.TokenBudget), MeshFieldTag.FieldOf(Assert.Throws<NotSupportedException>(() => IdolRequestMapper.RejectForOtherModels(Request() with { TokenBudget = 0.5 }, id))));
        IdolRequestMapper.RejectForOtherModels(Request() with { TokenBudget = 0.5 }, MeshModelIds.BudgetIdol);
        IdolRequestMapper.RejectForOtherModels(Request() with { TokenBudget = 0.5 }, MeshModelIds.BudgetIdolTurbo);
    }

    [Fact]
    public void TheBudget3DProfilesAndCatalogEntriesShareIdolsEncodersAndAddTheirCheckpoint()
    {
        foreach ((string id, string file) in new[] { (MeshModelIds.BudgetIdol, IdolImageTo3DPipeline.Budget3DFile), (MeshModelIds.BudgetIdolTurbo, IdolImageTo3DPipeline.Budget3DTurboFile) })
        {
            MeshModelProfile profile = MeshModelProfiles.Find(id)!;
            MeshFieldSpec budget = profile.Fields.Single(f => f.Name == nameof(MeshRequest.TokenBudget));
            Assert.Equal(MeshFieldAccess.Used, budget.Access);
            Assert.Equal(0.0625, budget.Min);
            Assert.Equal(1, budget.Max);
            Assert.Equal(MeshOutputKinds.Splats, profile.Outputs);

            string[] files = [.. HartsyInference.Engine.Registry.ModelCatalog.Find(id)!.Assets!.Select(a => a.TargetName ?? a.FileName)];
            Assert.Contains(file, files);
            Assert.DoesNotContain(IdolImageTo3DPipeline.DenoiserFile, files);   // the Budget3D checkpoint is the denoiser
            Assert.Contains(IdolImageTo3DPipeline.DinoFile, files);
            Assert.Contains(IdolImageTo3DPipeline.DecoderFile, files);
            Assert.DoesNotContain(IdolImageTo3DPipeline.TurboSlowFile, files);   // the turbo's experts are inside its file
            Assert.Equal(file, IdolImageTo3DPipeline.Required(MeshModelIds.IsIdolTurbo(id), budget3D: true)[0]);
        }

        Assert.Equal("5", MeshModelProfiles.Find(MeshModelIds.BudgetIdolTurbo)!.Fields.Single(f => f.Name == nameof(MeshRequest.Steps)).Default);
        Assert.Equal(MeshFieldAccess.Rejected, MeshModelProfiles.Find(MeshModelIds.IdolTurbo)!.Fields.Single(f => f.Name == nameof(MeshRequest.TokenBudget)).Access);
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
