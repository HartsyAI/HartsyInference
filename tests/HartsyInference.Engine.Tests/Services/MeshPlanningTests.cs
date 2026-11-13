using System.Reflection;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Planning;
using HartsyInference.Engine.Planning.Memory;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;

namespace HartsyInference.Engine.Tests.Services;

public sealed class MeshPlanningTests
{
    private static readonly string[] NonTunableProperties =
        [nameof(MeshRequest.Prompt), nameof(MeshRequest.Image), nameof(MeshRequest.SceneMesh), nameof(MeshRequest.ScenePrompt), nameof(MeshRequest.Vram), nameof(MeshRequest.Settings)];

    public static TheoryData<string> Ids()
    {
        TheoryData<string> data = [];
        foreach (string id in MeshModelIds.Described)
            data.Add(id);
        return data;
    }

    private static MeshModelProfile Profile(string id) => MeshModelProfiles.Find(id) ?? throw new InvalidOperationException($"No profile for {id}.");

    private static ImageData Image() => new()
    {
        Rgb = new byte[4 * 4 * 3],
        Width = 4,
        Height = 4,
        Alpha = [.. Enumerable.Range(0, 16).Select(i => i < 8 ? (byte)0 : (byte)255)],
    };

    private static SceneMeshData Scene() => new() { Data = System.Text.Encoding.ASCII.GetBytes("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"), Format = "obj" };

    private static MeshControl Box() => new() { Kind = MeshControlKind.BoundingBox, Values = [0.5f, 0.5f, 0.5f] };

    /// <summary>The smallest request the model accepts, built from what its profile says it takes.</summary>
    private static MeshRequest Valid(string id)
    {
        MeshInputKind input = Profile(id).Input;
        return new MeshRequest
        {
            Image = input is MeshInputKind.SceneMesh ? null : Image(),
            SceneMesh = input is MeshInputKind.SceneMesh or MeshInputKind.SceneMeshAndImage ? Scene() : null,
            Control = input is MeshInputKind.ImageAndControl ? Box() : null,
        };
    }

    /// <summary>A grid the model accepts when it reads one: its first listed value, else its default, else 64.</summary>
    private static int LegalGrid(string id)
    {
        MeshFieldSpec field = Profile(id).Fields.Single(f => f.Name == nameof(MeshRequest.GridResolution));
        string? text = field.Values?[0] ?? field.Default;
        return text is not null && int.TryParse(text, out int value) && value > 0 ? value : 64;
    }

    private static MeshRequest With(MeshRequest request, string field, string id) => field switch
    {
        nameof(MeshRequest.Steps) => request with { Steps = 5 },
        nameof(MeshRequest.GridResolution) => request with { GridResolution = LegalGrid(id) },
        nameof(MeshRequest.FieldOfViewRadians) => request with { FieldOfViewRadians = 0.5 },
        nameof(MeshRequest.Samples) => request with { Samples = 2 },
        nameof(MeshRequest.Colors) => request with { Colors = false },
        nameof(MeshRequest.Chains) => request with { Chains = false },
        nameof(MeshRequest.Rigged) => request with { Rigged = true },
        nameof(MeshRequest.Texture) => request with { Texture = true },
        nameof(MeshRequest.TextureSize) => request with { TextureSize = 512 },
        nameof(MeshRequest.Mesh) => request with { Mesh = true },
        nameof(MeshRequest.Control) => request with { Control = Box() },
        _ => throw new ArgumentOutOfRangeException(nameof(field), field, "No sample value."),
    };

    [Theory]
    [MemberData(nameof(Ids))]
    public void EveryModelDescribesEveryTunableField(string id)
        => Assert.Equal(MeshModelProfile.TunableFields.Order(), Profile(id).Fields.Select(f => f.Name).Order());

    [Fact]
    public void EveryRequestPropertyIsEitherTunableOrAnInputOrASetting()
    {
        string[] properties = [.. typeof(MeshRequest).GetProperties(BindingFlags.Public | BindingFlags.Instance).Select(p => p.Name)];

        Assert.Equal(properties.Order(), MeshModelProfile.TunableFields.Concat(NonTunableProperties).Order());
    }

    [Theory]
    [MemberData(nameof(Ids))]
    public void TheSmallestRequestOfEveryModelPassesTheValidator(string id)
        => MeshRequestValidator.Validate(id, Valid(id), matterAvailable: false);

    [Theory]
    [MemberData(nameof(Ids))]
    public void ARejectedFieldFailsTheValidator(string id)
    {
        foreach (MeshFieldSpec field in Profile(id).Fields.Where(f => f.Access == MeshFieldAccess.Rejected))
        {
            Exception? error = Record.Exception(() => MeshRequestValidator.Validate(id, With(Valid(id), field.Name, id), matterAvailable: false));

            Assert.True(error is NotSupportedException, $"{id} {field.Name} should be rejected, got {error?.GetType().Name ?? "no exception"}");
        }
    }

    [Theory]
    [MemberData(nameof(Ids))]
    public void AnIgnoredOrUsedFieldIsNotRejectedForBeingSet(string id)
    {
        string[] sampled = [nameof(MeshRequest.Steps), nameof(MeshRequest.GridResolution), nameof(MeshRequest.FieldOfViewRadians)];
        foreach (MeshFieldSpec field in Profile(id).Fields.Where(f => f.Access != MeshFieldAccess.Rejected && sampled.Contains(f.Name)))
        {
            Exception? error = Record.Exception(() => MeshRequestValidator.Validate(id, With(Valid(id), field.Name, id), matterAvailable: false));

            Assert.True(error is null, $"{id} {field.Name} is described as accepted but failed: {error?.Message}");
        }
    }

    [Theory]
    [InlineData("trellis2", 512, true)]
    [InlineData("trellis2", 1536, true)]
    [InlineData("trellis2", 768, false)]
    [InlineData("pixal3d", 1024, true)]
    [InlineData("pixal3d", 1536, true)]
    [InlineData("pixal3d", 512, false)]
    [InlineData("cola3d", 16, true)]
    [InlineData("cola3d", 1024, true)]
    [InlineData("cola3d", 1040, false)]
    [InlineData("cola3d", 250, false)]
    public void TheGridValuesAndRangesOfTheProfileAreTheOnesTheValidatorEnforces(string id, int grid, bool legal)
    {
        MeshRequest request = Valid(id) with { GridResolution = grid };
        Exception? error = Record.Exception(() => MeshRequestValidator.Validate(id, request, matterAvailable: false));

        Assert.Equal(legal, error is null);
        MeshFieldSpec field = Profile(id).Fields.Single(f => f.Name == nameof(MeshRequest.GridResolution));
        if (field.Values is not null)
            Assert.Equal(legal, field.Values.Contains(grid.ToString(System.Globalization.CultureInfo.InvariantCulture)));
        else
            Assert.Equal(legal, grid >= field.Min && grid <= field.Max && grid % 16 == 0);
    }

    [Theory]
    [InlineData("trellis2", 200, true)]
    [InlineData("trellis2", 201, false)]
    [InlineData("pixal3d", 201, false)]
    public void TheStepRangeOfTheProfileIsTheOneTheValidatorEnforces(string id, int steps, bool legal)
    {
        Exception? error = Record.Exception(() => MeshRequestValidator.Validate(id, Valid(id) with { Steps = steps }, matterAvailable: false));

        Assert.Equal(legal, error is null);
        Assert.Equal(200, Profile(id).Fields.Single(f => f.Name == nameof(MeshRequest.Steps)).Max);
    }

    [Theory]
    [InlineData(15, false)]
    [InlineData(16, true)]
    [InlineData(8192, true)]
    [InlineData(8193, false)]
    public void TheTextureSizeRangeIsEnforcedForTheModelsThatTexture(int size, bool legal)
    {
        MeshRequest request = Valid("trellis2") with { Texture = true, TextureSize = size };
        Exception? error = Record.Exception(() => MeshRequestValidator.Validate("trellis2", request, matterAvailable: false));

        Assert.Equal(legal, error is null);
        MeshFieldSpec field = Profile("trellis2").Fields.Single(f => f.Name == nameof(MeshRequest.TextureSize));
        Assert.Equal(16, field.Min);
        Assert.Equal(8192, field.Max);
    }

    [Theory]
    [InlineData(0.01, true)]
    [InlineData(1.5707, true)]
    [InlineData(0.0, false)]
    [InlineData(1.5707963267948966, false)]
    public void TheFieldOfViewOpenIntervalIsEnforced(double fov, bool legal)
    {
        MeshRequest request = Valid("pixal3d") with { FieldOfViewRadians = fov };
        Exception? error = Record.Exception(() => MeshRequestValidator.Validate("pixal3d", request, matterAvailable: false));

        Assert.Equal(legal, error is null);
    }

    [Theory]
    [MemberData(nameof(Ids))]
    public void EveryModelRejectsASeedBeyondTheSamplersRangeAndAcceptsTheLargestOne(string id)
    {
        MeshRequest beyond = Valid(id) with { Seed = (long)int.MaxValue + 1 };
        MeshRequest largest = Valid(id) with { Seed = int.MaxValue };
        MeshRequest wrapsToZero = Valid(id) with { Seed = 1L << 32 };

        foreach (MeshRequest request in new[] { beyond, wrapsToZero })
        {
            ArgumentOutOfRangeException error = Assert.Throws<ArgumentOutOfRangeException>(() => MeshRequestValidator.Validate(id, request, matterAvailable: false));
            Assert.Equal(nameof(MeshRequest.Seed), MeshFieldTag.FieldOf(error));
        }
        Assert.Null(Record.Exception(() => MeshRequestValidator.Validate(id, largest, matterAvailable: false)));
    }

    [Theory]
    [MemberData(nameof(Ids))]
    public void EveryProfileStatesTheSeedMaximum(string id)
    {
        MeshFieldSpec seed = Profile(id).Fields.Single(f => f.Name == nameof(MeshRequest.Seed));

        Assert.Equal((double)int.MaxValue, seed.Max);
        Assert.Contains(seed.Access, new[] { MeshFieldAccess.Used, MeshFieldAccess.Ignored });
    }

    [Theory]
    [InlineData("hunyuan3d")]
    [InlineData("trellis")]
    [InlineData("unirig")]
    [InlineData("triposr")]
    public async Task TheModelsThatOnceWrappedALargeSeedNowReportItAsAPlanError(string id)
    {
        using InferenceEngine engine = new("cpu");

        MeshPlan plan = await engine.MeshPlanning.PlanAsync(Spec(id), Valid(id) with { Seed = (1L << 32) + 5 });

        Assert.Contains(plan.Issues, i => i.Severity == MeshPlanIssueSeverity.Error && i.Field == nameof(MeshRequest.Seed));
    }

    [Fact]
    public void AnUnknownModelFailsAsUnknownBeforeItsSeedIsRangeChecked()
    {
        Assert.Throws<NotSupportedException>(() => MeshRequestValidator.Validate("no-such-model", Valid("hunyuan3d") with { Seed = 1L << 32 }, matterAvailable: false));
    }

    [Fact]
    public async Task AnIgnoredSeedOutOfRangeIsOneErrorNotAnErrorAndAnIgnoredWarning()
    {
        using InferenceEngine engine = new("cpu");

        MeshPlan plan = await engine.MeshPlanning.PlanAsync(Spec("triposr"), Valid("triposr") with { Seed = 1L << 32 });

        Assert.Single(plan.Issues, i => i.Field == nameof(MeshRequest.Seed));
        Assert.Equal(MeshPlanIssueSeverity.Error, plan.Issues.Single(i => i.Field == nameof(MeshRequest.Seed)).Severity);
    }

    [Fact]
    public void TheSamplerCastIsCheckedSoNoCallerCanWrapASeed()
    {
        Assert.Null(MeshSeed.ForSampler(-1));
        Assert.Equal(7, MeshSeed.ForSampler(7));
        Assert.Throws<OverflowException>(() => MeshSeed.ForSampler(1L << 32));
    }

    [Theory]
    [InlineData("hunyuan3d-part", "segments a triangle mesh into parts")]
    [InlineData("unirig", "rigs a triangle mesh")]
    public void ASceneModelWithoutAMeshSaysWhatItDoes(string id, string verb)
    {
        NotSupportedException error = Assert.Throws<NotSupportedException>(() => MeshRequestValidator.Validate(id, new MeshRequest(), matterAvailable: false));

        Assert.Contains(verb, error.Message);
        Assert.Equal(nameof(MeshRequest.SceneMesh), MeshFieldTag.FieldOf(error));
    }

    [Theory]
    [InlineData("trellis2", "TextureSize")]
    [InlineData("pixal3d", "TextureSize")]
    public async Task PlanNamesTheFieldOfAnIssue(string id, string field)
    {
        using InferenceEngine engine = new("cpu");

        MeshPlan plan = await engine.MeshPlanning.PlanAsync(Spec(id), Valid(id) with { TextureSize = 512 });

        Assert.Equal(field, Assert.Single(plan.Issues, i => i.Severity == MeshPlanIssueSeverity.Error).Field);
    }

    [Theory]
    [InlineData("trellis2", nameof(MeshRequest.GridResolution))]
    [InlineData("pixal3d", nameof(MeshRequest.GridResolution))]
    public async Task PlanNamesTheGridFieldOfAPipelineError(string id, string field)
    {
        using InferenceEngine engine = new("cpu");

        MeshPlan plan = await engine.MeshPlanning.PlanAsync(Spec(id), Valid(id) with { GridResolution = 768 });

        Assert.Equal(field, Assert.Single(plan.Issues, i => i.Severity == MeshPlanIssueSeverity.Error).Field);
    }

    [Fact]
    public async Task PlanNamesTheFieldsOfOtherPipelineErrors()
    {
        using InferenceEngine engine = new("cpu");

        MeshPlan fov = await engine.MeshPlanning.PlanAsync(Spec("pixal3d"), Valid("pixal3d") with { FieldOfViewRadians = 2.0 });
        MeshPlan steps = await engine.MeshPlanning.PlanAsync(Spec("trellis2"), Valid("trellis2") with { Steps = 500 });
        MeshPlan rigged = await engine.MeshPlanning.PlanAsync(Spec("hairdeck"), Valid("hairdeck") with { Rigged = true, Chains = false });
        MeshPlan seed = await engine.MeshPlanning.PlanAsync(Spec("hairdeck"), Valid("hairdeck") with { Seed = (long)int.MaxValue + 1 });

        Assert.Equal(nameof(MeshRequest.FieldOfViewRadians), fov.Issues.Single(i => i.Severity == MeshPlanIssueSeverity.Error).Field);
        Assert.Equal(nameof(MeshRequest.Steps), steps.Issues.Single(i => i.Severity == MeshPlanIssueSeverity.Error).Field);
        Assert.Equal(nameof(MeshRequest.Rigged), rigged.Issues.Single(i => i.Severity == MeshPlanIssueSeverity.Error).Field);
        Assert.Equal(nameof(MeshRequest.Seed), seed.Issues.Single(i => i.Severity == MeshPlanIssueSeverity.Error).Field);
    }

    [Theory]
    [InlineData(2, 2, 11, null)]
    [InlineData(2, 2, 12, 3)]
    public void AMalformedImageBufferIsReportedAgainstTheImageField(int width, int height, int rgbLength, int? alphaLength)
    {
        ImageData image = new() { Rgb = new byte[rgbLength], Width = width, Height = height, Alpha = alphaLength is int n ? new byte[n] : null };

        ArgumentException error = Assert.ThrowsAny<ArgumentException>(
            () => MeshRequestValidator.Validate("trellis2", new MeshRequest { Image = image }, matterAvailable: true));

        Assert.Equal(nameof(MeshRequest.Image), MeshFieldTag.FieldOf(error));
    }

    [Fact]
    public async Task PlanNamesTheFieldOfRejectedOptionsOfOtherModels()
    {
        using InferenceEngine engine = new("cpu");

        MeshPlan samples = await engine.MeshPlanning.PlanAsync(Spec("triposr"), Valid("triposr") with { Samples = 2 });
        MeshPlan chains = await engine.MeshPlanning.PlanAsync(Spec("triposr"), Valid("triposr") with { Chains = false });
        MeshPlan seed = await engine.MeshPlanning.PlanAsync(Spec("hunyuan3d-part"), Valid("hunyuan3d-part") with { Seed = (long)int.MaxValue + 1 });

        Assert.Equal(nameof(MeshRequest.Samples), samples.Issues.Single(i => i.Severity == MeshPlanIssueSeverity.Error).Field);
        Assert.Equal(nameof(MeshRequest.Chains), chains.Issues.Single(i => i.Severity == MeshPlanIssueSeverity.Error).Field);
        Assert.Equal(nameof(MeshRequest.Seed), seed.Issues.Single(i => i.Severity == MeshPlanIssueSeverity.Error).Field);
    }

    [Fact]
    public async Task APlanForAModelWithoutARecordedPeakSaysWhyThereIsNoFit()
    {
        using InferenceEngine engine = new("cpu");

        MeshPlan plan = await engine.MeshPlanning.PlanAsync(Spec("triposr"), Valid("triposr"));

        Assert.Equal(MemoryFitVerdict.Unknown, plan.Memory.Verdict);
        Assert.Contains("No device-memory peak", plan.Memory.Reason);
    }

    [Fact]
    public void AMissingRequiredInputIsReportedAgainstItsField()
    {
        NotSupportedException error = Assert.Throws<NotSupportedException>(() => MeshRequestValidator.Validate("triposr", new MeshRequest(), matterAvailable: false));

        Assert.Equal(nameof(MeshRequest.Image), MeshFieldTag.FieldOf(error));
    }

    [Fact]
    public void TRELLIS2WithoutAlphaOrAMatterIsRefusedAndWithTheMatterIsAccepted()
    {
        MeshRequest request = Valid("trellis2") with { Image = Image() with { Alpha = null } };

        HartsyInference.Core.Exceptions.HartsyInferenceException error =
            Assert.Throws<HartsyInference.Core.Exceptions.HartsyInferenceException>(() => MeshRequestValidator.Validate("trellis2", request, matterAvailable: false));
        Assert.Equal(nameof(MeshRequest.Image), MeshFieldTag.FieldOf(error));
        MeshRequestValidator.Validate("trellis2", request, matterAvailable: true);
    }

    [Fact]
    public void AnUnknownModelHasNoProfileAndFailsTheValidator()
    {
        Assert.Null(MeshModelProfiles.Find("hunyuan3d-mv"));
        Assert.Throws<NotSupportedException>(() => MeshRequestValidator.Validate("hunyuan3d-mv", Valid("trellis2"), matterAvailable: false));
    }

    [Fact]
    public void DescribeReturnsOneDescriptorPerDescribedModelWithItsGatedRepositories()
    {
        using InferenceEngine engine = new("cpu");
        IReadOnlyList<MeshModelDescriptor> all = engine.MeshPlanning.DescribeAll();

        Assert.Equal(MeshModelIds.Described.Order(), all.Select(d => d.Id).Order());
        MeshModelDescriptor hairdeck = engine.MeshPlanning.Describe("hairdeck");
        Assert.Contains("facebook/sam3", hairdeck.Licence.GatedRepos);
        Assert.Contains("facebook/dinov3-vitl16-pretrain-lvd1689m", hairdeck.Licence.GatedRepos);
        Assert.Empty(engine.MeshPlanning.Describe("hunyuan3d").Licence.GatedRepos);
        Assert.Equal(MeshOutputKinds.Splats | MeshOutputKinds.Mesh, engine.MeshPlanning.Describe("sam3d-objects").Outputs);
        Assert.Equal(MeshInputKind.ImageWithMask, engine.MeshPlanning.Describe("sam3d-objects").Input);
        Assert.Throws<NotSupportedException>(() => engine.MeshPlanning.Describe("hunyuan3d-mv"));
    }

    [Fact]
    public async Task PlanReportsAnUnsupportedFieldWithItsNameAndIsNotValid()
    {
        using InferenceEngine engine = new("cpu");
        ModelSpec spec = Spec("trellis2");

        MeshPlan plan = await engine.MeshPlanning.PlanAsync(spec, Valid("trellis2") with { FieldOfViewRadians = 0.5 });

        Assert.False(plan.IsValid);
        MeshPlanIssue issue = Assert.Single(plan.Issues, i => i.Code == "mesh.request.unsupported");
        Assert.Equal(nameof(MeshRequest.FieldOfViewRadians), issue.Field);
    }

    [Fact]
    public async Task PlanWarnsAboutAnIgnoredFieldAndStaysValid()
    {
        using InferenceEngine engine = new("cpu");

        MeshPlan plan = await engine.MeshPlanning.PlanAsync(Spec("triposr"), Valid("triposr") with { Steps = 12 });

        Assert.True(plan.IsValid);
        MeshPlanIssue warning = Assert.Single(plan.Issues, i => i.Code == "mesh.field.ignored");
        Assert.Equal(nameof(MeshRequest.Steps), warning.Field);
        Assert.False(plan.Settings.ContainsKey(nameof(MeshRequest.Steps)));
    }

    [Fact]
    public async Task PlanReportsWhereEachSettingCameFrom()
    {
        using InferenceEngine engine = new("cpu");

        MeshPlan plan = await engine.MeshPlanning.PlanAsync(Spec("pixal3d"), Valid("pixal3d") with { GridResolution = 1536 });

        Assert.Equal(new ResolvedParameter("1536", ParameterSource.UserOverride), plan.Settings[nameof(MeshRequest.GridResolution)]);
        Assert.Equal(ParameterSource.RecipeDefault, plan.Settings[nameof(MeshRequest.Steps)].Source);
        Assert.Equal("0.8575560450553894", plan.Settings[nameof(MeshRequest.FieldOfViewRadians)].Value);
    }

    [Fact]
    public async Task PlanForAModelWithoutAnyCheckpointIsNotValid()
    {
        using InferenceEngine engine = new("cpu");

        MeshPlan plan = await engine.MeshPlanning.PlanAsync(Spec("hunyuan3d") with { LocalPath = null }, Valid("hunyuan3d"));

        Assert.False(plan.IsValid);
        Assert.Contains(plan.Issues, i => i.Code == "mesh.model.not_installed");
    }

    [Fact]
    public async Task GenerateRunsTheSameValidationAsPlan()
    {
        using InferenceEngine engine = new("cpu");
        MeshRequest request = Valid("trellis2") with { FieldOfViewRadians = 0.5 };

        await Assert.ThrowsAsync<NotSupportedException>(() => engine.Mesh.GenerateAsync(Spec("trellis2"), request));
    }

    [Fact]
    public async Task ASam3dMeshRequestIsPlannedAsMeshTrueFromTheUserAndKeepsPreviewAndFrameUsed()
    {
        using InferenceEngine engine = new("cpu");
        MeshRequest request = Valid("sam3d-objects") with { Mesh = true };

        MeshPlan plan = await engine.MeshPlanning.PlanAsync(Spec("sam3d-objects"), request);
        MeshPlan splats = await engine.MeshPlanning.PlanAsync(Spec("sam3d-objects"), Valid("sam3d-objects"));

        Assert.Equal(new ResolvedParameter("true", ParameterSource.UserOverride), plan.Settings[nameof(MeshRequest.Mesh)]);
        Assert.Equal(new ResolvedParameter("false", ParameterSource.RecipeDefault), splats.Settings[nameof(MeshRequest.Mesh)]);
        // The finisher decimates a preview and applies the frame to every glb, so a model that can return a mesh describes both as used.
        Assert.All([nameof(MeshRequest.PreviewTriangles), nameof(MeshRequest.Frame)], name => Assert.Equal(MeshFieldAccess.Used, Profile("sam3d-objects").Fields.Single(f => f.Name == name).Access));
        Assert.Equal(MeshFieldAccess.Ignored, Profile("trellis").Fields.Single(f => f.Name == nameof(MeshRequest.Frame)).Access);
    }

    [Fact]
    public void ASam3dMeshRunHasAMeasuredPeakAndASplatRunHasNone()
    {
        MemoryEstimate? mesh = MeshMemoryProfile.Estimate("sam3d-objects", new MeshRequest { Mesh = true });

        Assert.NotNull(mesh);
        Assert.Equal(MemoryEstimateAccuracy.Measured, mesh.Accuracy);
        Assert.Null(MeshMemoryProfile.Estimate("sam3d-objects", new MeshRequest()));
    }

    [Fact]
    public void MeasuredPeaksBecomeAOneStageEstimateAndOthersHaveNone()
    {
        MemoryEstimate? p1024 = MeshMemoryProfile.Estimate("pixal3d", new MeshRequest());
        MemoryEstimate? p1536 = MeshMemoryProfile.Estimate("pixal3d", new MeshRequest { GridResolution = 1536 });

        Assert.NotNull(p1024);
        Assert.Equal(MemoryEstimateAccuracy.Measured, p1024.Accuracy);
        Assert.True(p1536!.PeakResidentBytes > p1024.PeakResidentBytes);
        Assert.Null(MeshMemoryProfile.Estimate("triposr", new MeshRequest()));
        Assert.Null(MeshMemoryProfile.Estimate("hunyuan3d-mv", new MeshRequest()));
    }

    [Fact]
    public void AMeasuredPeakIsHeldAgainstTheWholeCard()
    {
        MemoryEstimate estimate = MeshMemoryProfile.Estimate("pixal3d", new MeshRequest { GridResolution = 1536 })!;

        Assert.Equal(MemoryFitVerdict.Resident, MeshMemoryProfile.Judge(estimate, 12L << 30, HartsyInference.Core.Memory.Vram.VramTier.Auto).Verdict);
        Assert.Equal(MemoryFitVerdict.Infeasible, MeshMemoryProfile.Judge(estimate, 8L << 30, HartsyInference.Core.Memory.Vram.VramTier.Auto).Verdict);
    }

    [Fact]
    public async Task TheGenericMemoryServiceAnswersForMeshModelsWithAPeakAndRefusesTheRest()
    {
        using InferenceEngine engine = new("cpu");

        MemoryEstimate estimate = await engine.MemoryEstimation.EstimateAsync(Spec("pixal3d"), new MemoryEstimateRequest(1024, 1024));

        Assert.Equal("pixal3d", estimate.FamilyId);
        await Assert.ThrowsAsync<NotSupportedException>(() => engine.MemoryEstimation.EstimateAsync(Spec("triposr"), new MemoryEstimateRequest(1024, 1024)));
        MemoryFit fit = await engine.MemoryEstimation.AssessAsync(Spec("triposr"), new MemoryEstimateRequest(1024, 1024));
        Assert.Equal(MemoryFitVerdict.Unknown, fit.Verdict);
    }

    private static ModelSpec Spec(string id) => new()
    {
        Requested = id,
        Modality = Modality.Mesh,
        Catalog = HartsyInference.Engine.Registry.ModelCatalog.Find(id),
        LocalPath = Path.GetTempPath(),
    };
}
