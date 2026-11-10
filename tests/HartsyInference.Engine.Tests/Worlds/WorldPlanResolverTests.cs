using System.Text.Json;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.Engine.Worlds;

namespace HartsyInference.Engine.Tests.Worlds;

public sealed class WorldPlanResolverTests
{
    private const int ExpectedOasisFps = 24;
    private const int ExpectedOasisFrames = 16;
    private const int ExpectedOasisSteps = 10;

    private static CatalogEntry Fixture => WorldTestFixtures.FixtureEntry(WorldTestFixtures.SanaShaped());

    private static WorldRequest SanaRequest(int frames, int poses, WorldIntrinsics? intrinsics = null, long seed = 7) => new WorldRequest
    {
        InitImage = WorldTestFixtures.Image(),
        Frames = frames,
        Fps = 16,
        Seed = seed,
        Camera = new WorldCameraPath
        {
            Poses = WorldTestFixtures.IdentityPoses(poses),
            Intrinsics = intrinsics ?? WorldTestFixtures.InBoundsIntrinsics(),
        },
    };

    [Theory]
    [InlineData(9)]
    [InlineData(17)]
    [InlineData(161)]
    public void FrameGrid_AcceptsOnGridCountsAtAndInsideTheRange(int frames)
    {
        WorldPlan plan = WorldPlanResolver.Resolve(WorldTestFixtures.Spec(Fixture), SanaRequest(frames, frames), WorldTestFixtures.CudaSelector);

        Assert.Equal(frames, plan.Frames);
        Assert.Empty(plan.Notes);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(8)]
    [InlineData(10)]
    [InlineData(160)]
    [InlineData(162)]
    public void FrameGrid_RejectsOffGridAndOutOfRangeCounts(int frames)
    {
        WorldPlanningException ex = Assert.Throws<WorldPlanningException>(() =>
            WorldPlanResolver.Resolve(WorldTestFixtures.Spec(Fixture), SanaRequest(frames, frames), WorldTestFixtures.CudaSelector));

        Assert.Contains(frames.ToString(), ex.Message);
    }

    [Fact]
    public void FrameGrid_OffGridRejectionNamesTheNearestValidCounts()
    {
        WorldPlanningException ex = Assert.Throws<WorldPlanningException>(() =>
            WorldPlanResolver.Resolve(WorldTestFixtures.Spec(Fixture), SanaRequest(160, 160), WorldTestFixtures.CudaSelector));

        Assert.Contains("Nearest valid counts: 153, 161", ex.Message);
    }

    [Fact]
    public void FrameGrid_AboveMaximumIsRefusedBeforeSnapping()
    {
        // A snapping grid would clamp 170 to the cap; the resolver must refuse it instead.
        CatalogEntry snapping = WorldTestFixtures.FixtureEntry(WorldTestFixtures.SanaShaped(FrameNormalization.SnapNearest));

        WorldPlanningException ex = Assert.Throws<WorldPlanningException>(() =>
            WorldPlanResolver.Resolve(WorldTestFixtures.Spec(snapping), SanaRequest(170, 170), WorldTestFixtures.CudaSelector));

        Assert.Contains("at most 161 frames", ex.Message);
    }

    [Fact]
    public void FrameGrid_SnappedCountIsRecordedAsANote()
    {
        CatalogEntry snapping = WorldTestFixtures.FixtureEntry(WorldTestFixtures.SanaShaped(FrameNormalization.SnapNearest));

        WorldPlan plan = WorldPlanResolver.Resolve(WorldTestFixtures.Spec(snapping), SanaRequest(18, 17), WorldTestFixtures.CudaSelector);

        Assert.Equal(17, plan.Frames);
        Assert.Equal(18, plan.RequestedFrames);
        Assert.Contains(plan.Notes, n => n.Contains("normalized to 17", StringComparison.Ordinal));
    }

    [Fact]
    public void Camera_ShorterThanTheFrameCountIsAnError()
    {
        WorldPlanningException ex = Assert.Throws<WorldPlanningException>(() =>
            WorldPlanResolver.Resolve(WorldTestFixtures.Spec(Fixture), SanaRequest(17, 10), WorldTestFixtures.CudaSelector));

        Assert.Contains("10 poses but 17 frames", ex.Message);
    }

    [Fact]
    public void Camera_LongerThanTheFrameCountIsTruncatedAndReported()
    {
        WorldPlan plan = WorldPlanResolver.Resolve(WorldTestFixtures.Spec(Fixture), SanaRequest(17, 20), WorldTestFixtures.CudaSelector);

        Assert.Equal(17, plan.CameraPoses);
        Assert.Equal(20, plan.CameraTruncatedFrom);
        Assert.Equal(17, plan.Request.Camera!.Poses.Count);
        Assert.Contains(plan.Notes, n => n.Contains("truncated from 20 to 17", StringComparison.Ordinal));
    }

    [Fact]
    public void Intrinsics_AreRequiredForTypedCameraInput()
    {
        WorldRequest request = new WorldRequest
        {
            InitImage = WorldTestFixtures.Image(),
            Frames = 9,
            Fps = 16,
            Camera = new WorldCameraPath { Poses = WorldTestFixtures.IdentityPoses(9) },
        };

        WorldPlanningException ex = Assert.Throws<WorldPlanningException>(() =>
            WorldPlanResolver.Resolve(WorldTestFixtures.Spec(Fixture), request, WorldTestFixtures.CudaSelector));

        Assert.Contains("needs intrinsics", ex.Message);
    }

    [Theory]
    [InlineData(100.0, 320.0)]
    [InlineData(2000.0, 320.0)]
    [InlineData(320.0, 100.0)]
    public void Intrinsics_FieldOfViewOutsideTheBoundsIsAnError(double fx, double fy)
    {
        WorldIntrinsics intrinsics = new WorldIntrinsics { Fx = fx, Fy = fy, Cx = 320, Cy = 180 };

        WorldPlanningException ex = Assert.Throws<WorldPlanningException>(() =>
            WorldPlanResolver.Resolve(WorldTestFixtures.Spec(Fixture), SanaRequest(9, 9, intrinsics), WorldTestFixtures.CudaSelector));

        Assert.Contains("field of view must be 25-120 degrees", ex.Message);
    }

    [Fact]
    public void Intrinsics_PrincipalPointOutsideTheSeedImageIsAnError()
    {
        WorldIntrinsics intrinsics = new WorldIntrinsics { Fx = 320, Fy = 320, Cx = 700, Cy = 180 };

        WorldPlanningException ex = Assert.Throws<WorldPlanningException>(() =>
            WorldPlanResolver.Resolve(WorldTestFixtures.Spec(Fixture), SanaRequest(9, 9, intrinsics), WorldTestFixtures.CudaSelector));

        Assert.Contains("outside the 640x360 seed image", ex.Message);
    }

    [Fact]
    public void Fps_OutsideTheDeclaredSetIsRefused()
    {
        WorldRequest request = SanaRequest(9, 9) with { Fps = 24 };

        WorldPlanningException ex = Assert.Throws<WorldPlanningException>(() =>
            WorldPlanResolver.Resolve(WorldTestFixtures.Spec(Fixture), request, WorldTestFixtures.CudaSelector));

        Assert.Contains("runs at 16 fps; 24 requested", ex.Message);
    }

    [Fact]
    public void Fps_DefaultsFromTheModelWhenTheRequestNamesNone()
    {
        WorldPlan plan = WorldPlanResolver.Resolve(
            WorldTestFixtures.Spec(Fixture), SanaRequest(9, 9) with { Fps = null }, WorldTestFixtures.CudaSelector);

        Assert.Equal(16, plan.Fps);
    }

    [Fact]
    public void Backend_OutsideTheDeclaredSetIsRefusedBeforeAnyLoad()
    {
        int loads = 0;
        using InferenceEngine engine = new InferenceEngine("cpu");
        WorldService service = new WorldService(engine, _ => loads++);

        WorldPlanningException ex = Assert.Throws<WorldPlanningException>(() =>
            service.Open(WorldTestFixtures.Spec(Fixture), SanaRequest(9, 9)));

        Assert.Contains("does not run on the cpu backend", ex.Message);
        Assert.Equal(0, loads);
    }

    [Fact]
    public void OasisCameraInput_IsRefusedBeforeAnyLoad()
    {
        int loads = 0;
        using InferenceEngine engine = new InferenceEngine("cpu");
        WorldService service = new WorldService(engine, _ => loads++);
        ModelSpec oasis = WorldTestFixtures.Spec(ModelCatalog.Find("oasis"), "oasis");
        WorldRequest request = new WorldRequest
        {
            InitImage = WorldTestFixtures.Image(640, 360),
            Camera = new WorldCameraPath { Poses = WorldTestFixtures.IdentityPoses(16), Intrinsics = WorldTestFixtures.InBoundsIntrinsics() },
        };

        WorldPlanningException ex = Assert.Throws<WorldPlanningException>(() => service.Open(oasis, request));

        Assert.Contains("takes no camera path", ex.Message);
        Assert.Equal(0, loads);
    }

    [Fact]
    public void PlanningPasses_AndTheRequestReachesTheLoadHook()
    {
        int loads = 0;
        using InferenceEngine engine = new InferenceEngine("cpu");
        WorldService service = new WorldService(engine, _ =>
        {
            loads++;
            throw new InvalidOperationException("load reached");
        });

        // Oasis takes any backend, so planning passes on cpu and the seam stops the load before any checkpoint is read.
        ModelSpec oasis = WorldTestFixtures.Spec(ModelCatalog.Find("oasis"), "oasis");
        InvalidOperationException ex = Assert.Throws<InvalidOperationException>(() =>
            service.Open(oasis, new WorldRequest { InitImage = WorldTestFixtures.Image(640, 360) }));
        Assert.Equal("load reached", ex.Message);
        Assert.Equal(1, loads);
    }

    [Fact]
    public void Precision_OutsideTheDeclaredSetIsRefusedBeforeAnyLoad()
    {
        int loads = 0;
        using InferenceEngine engine = new InferenceEngine("cpu");
        WorldService service = new WorldService(engine, _ => loads++);
        // Any backend, so the refusal comes from the precision check rather than the backend check.
        CatalogEntry anyBackend = WorldTestFixtures.FixtureEntry(WorldTestFixtures.SanaShaped() with { Backends = null });
        WorldRequest request = SanaRequest(9, 9) with { Precision = "fp8" };

        WorldPlanningException ex = Assert.Throws<WorldPlanningException>(() =>
            service.Open(WorldTestFixtures.Spec(anyBackend), request));

        Assert.Contains("allowed: bf16", ex.Message);
        Assert.Equal(0, loads);
    }

    [Fact]
    public void Oasis_RefusesEachOptionItDoesNotDeclare()
    {
        ModelSpec oasis = WorldTestFixtures.Spec(ModelCatalog.Find("oasis"), "oasis");
        WorldRequest baseRequest = new WorldRequest { InitImage = WorldTestFixtures.Image(640, 360) };

        WorldCameraPath camera = new WorldCameraPath { Poses = WorldTestFixtures.IdentityPoses(16), Intrinsics = WorldTestFixtures.InBoundsIntrinsics() };
        Assert.Contains("takes no camera path", Assert.Throws<WorldPlanningException>(() =>
            WorldPlanResolver.Resolve(oasis, baseRequest with { Camera = camera }, "cpu")).Message);
        Assert.Contains("Unsupported progressive inference", Assert.Throws<WorldPlanningException>(() =>
            WorldPlanResolver.Resolve(oasis, baseRequest with { ProgressivePreview = true }, "cpu")).Message);
        Assert.Contains("no refiner stage", Assert.Throws<WorldPlanningException>(() =>
            WorldPlanResolver.Resolve(oasis, baseRequest with { Refiner = true }, "cpu")).Message);
        Assert.Contains("no guidance option", Assert.Throws<WorldPlanningException>(() =>
            WorldPlanResolver.Resolve(oasis, baseRequest with { Guidance = 4.0 }, "cpu")).Message);
        Assert.Contains("takes no sampler option", Assert.Throws<WorldPlanningException>(() =>
            WorldPlanResolver.Resolve(oasis, baseRequest with { Sampler = "euler" }, "cpu")).Message);
        Assert.Contains("allowed: native", Assert.Throws<WorldPlanningException>(() =>
            WorldPlanResolver.Resolve(oasis, baseRequest with { Precision = "bf16" }, "cpu")).Message);
        Assert.Contains("at least 2 frames", Assert.Throws<WorldPlanningException>(() =>
            WorldPlanResolver.Resolve(oasis, baseRequest with { Frames = 1 }, "cpu")).Message);
    }

    [Fact]
    public void Seed_EffectiveSeedIsRecordedAndDrawnOnlyWhenTheRequestNamesNone()
    {
        ModelSpec oasis = WorldTestFixtures.Spec(ModelCatalog.Find("oasis"), "oasis");
        WorldRequest drawn = new WorldRequest { InitImage = WorldTestFixtures.Image(640, 360) };

        WorldPlan random = WorldPlanResolver.Resolve(oasis, drawn, "cpu", () => 42);
        WorldPlan named = WorldPlanResolver.Resolve(oasis, drawn with { Seed = 7 }, "cpu", () => 42);

        Assert.True(random.SeedDrawn);
        Assert.Equal(42, random.Seed);
        Assert.Equal(42, random.Request.Seed);
        Assert.False(named.SeedDrawn);
        Assert.Equal(7, named.Seed);
    }

    [Fact]
    public void Oasis_ResolvedDefaultsAreUnchanged()
    {
        ModelSpec oasis = WorldTestFixtures.Spec(ModelCatalog.Find("oasis"), "oasis");
        WorldPlan plan = WorldPlanResolver.Resolve(oasis, new WorldRequest { InitImage = WorldTestFixtures.Image(640, 360), Seed = 3 }, "cpu");

        // fps 24 is the artifact writer's historical default; 16 frames and 10 steps are the CLI's and Oasis's defaults.
        Assert.Equal(ExpectedOasisFps, plan.Fps);
        Assert.Equal(ExpectedOasisFrames, plan.Frames);
        Assert.Equal(ExpectedOasisSteps, plan.Steps);
        Assert.Equal(
            "model=oasis\nbackend=cpu\nprecision=native\nrequestedFrames=16\nframes=16\nfps=24\nsteps=10\nguidance=-\nflowShift=-\n"
            + "sampler=-\nrefiner=False\nprogressivePreview=False\nseed=3\nseedDrawn=False\ncameraPoses=0\ncameraTruncatedFrom=-\nintrinsics=-",
            plan.Snapshot());
    }

    [Fact]
    public void Diamond_ResolvedDefaultsAreUnchanged()
    {
        ModelSpec diamond = WorldTestFixtures.Spec(ModelCatalog.Find("diamond"), "diamond");
        WorldPlan plan = WorldPlanResolver.Resolve(diamond, new WorldRequest { InitImage = WorldTestFixtures.Image(64, 64), Seed = 3 }, "cpu");

        // DIAMOND fixes its own step count, so the resolved steps are zero and no note is raised for the default.
        Assert.Equal(24, plan.Fps);
        Assert.Equal(16, plan.Frames);
        Assert.Equal(0, plan.Steps);
        Assert.Empty(plan.Notes);
        Assert.Equal(
            "model=diamond\nbackend=cpu\nprecision=native\nrequestedFrames=16\nframes=16\nfps=24\nsteps=0\nguidance=-\nflowShift=-\n"
            + "sampler=-\nrefiner=False\nprogressivePreview=False\nseed=3\nseedDrawn=False\ncameraPoses=0\ncameraTruncatedFrom=-\nintrinsics=-",
            plan.Snapshot());
    }

    [Fact]
    public void FrameTimestamps_DeriveFromFpsAndIndex()
    {
        Assert.Equal(0, WorldPlan.FrameTimestampMs(0, 16));
        Assert.Equal(500, WorldPlan.FrameTimestampMs(8, 16));
        Assert.Equal(1000, WorldPlan.FrameTimestampMs(16, 16));
        Assert.Equal(42, WorldPlan.FrameTimestampMs(1, 24));
    }

    [Fact]
    public void CliAndHttpRequests_ResolveToIdenticalSnapshots()
    {
        // The CLI path: flags mapped by WorldRequestOptions, with the camera read from a file.
        string cameraPath = Path.Combine(Path.GetTempPath(), $"world-camera-{Guid.NewGuid():N}.json");
        File.WriteAllText(cameraPath, WorldTestFixtures.CameraJson(20));
        try
        {
            WorldRequestOptions options = new WorldRequestOptions
            {
                Frames = 17,
                Fps = 16,
                Seed = 5,
                CameraPath = cameraPath,
                Intrinsics = "320,320,320,180",
            };
            WorldRequest cliRequest = options.ToRequest(WorldTestFixtures.Image());

            // The HTTP path: the same fields as the JSON body the route deserializes into WorldRequest.
            string body = "{\"frames\": 17, \"fps\": 16, \"seed\": 5, \"camera\": " + WorldTestFixtures.CameraJson(20) + "}";
            WorldRequest parsed = JsonSerializer.Deserialize<WorldRequest>(body, new JsonSerializerOptions(JsonSerializerDefaults.Web))!;
            WorldRequest httpRequest = parsed with { InitImage = WorldTestFixtures.Image() };

            WorldPlan cli = WorldPlanResolver.Resolve(WorldTestFixtures.Spec(Fixture), cliRequest, WorldTestFixtures.CudaSelector);
            WorldPlan http = WorldPlanResolver.Resolve(WorldTestFixtures.Spec(Fixture), httpRequest, WorldTestFixtures.CudaSelector);

            Assert.Equal(cli.Snapshot(), http.Snapshot());
            Assert.Contains("cameraTruncatedFrom=20", cli.Snapshot());
            Assert.Contains("intrinsics=320,320,320,180", cli.Snapshot());
        }
        finally
        {
            File.Delete(cameraPath);
        }
    }

    [Fact]
    public void Resolution_IsIdempotentOnAResolvedRequest()
    {
        WorldPlan first = WorldPlanResolver.Resolve(WorldTestFixtures.Spec(Fixture), SanaRequest(17, 20), WorldTestFixtures.CudaSelector);
        WorldPlan second = WorldPlanResolver.Resolve(WorldTestFixtures.Spec(Fixture), first.Request, WorldTestFixtures.CudaSelector);

        Assert.Equal(first.Frames, second.Frames);
        Assert.Equal(first.Fps, second.Fps);
        Assert.Equal(first.Seed, second.Seed);
        Assert.Equal(17, second.Request.Camera!.Poses.Count);
        Assert.Null(second.CameraTruncatedFrom);
    }
}
