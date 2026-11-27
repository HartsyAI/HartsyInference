using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Worlds;

namespace HartsyInference.Engine.Tests.Worlds;

/// <summary><see cref="WorldPlanResolver"/>'s streaming rules against the catalogued <c>sana-wm-streaming</c> entry (a model that
/// only streams, with unbounded runs and live actions) and the batch <c>sana-wm</c> entry: which mode a request gets, when an
/// unbounded run is allowed, and how a live stream's camera differs from a fixed path.</summary>
public sealed class WorldPlanStreamingTests
{
    private static ModelSpec Streaming => WorldTestFixtures.Spec(
        ModelCatalog.Find("sana-wm-streaming") ?? throw new InvalidOperationException("sana-wm-streaming is not catalogued."),
        "sana-wm-streaming");

    private static ModelSpec Batch => WorldTestFixtures.Spec(
        ModelCatalog.Find("sana-wm") ?? throw new InvalidOperationException("sana-wm is not catalogued."), "sana-wm");

    private static WorldRequest Request(int? frames = 73, WorldCameraPath? camera = null, bool unbounded = false,
        bool? streaming = null) => new()
        {
            InitImage = WorldTestFixtures.Image(),
            Frames = frames,
            Unbounded = unbounded,
            Streaming = streaming,
            Seed = 1,
            Camera = camera ?? new WorldCameraPath { Script = "w-24", Intrinsics = WorldTestFixtures.InBoundsIntrinsics() },
        };

    [Fact]
    public void StreamOnlyModel_StreamsByDefault()
    {
        WorldPlan plan = WorldPlanResolver.Resolve(Streaming, Request(), WorldTestFixtures.CudaSelector);
        Assert.True(plan.Streaming);
        Assert.False(plan.Unbounded);
        Assert.Equal(73, plan.Frames);
        Assert.Equal("w-24", plan.Request.Camera!.Script);
    }

    [Fact]
    public void StreamOnlyModel_RefusesABatchRun()
    {
        WorldPlanningException error = Assert.Throws<WorldPlanningException>(
            () => WorldPlanResolver.Resolve(Streaming, Request(streaming: false), WorldTestFixtures.CudaSelector));
        Assert.Contains("only generates as a stream", error.Message);
    }

    [Fact]
    public void BatchModel_RefusesStreamingAndUnboundedRuns()
    {
        WorldRequest pose = Request(frames: 81, camera: new WorldCameraPath
        {
            Poses = WorldTestFixtures.IdentityPoses(81),
            Intrinsics = WorldTestFixtures.InBoundsIntrinsics(),
        });
        Assert.Contains("batch model", Assert.Throws<WorldPlanningException>(
            () => WorldPlanResolver.Resolve(Batch, pose with { Streaming = true }, WorldTestFixtures.CudaSelector)).Message);
        Assert.Contains("needs a frame count", Assert.Throws<WorldPlanningException>(
            () => WorldPlanResolver.Resolve(Batch, pose with { Frames = null, Unbounded = true }, WorldTestFixtures.CudaSelector)).Message);
    }

    [Fact]
    public void Unbounded_LiveStream_HasNoFrameCount()
    {
        WorldPlan plan = WorldPlanResolver.Resolve(Streaming, Request(frames: null, unbounded: true), WorldTestFixtures.CudaSelector);
        Assert.True(plan.Streaming);
        Assert.True(plan.Unbounded);
        Assert.Equal(0, plan.Frames);
        Assert.Null(plan.Request.Frames);
        Assert.Equal(0, plan.CameraPoses);
        Assert.Equal("w-24", plan.Request.Camera!.Script);
    }

    [Fact]
    public void Unbounded_RefusesAFrameCountOrAFixedPath()
    {
        Assert.Contains("not both", Assert.Throws<WorldPlanningException>(
            () => WorldPlanResolver.Resolve(Streaming, Request(frames: 73, unbounded: true), WorldTestFixtures.CudaSelector)).Message);
        WorldRequest path = Request(frames: null, unbounded: true, camera: new WorldCameraPath
        {
            Poses = WorldTestFixtures.IdentityPoses(73),
            Intrinsics = WorldTestFixtures.InBoundsIntrinsics(),
        });
        Assert.Contains("no fixed camera path", Assert.Throws<WorldPlanningException>(
            () => WorldPlanResolver.Resolve(Streaming, path, WorldTestFixtures.CudaSelector)).Message);
    }

    [Fact]
    public void LiveStream_RefusesPerFrameIntrinsicsAndABadScript()
    {
        WorldRequest perFrame = Request(camera: new WorldCameraPath
        {
            Script = "w-24",
            FrameIntrinsics = [WorldTestFixtures.InBoundsIntrinsics()],
        });
        Assert.Contains("one set of intrinsics", Assert.Throws<WorldPlanningException>(
            () => WorldPlanResolver.Resolve(Streaming, perFrame, WorldTestFixtures.CudaSelector)).Message);
        WorldRequest badScript = Request(camera: new WorldCameraPath { Script = "zz-3", Intrinsics = WorldTestFixtures.InBoundsIntrinsics() });
        Assert.Contains("action script", Assert.Throws<WorldPlanningException>(
            () => WorldPlanResolver.Resolve(Streaming, badScript, WorldTestFixtures.CudaSelector)).Message);
    }

    [Fact]
    public void FixedPathStream_ChecksThePathAgainstTheFrames()
    {
        WorldRequest longPath = Request(camera: new WorldCameraPath
        {
            Poses = WorldTestFixtures.IdentityPoses(97),
            Intrinsics = WorldTestFixtures.InBoundsIntrinsics(),
        });
        WorldPlan plan = WorldPlanResolver.Resolve(Streaming, longPath, WorldTestFixtures.CudaSelector);
        Assert.True(plan.Streaming);
        Assert.Equal(73, plan.CameraPoses);
        Assert.Equal(97, plan.CameraTruncatedFrom);

        WorldRequest shortPath = longPath with { Camera = longPath.Camera! with { Poses = WorldTestFixtures.IdentityPoses(49) } };
        Assert.Throws<WorldPlanningException>(() => WorldPlanResolver.Resolve(Streaming, shortPath, WorldTestFixtures.CudaSelector));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ResolvedRequest_ResolvesToTheSamePlan(bool unbounded)
    {
        WorldRequest request = unbounded ? Request(frames: null, unbounded: true) : Request();
        WorldPlan plan = WorldPlanResolver.Resolve(Streaming, request, WorldTestFixtures.CudaSelector);
        WorldPlan again = WorldPlanResolver.Resolve(Streaming, plan.Request, WorldTestFixtures.CudaSelector);
        Assert.Equal(plan.Snapshot(), again.Snapshot());
    }

    [Theory]
    [InlineData(25)]
    [InlineData(49)]
    [InlineData(241)]
    public void StreamFrameGrid_TakesWholeChunks(int frames)
    {
        WorldPlan plan = WorldPlanResolver.Resolve(Streaming, Request(frames), WorldTestFixtures.CudaSelector);
        Assert.Equal(frames, plan.Frames);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(24)]
    [InlineData(81)]
    public void StreamFrameGrid_RefusesPartialChunks(int frames) =>
        Assert.Throws<WorldPlanningException>(() => WorldPlanResolver.Resolve(Streaming, Request(frames), WorldTestFixtures.CudaSelector));
}
