using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.Engine.Worlds;
using HartsyInference.World.Pipelines;
using HartsyInference.World.Pipelines.SanaWm;

namespace HartsyInference.Engine.Tests.Worlds;

/// <summary>The <c>sana-wm</c> catalog contract, its planning-time refusals, the action-script camera, component resolution,
/// the one-shot adapter's request mapping, and the service refusing unknown world ids. No weights are read.</summary>
public sealed class SanaWmWorldTests
{
    private const string Script = "w-80,dw-40,aw-40";

    private static CatalogEntry SanaWm => ModelCatalog.Find("sana-wm") ?? throw new InvalidOperationException("sana-wm is not catalogued.");

    private static ModelSpec Spec => WorldTestFixtures.Spec(SanaWm, "sana-wm");

    private static WorldRequest ScriptRequest(int? frames = null, string script = Script) => new WorldRequest
    {
        Prompt = "a dry lakebed",
        InitImage = WorldTestFixtures.Image(),
        Frames = frames,
        Seed = 42,
        Camera = new WorldCameraPath { Script = script, Intrinsics = WorldTestFixtures.InBoundsIntrinsics() },
    };

    [Fact]
    public void Catalog_DeclaresTheStageOneContract()
    {
        CatalogEntry entry = SanaWm;
        WorldCapabilities caps = entry.World!;

        Assert.Equal(Modality.World, entry.Modality);
        Assert.Equal(ModelStatus.ValidationPending, entry.Status);
        Assert.True(entry.CliDrivable);
        Assert.Equal(["cuda"], caps.Backends!);
        Assert.Equal(["bf16"], caps.Precisions);
        Assert.Equal([16], caps.Fps!);
        Assert.Equal(16, caps.DefaultFps);
        Assert.Equal(161, caps.DefaultFrames);
        Assert.Equal(9, caps.MinFrames);
        Assert.Equal(FrameGrid.StepPlusOne(8, FrameNormalization.Reject, 161), caps.Frames);
        Assert.Equal(60, caps.DefaultSteps);
        Assert.Equal([WorldCameraMode.PerFramePose, WorldCameraMode.ActionScript], caps.CameraModes);
        Assert.NotNull(caps.CameraScript);
        Assert.True(caps.CameraRequired);
        Assert.Equal(WorldIntrinsicsPolicy.Required, caps.Intrinsics);
        Assert.Equal(["flow_euler_ltx"], caps.Samplers);
        Assert.True(caps.SupportsGuidance);
        Assert.True(caps.SupportsFlowShift);
        Assert.False(caps.SupportsRefiner);
        Assert.False(caps.SupportsProgressivePreview);
        Assert.True(caps.SupportsSequentialCfg);
        Assert.Equal("transformer", entry.Assets[0].Role);
    }

    [Fact]
    public void ActionScript_IsExpandedAtPlanning_AndTheResolvedRequestCarriesPoses()
    {
        WorldPlan plan = WorldPlanResolver.Resolve(Spec, ScriptRequest(), WorldTestFixtures.CudaSelector);

        Assert.Equal(161, plan.Frames);
        Assert.Equal(16, plan.Fps);
        Assert.Equal(60, plan.Steps);
        Assert.Equal(161, plan.CameraPoses);
        WorldCameraPath camera = plan.Request.Camera!;
        Assert.Null(camera.Script);
        Assert.Equal(161, camera.Poses.Count);
        Assert.Equal((float)(0.025 / (1 + 1e-6)), camera.Poses[1][11]);
        Assert.Contains(plan.Notes, note => note.Contains("expanded from action script", StringComparison.Ordinal));

        WorldPlan again = WorldPlanResolver.Resolve(Spec, plan.Request, WorldTestFixtures.CudaSelector);
        Assert.Equal(plan.Frames, again.Frames);
        Assert.Equal(161, again.Request.Camera!.Poses.Count);
    }

    [Fact]
    public void ActionScript_LongerThanTheFrames_IsTruncatedAndReported()
    {
        WorldPlan plan = WorldPlanResolver.Resolve(Spec, ScriptRequest(frames: 9), WorldTestFixtures.CudaSelector);

        Assert.Equal(9, plan.CameraPoses);
        Assert.Equal(161, plan.CameraTruncatedFrom);
    }

    [Fact]
    public void ActionScript_ShorterThanTheFrames_IsRefused()
    {
        WorldPlanningException ex = Assert.Throws<WorldPlanningException>(() =>
            WorldPlanResolver.Resolve(Spec, ScriptRequest(frames: 9, script: "w-7"), WorldTestFixtures.CudaSelector));

        Assert.Contains("has 8 poses but 9 frames", ex.Message);
    }

    [Fact]
    public void ActionScript_ThatDoesNotParse_IsRefusedWithTheReason()
    {
        WorldPlanningException ex = Assert.Throws<WorldPlanningException>(() =>
            WorldPlanResolver.Resolve(Spec, ScriptRequest(script: "q-3"), WorldTestFixtures.CudaSelector));

        Assert.Contains("action script", ex.Message);
        Assert.Contains("unknown key", ex.Message);
    }

    [Fact]
    public void ActionScript_ForAModelWithoutTheMode_IsRefused()
    {
        ModelSpec oasis = WorldTestFixtures.Spec(ModelCatalog.Find("oasis"), "oasis");
        WorldRequest request = ScriptRequest() with { InitImage = WorldTestFixtures.Image(640, 360) };

        WorldPlanningException ex = Assert.Throws<WorldPlanningException>(() => WorldPlanResolver.Resolve(oasis, request, "cpu"));

        Assert.Contains("takes no action-script camera path", ex.Message);
    }

    [Fact]
    public void PosesAndScriptTogether_AreRefused()
    {
        WorldRequest request = ScriptRequest() with
        {
            Camera = new WorldCameraPath
            {
                Script = Script,
                Poses = WorldTestFixtures.IdentityPoses(161),
                Intrinsics = WorldTestFixtures.InBoundsIntrinsics(),
            },
        };

        WorldPlanningException ex = Assert.Throws<WorldPlanningException>(() =>
            WorldPlanResolver.Resolve(Spec, request, WorldTestFixtures.CudaSelector));

        Assert.Contains("poses or an action script, not both", ex.Message);
    }

    public static TheoryData<string, string> Refusals => new()
    {
        { "frames=10", "8n+1" },
        { "frames=169", "at most 161 frames" },
        { "frames=1", "at least 9 frames" },
        { "fps=24", "runs at 16 fps" },
        { "sampler=flow_dpm-solver", "allowed: flow_euler_ltx" },
        { "precision=fp8", "allowed: bf16" },
        { "refiner", "no refiner stage" },
        { "progressive", "Unsupported progressive inference" },
        { "no-intrinsics", "needs intrinsics" },
        { "no-camera", "needs a camera path" },
    };

    [Theory]
    [MemberData(nameof(Refusals))]
    public void Planning_RefusesWhatStageOneDoesNotSupport(string change, string message)
    {
        WorldRequest request = ScriptRequest();
        request = change switch
        {
            "frames=10" => request with { Frames = 10 },
            "frames=169" => request with { Frames = 169 },
            "frames=1" => request with { Frames = 1 },
            "fps=24" => request with { Fps = 24 },
            "sampler=flow_dpm-solver" => request with { Sampler = "flow_dpm-solver" },
            "precision=fp8" => request with { Precision = "fp8" },
            "refiner" => request with { Refiner = true },
            "progressive" => request with { ProgressivePreview = true },
            "no-intrinsics" => request with { Camera = new WorldCameraPath { Script = Script } },
            _ => request with { Camera = null },
        };

        WorldPlanningException ex = Assert.Throws<WorldPlanningException>(() =>
            WorldPlanResolver.Resolve(Spec, request, WorldTestFixtures.CudaSelector));

        Assert.Contains(message, ex.Message);
    }

    [Fact]
    public void CpuBackend_IsRefusedBeforeAnyLoad()
    {
        int loads = 0;
        using InferenceEngine engine = new InferenceEngine("cpu");
        WorldService service = new WorldService(engine, _ => loads++);

        WorldPlanningException ex = Assert.Throws<WorldPlanningException>(() => service.Open(Spec, ScriptRequest()));

        Assert.Contains("does not run on the cpu backend", ex.Message);
        Assert.Equal(0, loads);
    }

    [Fact]
    public void PerFrameIntrinsics_AreValidatedAndCarried()
    {
        WorldIntrinsics k = WorldTestFixtures.InBoundsIntrinsics();
        WorldRequest request = ScriptRequest() with
        {
            Camera = new WorldCameraPath { Script = Script, FrameIntrinsics = [k, k with { Fx = 330 }, k] },
        };

        WorldPlan plan = WorldPlanResolver.Resolve(Spec, request, WorldTestFixtures.CudaSelector);

        Assert.Equal(3, plan.FrameIntrinsics);
        Assert.Equal(k, plan.Intrinsics);
        Assert.Contains("frameIntrinsics=3", plan.Snapshot());
        Assert.Equal(3, plan.Request.Camera!.FrameIntrinsics!.Count);

        WorldRequest outOfRange = request with { Camera = request.Camera! with { FrameIntrinsics = [k, k with { Fx = 50 }] } };
        Assert.Throws<WorldPlanningException>(() => WorldPlanResolver.Resolve(Spec, outOfRange, WorldTestFixtures.CudaSelector));
        WorldRequest both = request with { Camera = request.Camera! with { Intrinsics = k } };
        Assert.Throws<WorldPlanningException>(() => WorldPlanResolver.Resolve(Spec, both, WorldTestFixtures.CudaSelector));
    }

    [Fact]
    public void Options_MapScriptPromptGuidanceAndShift_OntoTheRequest()
    {
        WorldRequestOptions options = new WorldRequestOptions
        {
            Prompt = "a road",
            CameraScript = Script,
            Intrinsics = "320,320,320,180",
            Guidance = 4.5,
            FlowShift = 8.0,
            Frames = 81,
        };

        WorldRequest request = options.ToRequest(WorldTestFixtures.Image());
        WorldPlan plan = WorldPlanResolver.Resolve(Spec, request, WorldTestFixtures.CudaSelector);

        Assert.Equal("a road", request.Prompt);
        Assert.Equal(Script, request.Camera!.Script);
        Assert.Equal(4.5, plan.Guidance);
        Assert.Equal(8.0, plan.FlowShift);
        Assert.Equal(81, plan.CameraPoses);
        Assert.Throws<ArgumentException>(() => (options with { CameraPath = "poses.npy" }).ToRequest(WorldTestFixtures.Image()));
    }

    [Fact]
    public void ClipRequest_TakesThePlanAndFillsTheModelDefaults()
    {
        WorldPlan plan = WorldPlanResolver.Resolve(Spec, ScriptRequest(frames: 17), WorldTestFixtures.CudaSelector);

        SanaWmClipRequest clip = SanaWmClipPipeline.ToClipRequest(plan, null);

        Assert.Equal(17, clip.Frames);
        Assert.Equal(17, clip.CameraToWorld.Count);
        Assert.Equal(60, clip.Steps);
        Assert.Equal(SanaWmPipeline.DefaultGuidance, clip.Guidance);
        Assert.Equal(SanaWmPipeline.DefaultFlowShift, clip.FlowShift);
        Assert.Equal(42, clip.Seed);
        Assert.Equal("a dry lakebed", clip.Prompt);
        Assert.Equal(WorldTestFixtures.ImageWidth, clip.ImageWidth);
        Assert.Equal(320f, Assert.Single(clip.Intrinsics).Fx);
        Assert.False(clip.SequentialCfg);
    }

    [Fact]
    public async Task SessionActions_AreRefused_AndTheRunReleasesEverything()
    {
        using InferenceEngine engine = new InferenceEngine("cpu");
        WorldService service = new WorldService(engine);
        WorldPlan plan = WorldPlanResolver.Resolve(Spec, ScriptRequest(frames: 9), WorldTestFixtures.CudaSelector);
        SanaWmClipPipeline pipeline = new SanaWmClipPipeline(() => throw new InvalidOperationException("components must not load"));
        using IWorldSession session = service.OpenOneShot(pipeline, plan);
        session.SendAction("forward");

        InvalidOperationException ex = await Assert.ThrowsAsync<InvalidOperationException>(async () =>
        {
            await foreach (VideoFrame frame in session.StreamAsync(CancellationToken.None))
            {
                Assert.Fail($"No frame should stream; got {frame.Index}.");
            }
        });

        Assert.Contains("session actions are not accepted", ex.Message);
        Assert.Equal(0, service.RunLedger.Outstanding);
    }

    [Fact]
    public void UnknownWorldId_FailsWithAClearError_AndLoadsNothing()
    {
        int loads = 0;
        using InferenceEngine engine = new InferenceEngine("cpu");
        WorldService service = new WorldService(engine, _ => loads++);
        ModelSpec unknown = WorldTestFixtures.Spec(null, "not-a-world-model");
        ModelSpec fixture = WorldTestFixtures.Spec(WorldTestFixtures.FixtureEntry(new WorldCapabilities()));
        WorldRequest request = new WorldRequest { InitImage = WorldTestFixtures.Image(640, 360) };

        NotSupportedException catalogLess = Assert.Throws<NotSupportedException>(() => service.Open(unknown, request));
        NotSupportedException uncatalogued = Assert.Throws<NotSupportedException>(() => service.Open(fixture, request));

        Assert.Contains("'not-a-world-model' is not a world model this engine can load", catalogLess.Message);
        Assert.Contains("'fixture-world' is not a world model this engine can load", uncatalogued.Message);
        Assert.Equal(0, loads);
        Assert.Equal(0, service.RetainedModelCount);
    }

    [Fact]
    public void Components_ResolveFromTheReleaseLayout_AndAuxPathsOverrideIt()
    {
        string root = Path.Combine(Path.GetTempPath(), $"sana-wm-layout-{Guid.NewGuid():N}");
        string dit = Touch(root, "dit/sana_wm_1600m_720p.safetensors");
        string vae = Touch(root, "vae/diffusion_pytorch_model.safetensors");
        string gemma = Touch(root, "text_encoder/gemma-2-2b-it/gemma-2-2b-it.safetensors");
        string tokenizer = Touch(root, "text_encoder/gemma-2-2b-it/tokenizer.model");
        string otherVae = Touch(root, "elsewhere/vae.safetensors");
        try
        {
            ModelSpec spec = Spec with { LocalPath = dit };

            WorldComponents components = WorldComponentResolver.Resolve(spec);

            Assert.Equal(WorldArchitecture.SanaWm, components.Architecture);
            Assert.Equal(dit, components.Backbone);
            Assert.Equal(vae, components.Vae);
            Assert.Equal(gemma, components.TextEncoder);
            Assert.Equal(tokenizer, components.Tokenizer);
            ModelSpec overridden = spec with { Aux = new Dictionary<string, string> { [WorldComponentResolver.VaeAuxKey] = otherVae } };
            Assert.Equal(otherVae, WorldComponentResolver.Resolve(overridden).Vae);

            File.Delete(vae);
            FileNotFoundException missing = Assert.Throws<FileNotFoundException>(() => WorldComponentResolver.Resolve(spec));
            Assert.Contains(WorldComponentResolver.VaeAuxKey, missing.Message);
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    private static string Touch(string root, string relative)
    {
        string path = Path.Combine(root, relative.Replace('/', Path.DirectorySeparatorChar));
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllBytes(path, []);
        return path;
    }
}
