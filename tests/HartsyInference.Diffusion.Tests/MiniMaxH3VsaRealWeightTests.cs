using System.Security.Cryptography;
using HartsyInference.Cuda;
using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Planning;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Opt-in real-checkpoint reproduction of native VSA (sparse video attention): the Kijai consolidated
/// FastH3 checkpoint (T2VA, 4-step data-free distillation, ComfySol64V1 routing semantics). This is a standalone,
/// full-precision-swap checkpoint — not an adapter on the dense base — so it needs no pruned-base rebase and no
/// stacked VRAM the way PDD/ControlNet did against the dense base. ComfySol64V1 is released in
/// <see cref="VideoService.ApplyH3VsaReleaseGate"/> on the strength of this exact run (see class doc there), so
/// this proves the actual public <see cref="IVideoPlanningService.PlanAsync"/> path — no test-only bypass — and
/// checks the resulting clip's coherence and execution summary (Attention=ComfySol64V1).</summary>
[Collection("CudaSerial")]
[Trait("Category", "Integration")]
[Trait("Category", "RealWeights")]
public sealed class MiniMaxH3VsaRealWeightTests
{
    private const string RunEnvVar = "HARTSY_RUN_H3_VSA_REAL";
    private const int Width = 512;
    private const int Height = 288;
    private const int Frames = 39;
    private const int Fps = 24;
    private const int Steps = 4;

    private const string DitSha256 =
        "7221ae65d78780354d51e5048d29728d9f1f8fb9baf50b1dd3df85f5101413d3";

    private const string Prompt =
        "A joyful golden retriever puppy splashing through a shallow forest stream, cinematic natural light, "
        + "realistic water droplets, steady tracking camera, coherent motion, ambient rushing water and birds";

    private readonly ITestOutputHelper _output;

    public MiniMaxH3VsaRealWeightTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public async Task ComfySol64V1_RealCheckpoint_DetectedAndBlockedThenProducesCoherentClip()
    {
        if (Environment.GetEnvironmentVariable(RunEnvVar) != "1")
        {
            _output.WriteLine($"SKIPPED: set {RunEnvVar}=1 and {RealWeightGate.RequireEnvVar}=1 "
                + "to run the exact-artifact MiniMax-H3 native VSA canary.");
            return;
        }

        Assert.True(Environment.GetEnvironmentVariable(RealWeightGate.RequireEnvVar) == "1",
            $"{RunEnvVar}=1 must be paired with {RealWeightGate.RequireEnvVar}=1 so missing exact assets fail.");
        Assert.True(CudaContext.IsAvailable(), $"{RunEnvVar}=1 requires a CUDA device.");
        if (!RealWeightGate.Require(_output.WriteLine,
                TestPaths.MiniMaxH3.VsaDit,
                TestPaths.MiniMaxH3.TextEncoder,
                TestPaths.MiniMaxH3.VideoVae,
                TestPaths.MiniMaxH3.AudioVae))
        {
            return;
        }

        string ditHash = await Sha256Async(TestPaths.MiniMaxH3.VsaDit);
        Assert.Equal(DitSha256, ditHash);

        int gpu = int.TryParse(Environment.GetEnvironmentVariable("HARTSY_TEST_GPU"), out int requestedGpu)
            ? requestedGpu : 0;
        Assert.InRange(gpu, 0, CudaContext.GetDeviceCount() - 1);
        ModelSpec spec = ModelResolver.Resolve("minimax-h3", TestPaths.MiniMaxH3.VsaDit, Modality.Video);

        ComponentOverrides components = new()
        {
            Qwen = TestPaths.MiniMaxH3.TextEncoder,
            VideoVae = TestPaths.MiniMaxH3.VideoVae,
            AudioVae = TestPaths.MiniMaxH3.AudioVae,
        };
        VideoRequest request = new()
        {
            Prompt = Prompt,
            Width = Width,
            Height = Height,
            Frames = Frames,
            Fps = Fps,
            Steps = Steps,
            CfgScale = 1f,
            FlowShift = 12f,
            AudioFlowShift = 3f,
            Seed = 424242,
            Components = components,
        };

        using InferenceEngine engine = new("cuda", gpu);
        LoggingProgress progress = new(_output);

        _output.WriteLine("[1/2] Resolving the plan through the real public path: this hash must bind "
            + "ComfySol64V1/Vsa, and — released — the plan must be valid with no bypass needed.");
        VideoPlan plan = await engine.VideoPlanning.PlanAsync(spec, request);
        foreach (VideoPlanIssue issue in plan.Issues)
        {
            _output.WriteLine($"plan {issue.Severity}: {issue.Code}: {issue.Message}");
        }
        Assert.Equal("minimax-h3-fast-vsa-comfysol64-v1", plan.Profile.Id);
        Assert.Equal(VideoTaskFamily.T2Va, plan.Profile.Task);
        Assert.Equal(VideoAccelerationKind.Vsa, plan.Profile.Acceleration);
        Assert.Equal(VideoAttentionKind.ComfySol64V1, plan.Profile.Attention);
        Assert.DoesNotContain(plan.Issues, issue => issue.Code == "video.vsa.release_blocked");
        Assert.True(plan.IsValid, string.Join(Environment.NewLine,
            plan.Issues.Select(issue => $"{issue.Code}: {issue.Message}")));
        Assert.Equal(Steps, plan.EffectiveSettings.Steps);

        _output.WriteLine("[2/2] Generating a real 4-step VSA clip.");
        DateTime started = DateTime.UtcNow;
        VideoGenerationResult result = await engine.Video.GenerateAsync(plan, request, progress);
        TimeSpan elapsed = DateTime.UtcNow - started;
        _output.WriteLine($"Generation wall time: {elapsed.TotalSeconds:F1}s for {Steps} transformer evaluations.");

        VideoOutputWriter.Written written = Persist(result, Path.Combine(TestPaths.OutputDir, "h3-vsa-real-weight"),
            "minimax-h3-fastvsa-comfysol64v1");
        _output.WriteLine($"Output: {written.Directory}");

        VideoExecutionSummary execution = Assert.IsType<VideoExecutionSummary>(result.Execution);
        Assert.Equal(VideoAccelerationKind.Vsa, execution.Acceleration);
        Assert.Equal(VideoAttentionKind.ComfySol64V1, execution.Attention);
        Assert.Equal(VideoTaskFamily.T2Va, execution.Task);
        Assert.Equal(Steps, execution.Steps);
        Assert.Equal(1f, execution.CfgScale);
        _output.WriteLine($"Execution path: {execution.ExecutionPath}");

        AssertCoherent(result);
    }

    private static void AssertCoherent(VideoGenerationResult result)
    {
        Assert.Equal(Frames, result.Frames.Count);
        Assert.Equal(Frames, result.Frames.Select(frame => Convert.ToHexString(
            SHA256.HashData(frame.Rgb))).Distinct(StringComparer.Ordinal).Count());
        foreach (VideoFrame frame in result.Frames)
        {
            Assert.Equal(Width, frame.Width);
            Assert.Equal(Height, frame.Height);
            Assert.True(frame.Rgb.Count(value => value != 0) > frame.Rgb.Length / 10,
                $"frame {frame.Index} is effectively black.");
            Assert.True(frame.Rgb.Count(value => value != 255) > frame.Rgb.Length / 10,
                $"frame {frame.Index} is effectively white.");
        }
        AudioBuffer audio = Assert.IsType<AudioBuffer>(result.Audio);
        Assert.Equal(2, audio.ChannelCount);
        Assert.Equal(32_000, audio.SampleRate);
        Assert.All(audio.Channels, channel => Assert.All(channel, sample => Assert.True(float.IsFinite(sample))));
    }

    private static VideoOutputWriter.Written Persist(VideoGenerationResult result, string outputRoot, string slug)
    {
        VideoFrame first = result.Frames[0];
        return VideoOutputWriter.Write(
            result.Frames.Select(frame => frame.Rgb).ToArray(),
            first.Width,
            first.Height,
            outputRoot,
            slug,
            result.Audio,
            result.Fps ?? Fps);
    }

    private static async Task<string> Sha256Async(string path)
    {
        await using FileStream stream = new(path, FileMode.Open, FileAccess.Read, FileShare.Read,
            4 * 1024 * 1024, FileOptions.Asynchronous | FileOptions.SequentialScan);
        using SHA256 sha = SHA256.Create();
        return Convert.ToHexString(await sha.ComputeHashAsync(stream)).ToLowerInvariant();
    }

    private sealed class LoggingProgress(ITestOutputHelper output) : IProgress<StepPreview>
    {
        public void Report(StepPreview value) =>
            output.WriteLine($"denoise {value.Step}/{value.TotalSteps}");
    }
}
