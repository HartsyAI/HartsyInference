using HartsyInference.Core.Backends;
using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.World.Camera;
using HartsyInference.World.Pipelines;
using HartsyInference.World.Pipelines.SanaWm;
using Xunit;

namespace HartsyInference.World.Tests.Pipelines;

public sealed class SanaWmPipelineTests
{
    private const int Frames = 9;
    private const int LatentFrames = 2;
    private const int Plane = 22 * 40;

    [Fact]
    public void Success_OpensEachStageInOrder_AndReleasesItBeforeTheNext()
    {
        SanaWmFakeComponents components = new();

        SanaWmClip clip = new SanaWmPipeline(components).Generate(Request(steps: 2), CancellationToken.None);

        Assert.Equal(
            ["open:text", "release:text", "open:image", "release:image", "open:dit", "release:dit", "open:decoder", "release:decoder"],
            components.Events);
        Assert.Equal(Frames, clip.Frames.Count);
        Assert.Equal(SanaWmPipeline.Width * SanaWmPipeline.Height * 3, clip.Frames[0].Length);
        Assert.Equal(16, clip.Fps);
        Assert.Equal(LatentFrames, clip.LatentFrames);
    }

    [Fact]
    public void InjectedInitialLatent_SkipsTheImageEncoderStage()
    {
        SanaWmFakeComponents components = new();
        using Tensor z0 = Latent(0.5f);

        SanaWmClip clip = new SanaWmPipeline(components).Generate(Request(steps: 1) with { InitialLatent = z0 }, CancellationToken.None);

        Assert.Equal(["open:text", "release:text", "open:dit", "release:dit", "open:decoder", "release:decoder"], components.Events);
        Assert.True(clip.InitialLatentInjected);
    }

    [Fact]
    public void DenoiserFailure_ReleasesItsStage_AndOpensNoLaterStage()
    {
        SanaWmFakeComponents components = new()
        {
            BeforePredict = call =>
            {
                if (call == 1)
                    throw new InvalidOperationException("denoiser failed");
            },
        };

        InvalidOperationException ex = Assert.Throws<InvalidOperationException>(() =>
            new SanaWmPipeline(components).Generate(Request(steps: 4), CancellationToken.None));

        Assert.Equal("denoiser failed", ex.Message);
        Assert.Equal(["open:text", "release:text", "open:image", "release:image", "open:dit", "release:dit"], components.Events);
    }

    [Fact]
    public void CancelDuringSampling_ReleasesTheDenoiserStage_AndStopsBeforeTheNextStep()
    {
        using CancellationTokenSource cancel = new();
        SanaWmFakeComponents components = new()
        {
            BeforePredict = call =>
            {
                if (call == 1)
                    cancel.Cancel();
            },
        };

        Assert.ThrowsAny<OperationCanceledException>(() =>
            new SanaWmPipeline(components).Generate(Request(steps: 5), cancel.Token));

        Assert.Equal(2, components.Calls.Count);
        Assert.Equal(["open:text", "release:text", "open:image", "release:image", "open:dit", "release:dit"], components.Events);
    }

    [Fact]
    public void CancelBeforeTheRun_OpensNothing()
    {
        SanaWmFakeComponents components = new();

        Assert.ThrowsAny<OperationCanceledException>(() =>
            new SanaWmPipeline(components).Generate(Request(steps: 2), new CancellationToken(canceled: true)));

        Assert.Empty(components.Events);
    }

    [Fact]
    public void BatchedGuidance_RunsOneBatchTwoForwardPerStep_WithFrameZeroAtTimestepZero()
    {
        const int Steps = 3;
        SanaWmFakeComponents components = new();

        new SanaWmPipeline(components).Generate(Request(steps: Steps), CancellationToken.None);

        float[] timesteps = SanaWmSchedule.Build(Steps, SanaWmPipeline.DefaultFlowShift).Timesteps;
        Assert.Equal(Steps, components.Calls.Count);
        for (int step = 0; step < Steps; step++)
        {
            SanaWmFakeComponents.DenoiserCall call = components.Calls[step];
            Assert.Equal([SanaWmGuidanceBranch.Unconditional, SanaWmGuidanceBranch.Conditional], call.Branches);
            float t = timesteps[step];
            Assert.Equal([0f, t, 0f, t], call.Timesteps);
        }
    }

    [Fact]
    public void SequentialGuidance_RunsTwoSingleBranchForwardsPerStep_AndMatchesBatchedLatents()
    {
        const int Steps = 2;
        SanaWmFakeComponents batched = new();
        SanaWmFakeComponents sequential = new();
        using Tensor z0 = Latent(0.25f);
        float[] batchedFinal = FinalLatent(batched, Request(steps: Steps) with { InitialLatent = z0 });
        float[] sequentialFinal = FinalLatent(sequential, Request(steps: Steps) with { InitialLatent = z0, SequentialCfg = true });

        Assert.Equal(2 * Steps, sequential.Calls.Count);
        for (int i = 0; i < sequential.Calls.Count; i++)
        {
            SanaWmGuidanceBranch expected = i % 2 == 0 ? SanaWmGuidanceBranch.Unconditional : SanaWmGuidanceBranch.Conditional;
            Assert.Equal([expected], sequential.Calls[i].Branches);
            Assert.Equal(LatentFrames, sequential.Calls[i].Timesteps.Length);
        }
        Assert.Equal(batchedFinal, sequentialFinal);
    }

    [Fact]
    public void SequentialGuidance_IsRefusedWhenTheDenoiserNeedsBothBranches()
    {
        SanaWmFakeComponents components = new() { SingleBranch = false };

        Assert.Throws<NotSupportedException>(() =>
            new SanaWmPipeline(components).Generate(Request(steps: 1) with { SequentialCfg = true }, CancellationToken.None));
        Assert.Equal("release:dit", components.Events[^1]);
    }

    [Fact]
    public void OneStep_AppliesTheGuidedVelocityToEveryFrameButFrameZero()
    {
        SanaWmFakeComponents components = new();
        using Tensor z0 = Latent(0.5f);
        float[] final = FinalLatent(components, Request(steps: 1) with { InitialLatent = z0, Guidance = 5.0 });

        // u + 5(c − u) with u = 1 and c = 3; one step from sigma 1 to the terminal 0.
        float t = SanaWmSchedule.Build(1, SanaWmPipeline.DefaultFlowShift).Timesteps[0];
        float guided = SanaWmFakeComponents.UnconditionalVelocity
            + 5f * (SanaWmFakeComponents.ConditionalVelocity - SanaWmFakeComponents.UnconditionalVelocity);
        float expected = Bf16(0.5f + t / 1000f * -guided);
        for (int c = 0; c < SanaWmPipeline.LatentChannels; c++)
        {
            Assert.Equal(0.5f, final[(c * LatentFrames) * Plane]);
            Assert.Equal(expected, final[(c * LatentFrames + 1) * Plane + 17]);
        }
    }

    [Fact]
    public void GuidanceAtOrBelowOne_RunsOnlyTheConditionalBranch()
    {
        SanaWmFakeComponents components = new();
        using Tensor z0 = Latent(0.5f);
        float[] final = FinalLatent(components, Request(steps: 1) with { InitialLatent = z0, Guidance = 1.0 });

        Assert.Equal([SanaWmGuidanceBranch.Conditional], Assert.Single(components.Calls).Branches);
        float t = SanaWmSchedule.Build(1, SanaWmPipeline.DefaultFlowShift).Timesteps[0];
        Assert.Equal(Bf16(0.5f + t / 1000f * -SanaWmFakeComponents.ConditionalVelocity), final[Plane]);
    }

    [Fact]
    public void FrameZero_IsReassertedEveryStep_FromTheEncodedSeedFrame()
    {
        SanaWmFakeComponents components = new();
        float[] final = FinalLatent(components, Request(steps: 4));

        float first = Bf16(SanaWmFakeComponents.FirstLatentValue);
        for (int c = 0; c < SanaWmPipeline.LatentChannels; c++)
        {
            for (int p = 0; p < Plane; p++)
                Assert.Equal(first, final[c * LatentFrames * Plane + p]);
        }
        Assert.Equal(4, components.Calls.Count);
    }

    [Fact]
    public void CombineGuidance_RoundsTheBranchesAndTheResultToBf16()
    {
        // c = 258.5 rounds to 258, so u + 1.5(c − u) = 259, which BF16 rounds to the even neighbour 260.
        float[] unconditional = [256f, -2.5f];
        float[] conditional = [258.5f, 0.5f];

        SanaWmPipeline.CombineGuidance(unconditional, conditional, 1.5);

        Assert.Equal(260f, unconditional[0]);
        Assert.Equal(-2.5f + 1.5f * 3f, unconditional[1]);
    }

    [Theory]
    [InlineData(-1.0f, 0)]
    [InlineData(1.0f, 255)]
    [InlineData(0.0f, 127)]
    [InlineData(2.0f, 255)]
    [InlineData(-0.99609375f, 0)]
    [InlineData(0.58f, 202)]
    public void ToRgb24_TruncatesTheBf16ScaledValue(float value, byte expected)
    {
        float[] rgb = [value, value, value];

        byte[] pixels = SanaWmPipeline.ToRgb24(rgb, 1, 0, 1, 1);

        Assert.Equal([expected, expected, expected], pixels);
    }

    [Fact]
    public void Request_WithAnOffGridFrameCountOrShortCameraPath_IsRefusedBeforeAnyStage()
    {
        SanaWmFakeComponents components = new();
        SanaWmPipeline pipeline = new(components);

        Assert.Throws<ArgumentException>(() => pipeline.Generate(Request(steps: 1) with { Frames = 10 }, CancellationToken.None));
        Assert.Throws<ArgumentException>(() =>
            pipeline.Generate(Request(steps: 1) with { CameraToWorld = Identity(5) }, CancellationToken.None));
        Assert.Empty(components.Events);
    }

    private static float[] FinalLatent(SanaWmFakeComponents components, SanaWmClipRequest request)
    {
        float[]? final = null;
        new SanaWmPipeline(components).Generate(request with
        {
            Tap = (name, tensor) =>
            {
                if (name == "final_latent")
                    final = tensor.AsReadOnlySpan<float>().ToArray();
            },
        }, CancellationToken.None);
        return final ?? throw new InvalidOperationException("final_latent was not tapped.");
    }

    private static SanaWmClipRequest Request(int steps) => new()
    {
        ImageRgb = new byte[SanaWmPipeline.Width * SanaWmPipeline.Height * 3],
        ImageWidth = SanaWmPipeline.Width,
        ImageHeight = SanaWmPipeline.Height,
        Prompt = "a road",
        CameraToWorld = Identity(Frames),
        Intrinsics = [new PinholeIntrinsics(640f, 640f, 640f, 352f)],
        Frames = Frames,
        Steps = steps,
        Seed = 7,
    };

    private static float[][] Identity(int count)
    {
        float[][] poses = new float[count][];
        for (int i = 0; i < count; i++)
            poses[i] = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
        return poses;
    }

    private static Tensor Latent(float value)
    {
        Tensor latent = new(new TensorShape([1, SanaWmPipeline.LatentChannels, LatentFrames, 22, 40]), DType.F32, DeviceKind.Cpu);
        latent.AsSpan<float>().Fill(value);
        return latent;
    }

    private static float Bf16(float value) => BFloat16Bits.ToFloat(BFloat16Bits.FromFloat(value));
}
