using HartsyInference.World.Camera;
using Xunit;

namespace HartsyInference.World.Tests.Camera;

/// <summary><see cref="LiveCameraPath{TControls}"/>: scripted frames play first and in order, held controls steer the rest,
/// input that arrives during an extension steers only later extensions, and a live path fed a script reproduces the batch
/// <see cref="SanaWmAction.ToCameraToWorld"/> rollout exactly.</summary>
public sealed class LiveCameraPathTests
{
    [Fact]
    public void ScriptedFramesPlayFirst_ThenTheHeldControls()
    {
        LiveCameraPath<int> path = new(new RecordingIntegrator(), idle: 0);
        Assert.Equal(1, path.PoseCount);
        path.Enqueue(7, 2);
        path.Enqueue(8, 1);
        path.Hold(5);
        Assert.Equal(3, path.QueuedFrames);

        IReadOnlyList<float[]> poses = path.ExtendTo(6);
        Assert.Equal([-1, 7, 7, 8, 5, 5], poses.Select(p => (int)p[0]));
        Assert.Equal(0, path.QueuedFrames);
        Assert.Equal(6, path.PoseCount);

        // Asking for fewer poses than the path holds returns the prefix without integrating anything.
        Assert.Equal([-1, 7, 7], path.ExtendTo(3).Select(p => (int)p[0]));
        Assert.Equal(6, path.PoseCount);
    }

    [Fact]
    public void ClearQueued_DropsTheRestOfTheScript()
    {
        LiveCameraPath<int> path = new(new RecordingIntegrator(), idle: 0);
        path.Enqueue(3, 4);
        path.ExtendTo(3);
        Assert.Equal(2, path.QueuedFrames);
        path.ClearQueued();
        path.Hold(9);
        Assert.Equal([-1, 3, 3, 9, 9], path.ExtendTo(5).Select(p => (int)p[0]));
        Assert.Throws<ArgumentOutOfRangeException>(() => path.Enqueue(1, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => path.ExtendTo(0));
    }

    [Fact]
    public async Task HoldDuringExtension_SteersOnlyLaterExtensions()
    {
        LiveCameraPath<int> path = new(new RecordingIntegrator(), idle: 0);
        using CancellationTokenSource stop = new();
        Task input = Task.Run(() =>
        {
            int value = 1;
            while (!stop.IsCancellationRequested)
                path.Hold(value = 3 - value);
        });

        const int block = 16, blocks = 200;
        for (int b = 1; b <= blocks; b++)
            path.ExtendTo(1 + b * block);
        stop.Cancel();
        await input.WaitAsync(TimeSpan.FromSeconds(10));

        // Every frame one extension adds was integrated under the single value held when it began.
        IReadOnlyList<float[]> poses = path.ExtendTo(1 + blocks * block);
        for (int b = 0; b < blocks; b++)
        {
            int first = (int)poses[1 + b * block][0];
            for (int f = 1 + b * block; f < 1 + (b + 1) * block; f++)
                Assert.Equal(first, (int)poses[f][0]);
        }

        Assert.Equal(Enumerable.Range(0, 1 + blocks * block), poses.Select(p => (int)p[1]));
    }

    [Theory]
    [InlineData("w-24")]
    [InlineData("w-24,dw-10,none-5,a-8,ik-3")]
    public void SanaWmLivePath_FedAScript_EqualsTheBatchRollout(string script)
    {
        float[][] batch = SanaWmAction.ToCameraToWorld(script);
        LiveCameraPath<SanaWmControls> path = new(new SanaWmCameraIntegrator(), SanaWmControls.None);
        foreach (SanaWmControls controls in SanaWmAction.Parse(script))
            path.Enqueue(controls, 1);
        IReadOnlyList<float[]> live = path.ExtendTo(batch.Length);

        Assert.Equal(batch.Length, live.Count);
        for (int f = 0; f < batch.Length; f++)
            Assert.Equal(batch[f], live[f]);
        Assert.Equal(0, path.QueuedFrames);
    }

    /// <summary>Pose <c>[control, index]</c>: the controls that produced it and its position on the path; the start is
    /// <c>[-1, 0]</c>.</summary>
    private sealed class RecordingIntegrator : ICameraIntegrator<int>
    {
        private int _frames;

        public float[] Pose => [-1, 0];

        public float[] Step(int controls) => [controls, ++_frames];
    }
}
