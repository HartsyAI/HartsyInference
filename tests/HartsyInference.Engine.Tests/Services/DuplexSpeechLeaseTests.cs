using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Engine.Audio;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;

namespace HartsyInference.Engine.Tests.Services;

public sealed class DuplexSpeechLeaseTests
{
    private sealed class FakeSession : IDuplexSpeechSession
    {
        internal readonly List<string> Calls = [];
        internal bool Disposed;

        public int SampleRate => 24000;

        public int FrameSamples => 4;

        public DuplexFrame Start()
        {
            Calls.Add("start");
            return new DuplexFrame([0f, 0f, 0f, 0f], "", 0, 0.01, false);
        }

        public DuplexFrame Step(ReadOnlySpan<float> userFrame)
        {
            Calls.Add($"step:{userFrame.Length}:{userFrame[0]}");
            return new DuplexFrame([1f, 1f, 1f, 1f], "hi", Calls.Count - 1, 0.01, false);
        }

        public void Dispose() => Disposed = true;
    }

    [Fact]
    public void Calls_ReachTheSessionAndTheLeasePinsTheModelUntilDisposed()
    {
        AudioRuntime runtime = new();
        using CpuBackend backend = new();
        FakeSession inner = new();
        DuplexSpeechLease lease = new(runtime, "haru-test", inner, backend);
        Assert.True(runtime.Tts.IsPinned("haru-test"));
        Assert.Equal(24000, lease.SampleRate);
        Assert.Equal(4, lease.FrameSamples);

        Assert.Equal(0, lease.Start().Index);
        DuplexFrame frame = lease.Step([0.5f, 0f, 0f, 0f]);
        Assert.Equal("hi", frame.Text);
        Assert.Equal(["start", "step:4:0.5"], inner.Calls);

        lease.Dispose();
        Assert.True(inner.Disposed);
        Assert.False(runtime.Tts.IsPinned("haru-test"));
    }

    [Fact]
    public void ADisposedLeaseRefusesFurtherCalls()
    {
        AudioRuntime runtime = new();
        using CpuBackend backend = new();
        DuplexSpeechLease lease = new(runtime, "haru-test-2", new FakeSession(), backend);
        lease.Dispose();
        Assert.Throws<ObjectDisposedException>(() => lease.Start());
        Assert.Throws<ObjectDisposedException>(() => lease.Step(new float[4]));
        lease.Dispose();
    }
}
