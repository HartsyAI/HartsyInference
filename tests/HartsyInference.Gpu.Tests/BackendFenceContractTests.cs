using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Gpu.Tests;

/// <summary>What every GPU backend's fences must do for <c>GenericTransformer</c> to bound how far the host runs ahead.
/// A fence must be real (a 0 handle silently turns the bound off); once a wait on a fence recorded behind a long chain
/// of work returns, none of that work may be left (a full drain right after finds nothing), whether the backend had
/// queued the chain ahead or run it as it was issued; and a released fence must keep working when recorded again.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class BackendFenceContractTests
{
    private const int Size = 2048;
    private const int Chain = 24;
    private const double DrainAfterWaitMs = 2.0;

    private readonly ITestOutputHelper _output;

    public BackendFenceContractTests(ITestOutputHelper output) => _output = output;

    [Theory]
    [MemberData(nameof(BackendGate.GpuKinds), MemberType = typeof(BackendGate))]
    public void AFenceBlocksUntilTheWorkBeforeItIsDone(string kind)
    {
        if (!BackendGate.TryOpen(kind, _output.WriteLine, out IBackend? opened))
        {
            return;
        }
        using IBackend backend = opened!;
        using Tensor a = Filled(0.5f);
        using Tensor b = Filled(0.25f);
        using Tensor c = new(new TensorShape(Size, Size), DType.F32);
        backend.MatMul(c, a, b);
        backend.Sync();

        nint before = backend.RecordFence();
        for (int i = 0; i < Chain; i++)
        {
            backend.MatMul(c, a, b);
        }
        nint after = backend.RecordFence();
        try
        {
            Assert.NotEqual(0, before);
            Assert.NotEqual(0, after);
            Stopwatch clock = Stopwatch.StartNew();
            backend.WaitFence(before);
            double beforeMs = clock.Elapsed.TotalMilliseconds;
            clock.Restart();
            backend.WaitFence(after);
            double afterMs = clock.Elapsed.TotalMilliseconds;
            clock.Restart();
            backend.Sync();
            double drainMs = clock.Elapsed.TotalMilliseconds;
            _output.WriteLine($"[{kind}] wait on the early fence {beforeMs:F2} ms, on the late one {afterMs:F2} ms, "
                + $"sync afterwards {drainMs:F2} ms");
            // The chain is tens of milliseconds of device work, so a wait that returned early leaves a drain that long.
            Assert.True(drainMs < DrainAfterWaitMs,
                $"[{kind}] {drainMs:F2} ms of work before the late fence was still running after waiting on it");
        }
        finally
        {
            backend.ReleaseFence(before);
            backend.ReleaseFence(after);
        }

        nint again = backend.RecordFence();
        backend.WaitFence(again);
        backend.ReleaseFence(again);
        Assert.Equal(0.5f * 0.25f * Size, c.AsSpan<float>()[0]);
    }

    private static Tensor Filled(float value)
    {
        Tensor t = new(new TensorShape(Size, Size), DType.F32);
        t.AsSpan<float>().Fill(value);
        return t;
    }
}
