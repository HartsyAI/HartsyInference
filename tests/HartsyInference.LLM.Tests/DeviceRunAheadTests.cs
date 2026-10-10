using System.Reflection;
using HartsyInference.Core.Backends;
using HartsyInference.LLM.Transformer;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary><see cref="DeviceRunAhead"/>'s fence protocol against a backend that only logs fence calls: before each
/// step it waits on the fence from two steps back, never the step just issued; every fence is released exactly once,
/// waited on or not; a wait that fails still releases its fence; and a disabled bound or a backend without fences makes
/// no waits at all.</summary>
public sealed class DeviceRunAheadTests
{
    [Fact]
    public void EachStepWaitsOnTheFenceFromTwoStepsBack_AndTheLastTwoAreReleasedUnwaited()
    {
        IBackend backend = DispatchProxy.Create<IBackend, FenceLog>();
        FenceLog log = (FenceLog)(object)backend;

        DeviceRunAhead runAhead = new(backend, stackalloc nint[2], enabled: true);
        try
        {
            for (int step = 0; step < 5; step++)
            {
                runAhead.BeforeStep();
                log.Calls.Add($"issue {step}");
                runAhead.AfterStep();
            }
        }
        finally
        {
            runAhead.Dispose();
        }

        Assert.Equal(
        [
            "issue 0", "record 1",
            "issue 1", "record 2",
            "wait 1", "release 1", "issue 2", "record 3",
            "wait 2", "release 2", "issue 3", "record 4",
            "wait 3", "release 3", "issue 4", "record 5",
            "release 5", "release 4",
        ], log.Calls);
    }

    [Fact]
    public void AFailedWait_StillReleasesItsFence()
    {
        IBackend backend = DispatchProxy.Create<IBackend, FenceLog>();
        FenceLog log = (FenceLog)(object)backend;
        log.FailWaits = true;

        DeviceRunAhead runAhead = new(backend, stackalloc nint[1], enabled: true);
        runAhead.AfterStep();
        InvalidOperationException? failure = null;
        try
        {
            runAhead.BeforeStep();
        }
        catch (InvalidOperationException ex)
        {
            failure = ex;
        }
        runAhead.Dispose();

        Assert.NotNull(failure);
        Assert.Equal(["record 1", "wait 1", "release 1"], log.Calls);
    }

    [Theory]
    [InlineData(false, true)]
    [InlineData(true, false)]
    public void ADisabledBoundOrABackendWithoutFences_NeverWaits(bool enabled, bool backendHasFences)
    {
        IBackend backend = DispatchProxy.Create<IBackend, FenceLog>();
        FenceLog log = (FenceLog)(object)backend;
        log.NoFences = !backendHasFences;

        DeviceRunAhead runAhead = new(backend, stackalloc nint[2], enabled);
        for (int step = 0; step < 4; step++)
        {
            runAhead.BeforeStep();
            runAhead.AfterStep();
        }
        runAhead.Dispose();

        Assert.DoesNotContain(log.Calls, call => call.StartsWith("wait", StringComparison.Ordinal));
        Assert.DoesNotContain(log.Calls, call => call.StartsWith("release", StringComparison.Ordinal));
        if (!enabled)
        {
            Assert.Empty(log.Calls);
        }
    }

    /// <summary>An <see cref="IBackend"/> whose fences are numbered from 1 and whose fence calls are logged in order;
    /// any other member is unreachable from <see cref="DeviceRunAhead"/> and throws.</summary>
    public class FenceLog : DispatchProxy
    {
        private nint _next;

        public List<string> Calls { get; } = [];

        public bool FailWaits { get; set; }

        public bool NoFences { get; set; }

        protected override object? Invoke(MethodInfo? targetMethod, object?[]? args)
        {
            switch (targetMethod!.Name)
            {
                case nameof(IBackend.RecordFence):
                    if (NoFences)
                    {
                        return (nint)0;
                    }
                    _next++;
                    Calls.Add($"record {_next}");
                    return _next;
                case nameof(IBackend.WaitFence):
                    Calls.Add($"wait {(nint)args![0]!}");
                    if (FailWaits)
                    {
                        throw new InvalidOperationException("the device was lost");
                    }
                    return null;
                case nameof(IBackend.ReleaseFence):
                    Calls.Add($"release {(nint)args![0]!}");
                    return null;
                default:
                    throw new NotSupportedException($"{targetMethod.Name} is not reachable from DeviceRunAhead.");
            }
        }
    }
}
