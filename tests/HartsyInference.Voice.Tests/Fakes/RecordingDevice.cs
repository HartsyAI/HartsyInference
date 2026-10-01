using System.Collections.Concurrent;
using System.Reflection;
using System.Runtime.ExceptionServices;
using HartsyInference.Core.Backends;

namespace HartsyInference.Voice.Tests.Fakes;

/// <summary>An audio device for tests: an <see cref="IBackend"/> that forwards every call to a real backend but records
/// the memory calls the voice GPU thread makes (<c>FreeActivations</c> with its <c>trimPool</c> argument, and
/// <c>TrimMemoryPool</c>) instead of passing them on, so a test sees what a GPU device would have been asked to do.</summary>
public class RecordingDevice : DispatchProxy
{
    public const string FreeKeepingPool = "FreeActivations(trimPool: false)";
    public const string FreeTrimmingPool = "FreeActivations(trimPool: true)";
    public const string Trim = "TrimMemoryPool()";

    private readonly ConcurrentQueue<string> _calls = new();
    private IBackend _inner = null!;

    /// <summary>The device to hand to the code under test.</summary>
    public IBackend Backend => (IBackend)(object)this;

    /// <summary>Every memory call so far, in order.</summary>
    public IReadOnlyList<string> Calls => [.. _calls];

    public static RecordingDevice Wrap(IBackend inner)
    {
        IBackend proxy = Create<IBackend, RecordingDevice>();
        RecordingDevice device = (RecordingDevice)(object)proxy;
        device._inner = inner;
        return device;
    }

    protected override object? Invoke(MethodInfo? targetMethod, object?[]? args)
    {
        ArgumentNullException.ThrowIfNull(targetMethod);
        switch (targetMethod.Name)
        {
            case nameof(IBackend.FreeActivations):
                // The parameterless form means a pool trim on the GPU backends.
                _calls.Enqueue(args is { Length: 1 } && args[0] is false ? FreeKeepingPool : FreeTrimmingPool);
                return null;
            case nameof(IBackend.TrimMemoryPool):
                _calls.Enqueue(Trim);
                return null;
        }
        try
        {
            return targetMethod.Invoke(_inner, args);
        }
        catch (TargetInvocationException e) when (e.InnerException is not null)
        {
            ExceptionDispatchInfo.Throw(e.InnerException);
            throw;
        }
    }
}
