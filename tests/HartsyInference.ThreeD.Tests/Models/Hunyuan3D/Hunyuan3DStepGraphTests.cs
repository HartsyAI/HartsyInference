using System.Reflection;
using HartsyInference.Core.Backends;
using HartsyInference.ThreeD.Models.Hunyuan3D;
using Xunit;

namespace HartsyInference.ThreeD.Tests.Models.Hunyuan3D;

public sealed class Hunyuan3DStepGraphTests
{
    /// <summary>An <see cref="IBackend"/> that records which calls were made and keeps the step-graph owner slot; every other member does nothing.</summary>
    public class RecordingBackend : DispatchProxy
    {
        public List<string> Calls { get; } = [];

        public object? Owner { get; set; }

        protected override object? Invoke(MethodInfo? targetMethod, object?[]? args)
        {
            string name = targetMethod!.Name;
            Calls.Add(name);
            if (name == "get_StepGraphOwner")
                return Owner;
            if (name == "set_StepGraphOwner")
            {
                Owner = args![0];
                return null;
            }
            Type returnType = targetMethod.ReturnType;
            return returnType == typeof(void) || !returnType.IsValueType ? null : Activator.CreateInstance(returnType);
        }
    }

    private static (IBackend Backend, RecordingBackend Recorder) Backend()
    {
        IBackend backend = DispatchProxy.Create<IBackend, RecordingBackend>();
        return (backend, (RecordingBackend)backend);
    }

    private static Hunyuan3DDit Dit() => new(Hunyuan3DConfig.Hunyuan3D2 with { DepthDouble = 0, DepthSingle = 0 });

    [Fact]
    public void InvalidatingDropsTheGraphTheDitOwns()
    {
        (IBackend backend, RecordingBackend recorder) = Backend();
        Hunyuan3DDit dit = Dit();
        recorder.Owner = dit;

        dit.InvalidateStepGraph(backend);

        Assert.Contains("StepGraphReset", recorder.Calls);
        Assert.Null(recorder.Owner);
    }

    [Fact]
    public void AGraphAnotherOwnerCapturedIsLeftAlone()
    {
        (IBackend backend, RecordingBackend recorder) = Backend();
        object other = new();
        recorder.Owner = other;

        Dit().InvalidateStepGraph(backend);

        Assert.DoesNotContain("StepGraphReset", recorder.Calls);
        Assert.Same(other, recorder.Owner);
    }
}
