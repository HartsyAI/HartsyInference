using System.Reflection;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Pipelines;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The shared <see cref="ResidentWeights"/>: a bulk preload that throws part-way must not leave part of the model on the device.</summary>
[Trait("Category", "SyntheticSmoke")]
public sealed class ResidentWeightsTests
{
    [Fact]
    public void APreloadThatThrowsFreesEveryWeightAndStaysNotResident()
    {
        using Tensor first = new(new TensorShape(4), DType.F32);
        using Tensor second = new(new TensorShape(4), DType.F32);
        Tensor[] weights = [first, second];
        using CpuBackend cpu = new();
        IBackend backend = FailingPreloadBackend.Create(cpu, out FailingPreloadBackend recorder);
        ResidentWeights resident = new(backend, () => weights);

        Assert.ThrowsAny<Exception>(resident.Preload);
        Assert.False(resident.IsResident);
        Assert.Equal(weights, recorder.Freed);
    }

    /// <summary>Fails every preload, forwards everything else to the CPU backend, and records the tensors each free was given.</summary>
    public class FailingPreloadBackend : DispatchProxy
    {
        private IBackend? _inner;

        public List<Tensor> Freed { get; } = [];

        public static IBackend Create(IBackend inner, out FailingPreloadBackend recorder)
        {
            IBackend proxy = DispatchProxy.Create<IBackend, FailingPreloadBackend>();
            recorder = (FailingPreloadBackend)(object)proxy;
            recorder._inner = inner;
            return proxy;
        }

        protected override object? Invoke(MethodInfo? targetMethod, object?[]? args)
        {
            ArgumentNullException.ThrowIfNull(targetMethod);
            switch (targetMethod.Name)
            {
                case "PreloadWeights":
                    throw new InvalidOperationException("the upload ran out of device memory");
                case "FreeWeights":
                    Freed.AddRange(((IEnumerable<Tensor>)args![0]!).ToList());
                    return null;
                default:
                    return targetMethod.Invoke(_inner, args);
            }
        }
    }
}
