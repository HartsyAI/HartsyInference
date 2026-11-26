using System.Reflection;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Memory.Vram;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Pipelines;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The denoiser residency the FLUX.2-style pipelines share: what a preload and a release send to the backend under each VRAM profile.</summary>
public sealed class ResidentWeightsTests
{
    [Fact]
    public void AReleaseFreesThePreloadedWeights_WhenTheProfileDoesNotKeepModels()
    {
        using CpuBackend cpu = new();
        using Tensor weight = new(new TensorShape(4), DType.F32);
        IBackend backend = Recorder.Create(cpu, out Recorder recorder);
        VramPolicyRegistry.Set(backend, new VramPolicy { Tier = VramTier.Balanced, KeepResident = LeverState.Off });
        try
        {
            ResidentWeights resident = new(backend, () => [weight]);
            resident.Preload();
            Assert.True(resident.IsResident);

            resident.Release();

            Assert.False(resident.IsResident);
            Assert.Equal(new[] { "PreloadWeights", "FreeWeights" }, recorder.Calls.Select(c => c.Kind));
            Assert.All(recorder.Calls, c => Assert.Equal(new[] { weight }, c.Tensors));
        }
        finally
        {
            VramPolicyRegistry.Clear(backend);
        }
    }

    [Fact]
    public void AKeptModelStaysResident_UntilAForcedRelease()
    {
        using CpuBackend cpu = new();
        using Tensor weight = new(new TensorShape(4), DType.F32);
        IBackend backend = Recorder.Create(cpu, out Recorder recorder);
        VramPolicyRegistry.Set(backend, new VramPolicy { Tier = VramTier.Balanced, KeepResident = LeverState.On });
        try
        {
            ResidentWeights resident = new(backend, () => [weight]);
            resident.Preload();

            resident.Release();
            Assert.True(resident.IsResident);
            Assert.Equal(new[] { "PreloadWeights" }, recorder.Calls.Select(c => c.Kind));

            resident.Release(force: true);
            Assert.False(resident.IsResident);
            Assert.Equal(new[] { "PreloadWeights", "FreeWeights" }, recorder.Calls.Select(c => c.Kind));
        }
        finally
        {
            VramPolicyRegistry.Clear(backend);
        }
    }

    [Fact]
    public void NothingIsFreed_BeforeAPreload_OrAfterOneThatThrew()
    {
        using CpuBackend cpu = new();
        using Tensor weight = new(new TensorShape(4), DType.F32);
        IBackend backend = Recorder.Create(cpu, out Recorder recorder);
        VramPolicyRegistry.Set(backend, new VramPolicy { Tier = VramTier.Balanced, KeepResident = LeverState.Off });
        try
        {
            ResidentWeights resident = new(backend, () => [weight]);
            resident.Release(force: true);
            Assert.Empty(recorder.Calls);

            recorder.FailPreload = true;
            Assert.Throws<InvalidOperationException>(resident.Preload);
            Assert.False(resident.IsResident);
            resident.Release(force: true);
            Assert.Equal(new[] { "PreloadWeights" }, recorder.Calls.Select(c => c.Kind));
        }
        finally
        {
            VramPolicyRegistry.Clear(backend);
        }
    }

    /// <summary>Forwards every call to the CPU backend and records the tensors each preload and free was given; a preload can be made to throw.</summary>
    public class Recorder : DispatchProxy
    {
        private IBackend? _inner;

        public List<(string Kind, List<Tensor> Tensors)> Calls { get; } = [];

        public bool FailPreload { get; set; }

        public static IBackend Create(IBackend inner, out Recorder recorder)
        {
            IBackend proxy = DispatchProxy.Create<IBackend, Recorder>();
            recorder = (Recorder)(object)proxy;
            recorder._inner = inner;
            return proxy;
        }

        protected override object? Invoke(MethodInfo? targetMethod, object?[]? args)
        {
            ArgumentNullException.ThrowIfNull(targetMethod);
            if (targetMethod.Name is "PreloadWeights" or "FreeWeights")
            {
                List<Tensor> tensors = ((IEnumerable<Tensor>)args![0]!).ToList();
                Calls.Add((targetMethod.Name, tensors));
                if (FailPreload && targetMethod.Name == "PreloadWeights")
                    throw new InvalidOperationException("preload failed");
                return targetMethod.Invoke(_inner, [tensors, .. args.Skip(1)]);
            }
            return targetMethod.Invoke(_inner, args);
        }
    }
}
