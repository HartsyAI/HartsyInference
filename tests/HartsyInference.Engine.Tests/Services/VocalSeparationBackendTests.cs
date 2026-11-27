using System.Reflection;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Exceptions;
using HartsyInference.Engine.Services;
using Xunit;

namespace HartsyInference.Engine.Tests.Services;

/// <summary>Which backend <c>fx separate</c> runs the vocal separator on, and the fall back to the CPU when the card is full. No GPU and no weights.</summary>
public sealed class VocalSeparationBackendTests
{
    [Theory]
    [InlineData(DeviceType.Cuda)]
    [InlineData(DeviceType.Vulkan)]
    public async Task AGpuEngineRunsOnItsOwnBackend(DeviceType type)
    {
        using CpuBackend cpu = new();
        IBackend gpu = FakeDevice.Create(new DeviceKind(type, 0));
        List<IBackend> ran = [];

        await FxService.WithCpuFallbackAsync(gpu, cpu, backend => { ran.Add(backend); return Task.FromResult(0); }, CancellationToken.None);

        Assert.Same(gpu, Assert.Single(ran));
    }

    [Fact]
    public async Task ACpuEngineRunsOnTheCpuBackend()
    {
        using CpuBackend engine = new();
        using CpuBackend cpu = new();
        List<IBackend> ran = [];

        await FxService.WithCpuFallbackAsync(engine, cpu, backend => { ran.Add(backend); return Task.FromResult(0); }, CancellationToken.None);

        Assert.Same(cpu, Assert.Single(ran));
    }

    [Theory]
    [InlineData("plain")]
    [InlineData("wrapped")]
    [InlineData("aggregate")]
    public async Task AFullCardFallsBackToTheCpu(string shape)
    {
        using CpuBackend cpu = new();
        IBackend gpu = FakeDevice.Create(DeviceKind.Cuda(0));
        OutOfVramException full = new(1 << 30, 1 << 20);
        Exception thrown = shape switch
        {
            "wrapped" => new InvalidOperationException("kernel launch failed", full),
            "aggregate" => new AggregateException(full),
            _ => full,
        };

        IBackend result = await FxService.WithCpuFallbackAsync(gpu, cpu,
            backend => ReferenceEquals(backend, gpu) ? throw thrown : Task.FromResult(backend), CancellationToken.None);

        Assert.Same(cpu, result);
    }

    [Fact]
    public async Task AnotherFailureIsNotRetriedOnTheCpu()
    {
        using CpuBackend cpu = new();
        IBackend gpu = FakeDevice.Create(DeviceKind.Cuda(0));
        int runs = 0;

        await Assert.ThrowsAsync<InvalidDataException>(() => FxService.WithCpuFallbackAsync<int>(gpu, cpu,
            _ => { runs++; throw new InvalidDataException("bad weights"); }, CancellationToken.None));

        Assert.Equal(1, runs);
    }

    [Fact]
    public async Task ACancelledRequestDoesNotStartTheCpuRun()
    {
        using CpuBackend cpu = new();
        IBackend gpu = FakeDevice.Create(DeviceKind.Cuda(0));
        using CancellationTokenSource cancel = new();
        int runs = 0;

        await Assert.ThrowsAsync<OutOfVramException>(() => FxService.WithCpuFallbackAsync<int>(gpu, cpu,
            _ => { runs++; cancel.Cancel(); throw new OutOfVramException(1, 0); }, cancel.Token));

        Assert.Equal(1, runs);
    }

    /// <summary>A backend that only answers which device it is.</summary>
    public class FakeDevice : DispatchProxy
    {
        private DeviceKind _device;

        public static IBackend Create(DeviceKind device)
        {
            IBackend proxy = DispatchProxy.Create<IBackend, FakeDevice>();
            ((FakeDevice)(object)proxy)._device = device;
            return proxy;
        }

        protected override object? Invoke(MethodInfo? targetMethod, object?[]? args) =>
            targetMethod?.Name == "get_" + nameof(IBackend.Device) ? _device : throw new NotSupportedException(targetMethod?.Name);
    }
}
