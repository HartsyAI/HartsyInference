using System.Runtime;
using HartsyInference.VoiceHost.Config;
using HartsyInference.VoiceHost.Runtime;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Configuration.EnvironmentVariables;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Hosting.Internal;
using Microsoft.Extensions.Options;
using Xunit;

namespace HartsyInference.VoiceHost.Tests;

/// <summary>Process settings: the pool floor covers every CPU kernel thread plus the headroom, the GC holds sustained low
/// latency exactly while a call is up, and the generic host has the console lifetime (SIGTERM), the bounded shutdown and
/// no configuration source that reads the environment.</summary>
public sealed class HostRuntimeTests
{
    [Fact]
    public void ThePoolFloorCoversTheCpuKernelThreadsPlusHeadroom()
    {
        int floor = HostRuntimeTuning.ApplyThreadPool(4);
        ThreadPool.GetMinThreads(out int workers, out _);

        Assert.True(floor >= Math.Min(4, Environment.ProcessorCount) + HostRuntimeTuning.PoolHeadroom);
        Assert.True(workers >= floor);
    }

    [Fact]
    public void TheGcHoldsSustainedLowLatencyExactlyWhileACallIsUp()
    {
        Assert.Equal(0, HostRuntimeTuning.ActiveCalls);
        GCLatencyMode before = GCSettings.LatencyMode;
        HostRuntimeTuning.CallStarted();
        HostRuntimeTuning.CallStarted();
        Assert.Equal(GCLatencyMode.SustainedLowLatency, GCSettings.LatencyMode);
        HostRuntimeTuning.CallEnded();
        Assert.Equal(GCLatencyMode.SustainedLowLatency, GCSettings.LatencyMode);
        HostRuntimeTuning.CallEnded();
        Assert.Equal(before, GCSettings.LatencyMode);
        Assert.Equal(0, HostRuntimeTuning.ActiveCalls);
    }

    [Fact]
    public void TheGenericHostStopsOnSigtermWithinItsTimeoutAndReadsNoEnvironment()
    {
        VoiceHostSettings settings = VoiceHostConfigLoader.Resolve(new VoiceHostConfig());
        using IHost host = VoiceHostProgram.Build(settings, _ => throw new InvalidOperationException("not started in this test"));

        Assert.IsType<ConsoleLifetime>(host.Services.GetRequiredService<IHostLifetime>());
        Assert.Equal(VoiceHostProgram.ShutdownTimeout, host.Services.GetRequiredService<IOptions<HostOptions>>().Value.ShutdownTimeout);
        IConfigurationRoot configuration = (IConfigurationRoot)host.Services.GetRequiredService<IConfiguration>();
        Assert.DoesNotContain(configuration.Providers, provider => provider is EnvironmentVariablesConfigurationProvider);
        Assert.Same(settings, host.Services.GetRequiredService<VoiceHostSettings>());
    }
}
