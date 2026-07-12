using System.Runtime.InteropServices;
using HartsyInference.Core.Runtime;
using Xunit;

namespace HartsyInference.Core.Tests;

public sealed class HostMemoryInfoTests
{
    [Fact]
    public void AvailableBytes_NeverThrows()
    {
        Exception? ex = Record.Exception(() => HostMemoryInfo.AvailableBytes());
        Assert.Null(ex);
    }

    [Fact]
    public void AvailableBytes_OnLinux_ReturnsPlausibleValue()
    {
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Linux)) return; // covered on non-Linux by the no-throw test
        long? bytes = HostMemoryInfo.AvailableBytes();
        Assert.NotNull(bytes);
        Assert.InRange(bytes!.Value, 1024L * 1024, 1024L * 1024 * 1024 * 1024); // between 1MB and 1TB — sanity bounds, not a real limit
    }
}
