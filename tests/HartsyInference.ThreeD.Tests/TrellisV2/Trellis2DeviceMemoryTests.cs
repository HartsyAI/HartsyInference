using HartsyInference.Core.Backends.Cpu;
using HartsyInference.ThreeD.Pipelines.Stages;
using Xunit;

namespace HartsyInference.ThreeD.Tests.TrellisV2;

public sealed class Trellis2DeviceMemoryTests
{
    // The release calls matter on a CUDA card (a flow that starts on top of the previous stage's pool and cached weights runs 6 to 35
    // times slower on 12 GB), which only a real-weight run shows; on the CPU backend they must stay inert.
    [Fact]
    public void ReleaseBetweenStages_IsInertOnTheCpuBackend()
    {
        using CpuBackend backend = new();

        Trellis2DeviceMemory.ReleaseBetweenStages(backend);
        Trellis2DeviceMemory.ReleaseAll(backend);
        Trellis2DeviceMemory.ReleaseBetweenStages(backend);
    }
}
