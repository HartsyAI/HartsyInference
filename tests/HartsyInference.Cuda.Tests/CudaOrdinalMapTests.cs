using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Prints the live CUDA ordinal → device-name → VRAM map in a machine-parseable format. CUDA enumerates
/// fastest-first, which does NOT match nvidia-smi's PCI-bus order on multi-GPU boxes (verified reversed on the
/// dev 4090+3060 pair) — <c>tests/run-multigpu-campaign.sh</c> runs this first and parses the ORDINAL lines so
/// every later log can name cards instead of guessing ordinals.</summary>
[Collection("CudaSerial")]
public sealed class CudaOrdinalMapTests
{
    private readonly ITestOutputHelper _output;
    public CudaOrdinalMapTests(ITestOutputHelper output) => _output = output;

    [Fact]
    [Trait("Category", "GpuIntegration")]
    public void OrdinalMap_PrintsNameAndFreeVramPerDevice()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }

        IReadOnlyList<GpuTopologyInfo> devices = CudaTopology.Probe();
        Assert.NotEmpty(devices);
        foreach (GpuTopologyInfo gpu in devices)
        {
            _output.WriteLine($"ORDINAL {gpu.Ordinal}: name=\"{gpu.Name}\" sm={gpu.CcMajor}{gpu.CcMinor} "
                + $"free={gpu.FreeMemoryBytes >> 20} MiB total={gpu.TotalMemoryBytes >> 20} MiB");
        }
    }
}
