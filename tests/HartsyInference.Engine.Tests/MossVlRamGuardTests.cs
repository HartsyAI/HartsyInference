using HartsyInference.Core.Exceptions;
using HartsyInference.Engine.Services;

namespace HartsyInference.Engine.Tests;

/// <summary>The host-RAM check for a MOSS-VL checkpoint folder loaded on the CPU, where every projection is widened to F32.</summary>
public sealed class MossVlRamGuardTests
{
    [Fact]
    public void CpuFolderLoadNeedsTheWidenedShardsInRam()
    {
        string folder = Path.Combine(Path.GetTempPath(), "hartsy-moss-ram-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(folder);
        try
        {
            using (FileStream shard = File.Create(Path.Combine(folder, "model-00001-of-00001.safetensors")))
                shard.SetLength(1L << 30);   // the guard reads the size only

            // 1 GiB of shards needs 2 GiB widened plus the 4 GiB margin: 6 GiB.
            HartsyInferenceException error = Assert.Throws<HartsyInferenceException>(
                () => TextService.EnsureRamHeadroomForMossVlFolderOnCpu(folder, availableKb: 5L << 20));
            Assert.Contains("F32", error.Message);
            TextService.EnsureRamHeadroomForMossVlFolderOnCpu(folder, availableKb: 7L << 20);
            TextService.EnsureRamHeadroomForMossVlFolderOnCpu(folder, availableKb: 0);
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }
}
