using HartsyInference.Core.Memory.Host;
using Xunit;

namespace HartsyInference.Core.Tests.Runtime;

/// <summary>Free host memory under a cgroup limit, read from fake cgroup trees.</summary>
public sealed class HostMemoryInfoTests : IDisposable
{
    private const long GiB = 1L << 30;
    private readonly string _root = Path.Combine(Path.GetTempPath(), $"cgroup-{Guid.NewGuid():N}");

    public HostMemoryInfoTests() => Directory.CreateDirectory(_root);

    public void Dispose() => Directory.Delete(_root, recursive: true);

    [Fact]
    public void MemAvailable_IsReadInBytes()
    {
        Assert.Equal(3L * 1024, HostMemoryInfo.MemAvailableBytes("MemTotal:  8 kB\nMemFree:  1 kB\nMemAvailable:  3 kB\n"));
        Assert.Null(HostMemoryInfo.MemAvailableBytes("MemTotal:  8 kB\n"));
    }

    [Fact]
    public void V2_LimitLessWorkingSet_CountsInactiveFileCacheAsFree()
    {
        // A 46 GiB pod with 40 GiB charged, 10 GiB of it inactive file cache: 46 - (40 - 10) = 16 GiB.
        WriteV2(_root, $"{46 * GiB}", 40 * GiB, 10 * GiB);
        Assert.Equal(16 * GiB, HostMemoryInfo.CgroupAvailableBytes(_root, "0::/\n"));
    }

    [Fact]
    public void V2_NoLimitAnywhere_IsNull()
    {
        WriteV2(_root, "max", 40 * GiB, 0);
        Assert.Null(HostMemoryInfo.CgroupAvailableBytes(_root, "0::/\n"));
    }

    [Fact]
    public void V2_TheTightestAncestorWins()
    {
        // The process sits in /slice/app; the app has no limit, its slice leaves 4 GiB and the root leaves 20.
        WriteV2(_root, $"{64 * GiB}", 44 * GiB, 0);
        WriteV2(Path.Combine(_root, "slice"), $"{12 * GiB}", 8 * GiB, 0);
        WriteV2(Path.Combine(_root, "slice", "app"), "max", 8 * GiB, 0);
        Assert.Equal(4 * GiB, HostMemoryInfo.CgroupAvailableBytes(_root, "0::/slice/app\n"));
    }

    [Fact]
    public void V2_AGroupTheMountDoesNotShow_FallsBackToTheRoot()
    {
        WriteV2(_root, $"{46 * GiB}", 6 * GiB, 0);
        Assert.Equal(40 * GiB, HostMemoryInfo.CgroupAvailableBytes(_root, "0::/docker/abc123\n"));
    }

    [Fact]
    public void Usage_OverTheLimit_IsZeroNotNegative()
    {
        WriteV2(_root, $"{4 * GiB}", 5 * GiB, 0);
        Assert.Equal(0, HostMemoryInfo.CgroupAvailableBytes(_root, "0::/\n"));
    }

    [Fact]
    public void V1_LimitAndTotalInactiveFile()
    {
        string v1 = Path.Combine(_root, "memory");
        Directory.CreateDirectory(v1);
        File.WriteAllText(Path.Combine(v1, "memory.limit_in_bytes"), $"{32 * GiB}\n");
        File.WriteAllText(Path.Combine(v1, "memory.usage_in_bytes"), $"{30 * GiB}\n");
        File.WriteAllText(Path.Combine(v1, "memory.stat"), $"cache 1\ninactive_file 1\ntotal_inactive_file {6 * GiB}\n");
        Assert.Equal(8 * GiB, HostMemoryInfo.CgroupAvailableBytes(_root, "12:memory:/\n"));
    }

    [Fact]
    public void V1_UnlimitedSentinel_IsNull()
    {
        string v1 = Path.Combine(_root, "memory");
        Directory.CreateDirectory(v1);
        File.WriteAllText(Path.Combine(v1, "memory.limit_in_bytes"), "9223372036854771712\n");
        File.WriteAllText(Path.Combine(v1, "memory.usage_in_bytes"), $"{30 * GiB}\n");
        Assert.Null(HostMemoryInfo.CgroupAvailableBytes(_root, "12:memory:/\n"));
    }

    [Fact]
    public void NoCgroupFiles_IsNull() => Assert.Null(HostMemoryInfo.CgroupAvailableBytes(_root, "0::/\n"));

    private static void WriteV2(string dir, string max, long current, long inactiveFile)
    {
        Directory.CreateDirectory(dir);
        File.WriteAllText(Path.Combine(dir, "memory.max"), max + "\n");
        File.WriteAllText(Path.Combine(dir, "memory.current"), $"{current}\n");
        File.WriteAllText(Path.Combine(dir, "memory.stat"), $"anon {current - inactiveFile}\nfile {inactiveFile}\ninactive_file {inactiveFile}\n");
    }
}
