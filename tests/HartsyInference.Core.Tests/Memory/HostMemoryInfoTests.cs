using HartsyInference.Core.Memory.Host;
using Xunit;

namespace HartsyInference.Core.Tests.Memory;

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
    public void V2_ASubtreeMount_ResolvesTheMembershipAgainstTheMountRoot()
    {
        // The mount shows /docker/abc; the process sits in /docker/abc/worker, capped at 4 GiB under the container's 32.
        WriteV2(_root, $"{32 * GiB}", 2 * GiB, 0);
        WriteV2(Path.Combine(_root, "worker"), $"{4 * GiB}", 1 * GiB, 0);
        string mountInfo = $"30 25 0:26 /docker/abc {Escaped(_root)} rw,nosuid shared:4 - cgroup2 cgroup2 rw\n";
        Assert.Equal(3 * GiB, HostMemoryInfo.CgroupAvailableBytes(_root, "0::/docker/abc/worker\n", mountInfo));
    }

    [Fact]
    public void V2_AMembershipOutsideTheMountRoot_ReadsTheMount()
    {
        WriteV2(_root, $"{32 * GiB}", 2 * GiB, 0);
        string mountInfo = $"30 25 0:26 /docker/abc {Escaped(_root)} rw - cgroup2 cgroup2 rw\n";
        Assert.Equal(30 * GiB, HostMemoryInfo.CgroupAvailableBytes(_root, "0::/system.slice/other\n", mountInfo));
    }

    [Fact]
    public void V2_ALevelWithoutUsage_IsSkippedAndTheAncestorCounts()
    {
        WriteV2(_root, $"{46 * GiB}", 6 * GiB, 0);
        string app = Path.Combine(_root, "app");
        Directory.CreateDirectory(app);
        File.WriteAllText(Path.Combine(app, "memory.max"), $"{GiB}\n");   // no memory.current
        Assert.Equal(40 * GiB, HostMemoryInfo.CgroupAvailableBytes(_root, "0::/app\n"));
    }

    [Fact]
    public void V2_APathOutsideTheNamespaceRoot_IsReadAsTheRoot()
    {
        WriteV2(_root, $"{46 * GiB}", 6 * GiB, 0);
        Assert.Equal(40 * GiB, HostMemoryInfo.CgroupAvailableBytes(_root, "0::/../../user.slice\n"));
    }

    [Fact]
    public void AnUnreadableLevel_KeepsTheLevelsAlreadyRead()
    {
        WriteV2(_root, $"{64 * GiB}", 4 * GiB, 0);
        WriteV2(Path.Combine(_root, "app"), $"{8 * GiB}", 6 * GiB, 0);
        // Hold the root's stat file exclusively, so reading that level throws; the app's 2 GiB, read first, must survive.
        using FileStream locked = new(Path.Combine(_root, "memory.stat"), FileMode.Open, FileAccess.ReadWrite, FileShare.None);
        Assert.Equal(2 * GiB, HostMemoryInfo.CgroupAvailableBytes(_root, "0::/app\n"));
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
        WriteV1(Path.Combine(_root, "memory"), 32 * GiB, 30 * GiB, $"cache 1\ninactive_file 1\ntotal_inactive_file {6 * GiB}\n");
        Assert.Equal(8 * GiB, HostMemoryInfo.CgroupAvailableBytes(_root, "12:memory:/\n"));
    }

    [Fact]
    public void V1_TheProcessGroupUnderAnUnlimitedRoot_IsRead()
    {
        WriteV1(Path.Combine(_root, "memory"), 9223372036854771712, 30 * GiB, "");
        WriteV1(Path.Combine(_root, "memory", "system.slice", "app.service"), 16 * GiB, 10 * GiB, "");
        Assert.Equal(6 * GiB, HostMemoryInfo.CgroupAvailableBytes(_root, "12:cpu,memory:/system.slice/app.service\n"));
    }

    [Fact]
    public void V1_HierarchicalLimitCapsTheGroupsOwn()
    {
        // The group sets no limit of its own, but an ancestor the mount does not show does: memory.stat carries it down.
        WriteV1(Path.Combine(_root, "memory"), 9223372036854771712, 10 * GiB, $"hierarchical_memory_limit {12 * GiB}\n");
        Assert.Equal(2 * GiB, HostMemoryInfo.CgroupAvailableBytes(_root, "12:memory:/\n"));
    }

    [Fact]
    public void V1_UnlimitedSentinel_IsNull()
    {
        WriteV1(Path.Combine(_root, "memory"), 9223372036854771712, 30 * GiB, "");
        Assert.Null(HostMemoryInfo.CgroupAvailableBytes(_root, "12:memory:/\n"));
    }

    [Fact]
    public void Hybrid_WithNoV2Limit_FallsBackToTheV1Controller()
    {
        WriteV1(Path.Combine(_root, "memory"), 20 * GiB, 5 * GiB, "");
        Assert.Equal(15 * GiB, HostMemoryInfo.CgroupAvailableBytes(_root, "12:memory:/\n0::/\n"));
    }

    [Fact]
    public void NoCgroupFiles_IsNull() => Assert.Null(HostMemoryInfo.CgroupAvailableBytes(_root, "0::/\n"));

    // mountinfo writes a space in a path as \040.
    private static string Escaped(string path) => path.Replace(" ", "\\040");

    private static void WriteV2(string dir, string max, long current, long inactiveFile)
    {
        Directory.CreateDirectory(dir);
        File.WriteAllText(Path.Combine(dir, "memory.max"), max + "\n");
        File.WriteAllText(Path.Combine(dir, "memory.current"), $"{current}\n");
        File.WriteAllText(Path.Combine(dir, "memory.stat"), $"anon {current - inactiveFile}\nfile {inactiveFile}\ninactive_file {inactiveFile}\n");
    }

    private static void WriteV1(string dir, long limit, long usage, string stat)
    {
        Directory.CreateDirectory(dir);
        File.WriteAllText(Path.Combine(dir, "memory.limit_in_bytes"), $"{limit}\n");
        File.WriteAllText(Path.Combine(dir, "memory.usage_in_bytes"), $"{usage}\n");
        File.WriteAllText(Path.Combine(dir, "memory.stat"), stat);
    }
}
