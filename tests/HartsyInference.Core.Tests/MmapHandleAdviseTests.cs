using HartsyInference.Core.Memory;
using Xunit;

namespace HartsyInference.Core.Tests;

public sealed class MmapHandleAdviseTests : IDisposable
{
    private readonly string _path = Path.Combine(Path.GetTempPath(), $"advise_{Guid.NewGuid():N}.bin");

    public MmapHandleAdviseTests() => File.WriteAllBytes(_path, new byte[64 * 1024]);

    public void Dispose() => File.Delete(_path);

    [Fact]
    public void Advise_RandomOverTheWholeMapping_Succeeds()
    {
        using MmapHandle handle = MmapHandle.OpenRead(_path);

        Assert.Equal(OperatingSystem.IsLinux(), handle.Advise(0, handle.ByteLength, MmapAdvice.Random));
    }

    [Fact]
    public void Advise_UnalignedStart_IsRoundedDownRatherThanRejected()
    {
        using MmapHandle handle = MmapHandle.OpenRead(_path);

        Assert.Equal(OperatingSystem.IsLinux(), handle.Advise(5001, 100, MmapAdvice.WillNeed));
    }

    [Fact]
    public void Advise_OutsideTheMapping_Throws()
    {
        using MmapHandle handle = MmapHandle.OpenRead(_path);

        Assert.Throws<ArgumentOutOfRangeException>(() => handle.Advise(handle.ByteLength - 10, 20, MmapAdvice.Random));
        Assert.Throws<ArgumentOutOfRangeException>(() => handle.Advise(-1, 10, MmapAdvice.Random));
    }

    [Fact]
    public void Advise_AfterDispose_Throws()
    {
        MmapHandle handle = MmapHandle.OpenRead(_path);
        handle.Dispose();

        Assert.Throws<ObjectDisposedException>(() => handle.Advise(0, 10, MmapAdvice.Random));
    }
}
