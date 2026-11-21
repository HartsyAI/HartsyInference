using System.Buffers.Binary;
using HartsyInference.Vulkan;
using Xunit.Abstractions;

namespace HartsyInference.LLM.Tests;

/// <summary>The pipeline-cache file's envelope, its driver-header comparison and its atomic replace, with no device:
/// every file lives in a directory this class creates and removes.
/// <para>Bounds fixed before the first run: a file that is not exactly what <c>WriteAtomic</c> wrote yields no blob
/// and the status named in each test; a load that runs beside two writers is nothing or one of the two written
/// blobs, byte for byte; a writer that dies leaves the previous file loadable.</para></summary>
public sealed class VulkanPipelineCacheFileTests : IDisposable
{
    private static readonly VulkanPipelineCacheIdentity Device = new(0x10DE, 0x2504, "00112233445566778899aabbccddeeff");

    private readonly ITestOutputHelper _output;
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "hartsy-vkpc-tests-" + Guid.NewGuid().ToString("N"));
    private readonly string _path;

    public VulkanPipelineCacheFileTests(ITestOutputHelper output)
    {
        _output = output;
        Directory.CreateDirectory(_dir);
        _path = Path.Combine(_dir, "NVIDIA-2504" + VulkanPipelineCacheFile.Extension);
    }

    public void Dispose() => Directory.Delete(_dir, recursive: true);

    [Theory]
    [InlineData(0)] // header only: what a driver that persists no pipelines returns
    [InlineData(1)]
    [InlineData(100_000)]
    public void WrittenBlobLoadsBackByteForByte(int bodyBytes)
    {
        byte[] blob = DriverBlob(Device, bodyBytes, seed: 1);
        VulkanPipelineCacheFile.WriteAtomic(_path, blob);

        VulkanPipelineCacheLoad load = VulkanPipelineCacheFile.Load(_path, Device);

        Assert.Equal(VulkanPipelineCacheLoadStatus.Loaded, load.Status);
        Assert.Equal(blob, load.Blob.ToArray());
        Assert.Equal(VulkanPipelineCacheFile.EnvelopeSize + blob.Length, new FileInfo(_path).Length);
        Assert.Equal([_path], Directory.GetFiles(_dir));
    }

    /// <summary>Cut inside the envelope, at its end, inside the driver header, just after it, and one byte short.</summary>
    [Theory]
    [InlineData(1, nameof(VulkanPipelineCacheLoadStatus.TruncatedEnvelope))]
    [InlineData(20, nameof(VulkanPipelineCacheLoadStatus.TruncatedEnvelope))]
    [InlineData(47, nameof(VulkanPipelineCacheLoadStatus.TruncatedEnvelope))]
    [InlineData(48, nameof(VulkanPipelineCacheLoadStatus.LengthMismatch))]
    [InlineData(48 + 10, nameof(VulkanPipelineCacheLoadStatus.LengthMismatch))]
    [InlineData(48 + 32, nameof(VulkanPipelineCacheLoadStatus.LengthMismatch))]
    [InlineData(48 + 33, nameof(VulkanPipelineCacheLoadStatus.LengthMismatch))]
    [InlineData(-1, nameof(VulkanPipelineCacheLoadStatus.LengthMismatch))]
    public void TruncatedFileIsRejected(int keep, string expected)
    {
        byte[] file = VulkanPipelineCacheFile.Wrap(DriverBlob(Device, 4096, seed: 2));
        File.WriteAllBytes(_path, file[..(keep < 0 ? file.Length + keep : keep)]);

        VulkanPipelineCacheLoad load = VulkanPipelineCacheFile.Load(_path, Device);

        Assert.Equal(expected, load.Status.ToString());
        Assert.True(load.Blob.IsEmpty);
    }

    [Fact]
    public void FileWithBytesAppendedIsRejected()
    {
        byte[] file = VulkanPipelineCacheFile.Wrap(DriverBlob(Device, 4096, seed: 3));
        File.WriteAllBytes(_path, [.. file, 0]);

        Assert.Equal(VulkanPipelineCacheLoadStatus.LengthMismatch, VulkanPipelineCacheFile.Load(_path, Device).Status);
    }

    /// <summary>One bit flipped past the driver header, where only the checksum can see it; then in the driver
    /// header and in the stored checksum, which the checksum also catches first.</summary>
    [Theory]
    [InlineData(48 + 32)]
    [InlineData(48 + 32 + 2000)]
    [InlineData(48 + 32 + 4095)]
    [InlineData(48 + 8)]
    [InlineData(48 + 20)]
    [InlineData(16)]
    public void FlippedByteIsRejectedByTheChecksum(int offset)
    {
        byte[] file = VulkanPipelineCacheFile.Wrap(DriverBlob(Device, 4096, seed: 4));
        file[offset] ^= 0x01;
        File.WriteAllBytes(_path, file);

        VulkanPipelineCacheLoad load = VulkanPipelineCacheFile.Load(_path, Device);

        Assert.Equal(VulkanPipelineCacheLoadStatus.ChecksumMismatch, load.Status);
        Assert.True(load.Blob.IsEmpty);
    }

    /// <summary>Whole files (checksum recomputed) whose driver header is not this device's, so the header
    /// comparison is what rejects them.</summary>
    [Fact]
    public void WholeFileForAnotherDeviceIsRejectedByTheDriverHeader()
    {
        (VulkanPipelineCacheIdentity Written, VulkanPipelineCacheLoadStatus Expected)[] cases =
        [
            (Device with { VendorId = 0x1002 }, VulkanPipelineCacheLoadStatus.VendorMismatch),
            (Device with { DeviceId = 0x2684 }, VulkanPipelineCacheLoadStatus.DeviceMismatch),
            (Device with { PipelineCacheUuid = "ff112233445566778899aabbccddeeff" }, VulkanPipelineCacheLoadStatus.UuidMismatch),
            (Device with { PipelineCacheUuid = "00112233445566778899aabbccddeefe" }, VulkanPipelineCacheLoadStatus.UuidMismatch),
        ];
        foreach ((VulkanPipelineCacheIdentity written, VulkanPipelineCacheLoadStatus expected) in cases)
        {
            VulkanPipelineCacheFile.WriteAtomic(_path, DriverBlob(written, 4096, seed: 5));

            VulkanPipelineCacheLoad load = VulkanPipelineCacheFile.Load(_path, Device);

            Assert.Equal(expected, load.Status);
            Assert.True(load.IsForAnotherDevice);
            Assert.True(load.Blob.IsEmpty);
        }
    }

    [Fact]
    public void WholeFileWithoutAVersionOneDriverHeaderIsRejected()
    {
        byte[] wrongSize = DriverBlob(Device, 4096, seed: 6);
        BinaryPrimitives.WriteUInt32LittleEndian(wrongSize, 64);
        byte[] wrongVersion = DriverBlob(Device, 4096, seed: 6);
        BinaryPrimitives.WriteUInt32LittleEndian(wrongVersion.AsSpan(4), 2);
        byte[] shorterThanAHeader = DriverBlob(Device, 0, seed: 6)[..31];

        foreach (byte[] blob in (byte[][])[wrongSize, wrongVersion, shorterThanAHeader, []])
        {
            VulkanPipelineCacheFile.WriteAtomic(_path, blob);

            VulkanPipelineCacheLoad load = VulkanPipelineCacheFile.Load(_path, Device);

            Assert.Equal(VulkanPipelineCacheLoadStatus.BadDriverHeader, load.Status);
            Assert.False(load.IsForAnotherDevice);
            Assert.True(load.Blob.IsEmpty);
        }
    }

    [Fact]
    public void EmptyMissingAndForeignFilesGiveNoData()
    {
        Assert.Equal(VulkanPipelineCacheLoadStatus.Missing, VulkanPipelineCacheFile.Load(_path, Device).Status);
        Assert.Equal(VulkanPipelineCacheLoadStatus.Missing,
            VulkanPipelineCacheFile.Load(Path.Combine(_dir, "absent", "x" + VulkanPipelineCacheFile.Extension), Device).Status);

        File.WriteAllBytes(_path, []);
        Assert.Equal(VulkanPipelineCacheLoadStatus.Empty, VulkanPipelineCacheFile.Load(_path, Device).Status);

        // A bare driver blob, which is what the earlier file format held: its first word is 32, not the magic.
        File.WriteAllBytes(_path, DriverBlob(Device, 4096, seed: 7));
        Assert.Equal(VulkanPipelineCacheLoadStatus.BadMagic, VulkanPipelineCacheFile.Load(_path, Device).Status);

        byte[] newerFormat = VulkanPipelineCacheFile.Wrap(DriverBlob(Device, 4096, seed: 7));
        BinaryPrimitives.WriteUInt32LittleEndian(newerFormat.AsSpan(4), 2);
        File.WriteAllBytes(_path, newerFormat);
        Assert.Equal(VulkanPipelineCacheLoadStatus.UnsupportedVersion, VulkanPipelineCacheFile.Load(_path, Device).Status);

        Directory.CreateDirectory(_path + ".d");
        VulkanPipelineCacheLoad unreadable = VulkanPipelineCacheFile.Load(_path + ".d", Device);
        Assert.Equal(VulkanPipelineCacheLoadStatus.Unreadable, unreadable.Status);
        Assert.False(string.IsNullOrEmpty(unreadable.Detail));
    }

    /// <summary>Two threads replace one file 300 times each with blobs of different sizes while a third loads it.
    /// <para>Bounds, fixed before the first run: no load reports a cut-short or altered file (only Loaded, Missing or
    /// Unreadable), every loaded blob is one of the two written, the reader loads at least one blob, the file left
    /// at the end loads, and no temporary file remains. A write that fails is counted: expected 0, reported, and
    /// bounded at a tenth of the writes.</para></summary>
    [Fact]
    public void LoadBesideTwoWritersSeesAWholeFileOrNone()
    {
        const int writesPerThread = 300;
        byte[] small = DriverBlob(Device, 40_000, seed: 8);
        byte[] large = DriverBlob(Device, 1_800_000, seed: 9);
        int failedWrites = 0, loaded = 0, unreadable = 0, missing = 0, running = 2;
        List<string> violations = [];

        Thread Writer(byte[] blob) => new(() =>
        {
            for (int i = 0; i < writesPerThread; i++)
            {
                try
                {
                    VulkanPipelineCacheFile.WriteAtomic(_path, blob);
                }
                catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
                {
                    Interlocked.Increment(ref failedWrites);
                }
            }
            Interlocked.Decrement(ref running);
        });
        Thread reader = new(() =>
        {
            while (Volatile.Read(ref running) > 0)
            {
                VulkanPipelineCacheLoad load = VulkanPipelineCacheFile.Load(_path, Device);
                switch (load.Status)
                {
                    case VulkanPipelineCacheLoadStatus.Loaded:
                        loaded++;
                        if (!load.Blob.Span.SequenceEqual(small) && !load.Blob.Span.SequenceEqual(large))
                            violations.Add($"loaded {load.Blob.Length} bytes that neither writer wrote");
                        break;
                    case VulkanPipelineCacheLoadStatus.Missing: missing++; break;
                    case VulkanPipelineCacheLoadStatus.Unreadable: unreadable++; break;
                    default: violations.Add(load.Status.ToString()); break;
                }
            }
        });

        Thread[] threads = [Writer(small), Writer(large), reader];
        foreach (Thread thread in threads) thread.Start();
        foreach (Thread thread in threads) thread.Join();

        _output.WriteLine($"writes failed {failedWrites} of {2 * writesPerThread}; "
            + $"loads: {loaded} whole, {unreadable} unreadable, {missing} missing");
        Assert.Empty(violations);
        Assert.True(loaded > 0, "the reader never loaded a blob, so the run showed nothing");
        Assert.True(failedWrites <= 2 * writesPerThread / 10, $"{failedWrites} of {2 * writesPerThread} writes failed");
        Assert.Equal(VulkanPipelineCacheLoadStatus.Loaded, VulkanPipelineCacheFile.Load(_path, Device).Status);
        Assert.Equal([_path], Directory.GetFiles(_dir));
    }

    /// <summary>A writer that died mid-write is its partial temporary file beside the untouched previous file.</summary>
    [Fact]
    public void WriterThatDiedMidWriteLeavesThePreviousFileLoadable()
    {
        byte[] previous = DriverBlob(Device, 50_000, seed: 10);
        VulkanPipelineCacheFile.WriteAtomic(_path, previous);
        byte[] next = VulkanPipelineCacheFile.Wrap(DriverBlob(Device, 80_000, seed: 11));
        string deadWriterTemp = _path + ".999999.7.tmp";
        File.WriteAllBytes(deadWriterTemp, next[..(next.Length * 6 / 10)]);

        VulkanPipelineCacheLoad load = VulkanPipelineCacheFile.Load(_path, Device);

        Assert.Equal(VulkanPipelineCacheLoadStatus.Loaded, load.Status);
        Assert.Equal(previous, load.Blob.ToArray());

        // A later write leaves a fresh temporary file alone (its writer may be alive) and removes an old one.
        VulkanPipelineCacheFile.WriteAtomic(_path, previous);
        Assert.True(File.Exists(deadWriterTemp));
        File.SetLastWriteTimeUtc(deadWriterTemp, DateTime.UtcNow.AddHours(-2));
        VulkanPipelineCacheFile.WriteAtomic(_path, previous);
        Assert.Equal([_path], Directory.GetFiles(_dir));
        Assert.Equal(VulkanPipelineCacheLoadStatus.Loaded, VulkanPipelineCacheFile.Load(_path, Device).Status);
    }

    [Fact]
    public void FailedWriteThrowsAndLeavesNoTemporaryFile()
    {
        byte[] previous = DriverBlob(Device, 5000, seed: 12);
        VulkanPipelineCacheFile.WriteAtomic(_path, previous);
        // A handle that shares nothing makes the move over the target fail on Windows; elsewhere a directory does.
        string target = OperatingSystem.IsWindows() ? _path : Path.Combine(_dir, "occupied");
        if (!OperatingSystem.IsWindows()) Directory.CreateDirectory(Path.Combine(target, "child"));
        using (FileStream? held = OperatingSystem.IsWindows() ? new(_path, FileMode.Open, FileAccess.Read, FileShare.None) : null)
        {
            Exception? thrown = Record.Exception(() => VulkanPipelineCacheFile.WriteAtomic(target, DriverBlob(Device, 9000, seed: 13)));

            Assert.True(thrown is IOException or UnauthorizedAccessException, $"threw {thrown?.GetType().Name ?? "nothing"}");
        }
        Assert.Empty(Directory.GetFiles(_dir, "*.tmp"));
        Assert.Equal(previous, VulkanPipelineCacheFile.Load(_path, Device).Blob.ToArray());
    }

    private static byte[] DriverBlob(VulkanPipelineCacheIdentity identity, int bodyBytes, int seed)
    {
        byte[] blob = new byte[VulkanPipelineCacheFile.DriverHeaderSize + bodyBytes];
        BinaryPrimitives.WriteUInt32LittleEndian(blob, VulkanPipelineCacheFile.DriverHeaderSize);
        BinaryPrimitives.WriteUInt32LittleEndian(blob.AsSpan(4), 1);
        BinaryPrimitives.WriteUInt32LittleEndian(blob.AsSpan(8), identity.VendorId);
        BinaryPrimitives.WriteUInt32LittleEndian(blob.AsSpan(12), identity.DeviceId);
        Convert.FromHexString(identity.PipelineCacheUuid).CopyTo(blob.AsSpan(16));
        new Random(seed).NextBytes(blob.AsSpan(VulkanPipelineCacheFile.DriverHeaderSize));
        return blob;
    }
}
