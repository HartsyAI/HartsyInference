using System.Buffers.Binary;
using System.Security.Cryptography;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Runtime;

namespace HartsyInference.Vulkan;

/// <summary>The vendor, device and pipeline-cache UUID a driver blob must name to belong to this device and driver build.</summary>
internal readonly record struct VulkanPipelineCacheIdentity(uint VendorId, uint DeviceId, string PipelineCacheUuid);

/// <summary>Why a pipeline-cache file did or did not yield a blob.</summary>
internal enum VulkanPipelineCacheLoadStatus
{
    Loaded,
    Missing,
    Empty,
    Unreadable,
    /// <summary>Shorter than the envelope.</summary>
    TruncatedEnvelope,
    BadMagic,
    UnsupportedVersion,
    /// <summary>The envelope's blob length is not the number of bytes that follow it.</summary>
    LengthMismatch,
    ChecksumMismatch,
    /// <summary>The blob does not start with a <c>VkPipelineCacheHeaderVersionOne</c>.</summary>
    BadDriverHeader,
    VendorMismatch,
    DeviceMismatch,
    UuidMismatch,
}

/// <summary>A load's outcome; <see cref="Blob"/> is empty unless the status is <see cref="VulkanPipelineCacheLoadStatus.Loaded"/>.</summary>
/// <param name="Detail">The I/O error's message when the status is <see cref="VulkanPipelineCacheLoadStatus.Unreadable"/>.</param>
internal readonly record struct VulkanPipelineCacheLoad(VulkanPipelineCacheLoadStatus Status, ReadOnlyMemory<byte> Blob, string? Detail = null)
{
    /// <summary>True when the file is whole but names another device or driver build, as after a driver update.</summary>
    public bool IsForAnotherDevice => Status is VulkanPipelineCacheLoadStatus.VendorMismatch
        or VulkanPipelineCacheLoadStatus.DeviceMismatch or VulkanPipelineCacheLoadStatus.UuidMismatch;
}

/// <summary>The on-disk Vulkan pipeline cache: the driver blob behind an envelope with its length and SHA-256, replaced atomically.</summary>
/// <remarks>Little-endian layout: magic <c>HVPC</c> (4), format version (4), blob length (8), SHA-256 of the blob (32), the blob.
/// The driver's own header names the device but not the blob's length, so it cannot show a file cut short.</remarks>
internal static class VulkanPipelineCacheFile
{
    /// <summary>Bytes in front of the driver blob.</summary>
    public const int EnvelopeSize = 48;

    /// <summary>Size of <c>VkPipelineCacheHeaderVersionOne</c>, the smallest blob a driver returns.</summary>
    public const int DriverHeaderSize = 32;

    /// <summary>Suffix of the enveloped file; a new name, so builds that read the bare <c>.pipeline_cache</c> blob never meet an envelope.</summary>
    public const string Extension = ".pipeline_cache.v2";

    private const uint Magic = 0x43505648;
    private const uint FormatVersion = 1;
    private const uint DriverHeaderVersionOne = 1;
    private const string TempSuffix = ".tmp";

    /// <summary>A temporary file older than this was left by a writer that died.</summary>
    private static readonly TimeSpan StaleTempAge = TimeSpan.FromHours(1);

    /// <summary>Reads and validates <paramref name="path"/>; a failure is a status with no blob, never an exception.</summary>
    public static VulkanPipelineCacheLoad Load(string path, VulkanPipelineCacheIdentity expected)
    {
        byte[] file;
        try
        {
            // Sharing delete lets a concurrent WriteAtomic replace the file under this handle on Windows.
            using FileStream stream = new(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete);
            if (stream.Length > Array.MaxLength) return new(VulkanPipelineCacheLoadStatus.LengthMismatch, default);
            file = new byte[stream.Length];
            stream.ReadExactly(file);
        }
        catch (Exception ex) when (ex is FileNotFoundException or DirectoryNotFoundException)
        {
            return new(VulkanPipelineCacheLoadStatus.Missing, default);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            return new(VulkanPipelineCacheLoadStatus.Unreadable, default, ex.Message);
        }
        return Parse(file, expected);
    }

    /// <summary>Validates an enveloped file's bytes and returns the driver blob inside it.</summary>
    public static VulkanPipelineCacheLoad Parse(ReadOnlyMemory<byte> file, VulkanPipelineCacheIdentity expected)
    {
        ReadOnlySpan<byte> bytes = file.Span;
        if (bytes.Length == 0) return new(VulkanPipelineCacheLoadStatus.Empty, default);
        if (bytes.Length < EnvelopeSize) return new(VulkanPipelineCacheLoadStatus.TruncatedEnvelope, default);
        if (BinaryPrimitives.ReadUInt32LittleEndian(bytes) != Magic) return new(VulkanPipelineCacheLoadStatus.BadMagic, default);
        if (BinaryPrimitives.ReadUInt32LittleEndian(bytes[4..]) != FormatVersion)
            return new(VulkanPipelineCacheLoadStatus.UnsupportedVersion, default);
        if (BinaryPrimitives.ReadUInt64LittleEndian(bytes[8..]) != (ulong)(bytes.Length - EnvelopeSize))
            return new(VulkanPipelineCacheLoadStatus.LengthMismatch, default);

        ReadOnlySpan<byte> blob = bytes[EnvelopeSize..];
        Span<byte> digest = stackalloc byte[SHA256.HashSizeInBytes];
        SHA256.HashData(blob, digest);
        if (!digest.SequenceEqual(bytes.Slice(16, SHA256.HashSizeInBytes)))
            return new(VulkanPipelineCacheLoadStatus.ChecksumMismatch, default);

        VulkanPipelineCacheLoadStatus header = CheckDriverHeader(blob, expected);
        return header == VulkanPipelineCacheLoadStatus.Loaded ? new(header, file[EnvelopeSize..]) : new(header, default);
    }

    /// <summary>Compares a driver blob's header with <paramref name="expected"/>; <c>Loaded</c> means it matches.</summary>
    public static VulkanPipelineCacheLoadStatus CheckDriverHeader(ReadOnlySpan<byte> blob, VulkanPipelineCacheIdentity expected)
    {
        if (blob.Length < DriverHeaderSize
            || BinaryPrimitives.ReadUInt32LittleEndian(blob) != DriverHeaderSize
            || BinaryPrimitives.ReadUInt32LittleEndian(blob[4..]) != DriverHeaderVersionOne)
            return VulkanPipelineCacheLoadStatus.BadDriverHeader;
        if (BinaryPrimitives.ReadUInt32LittleEndian(blob[8..]) != expected.VendorId) return VulkanPipelineCacheLoadStatus.VendorMismatch;
        if (BinaryPrimitives.ReadUInt32LittleEndian(blob[12..]) != expected.DeviceId) return VulkanPipelineCacheLoadStatus.DeviceMismatch;
        return Convert.ToHexString(blob.Slice(16, 16)).Equals(expected.PipelineCacheUuid, StringComparison.OrdinalIgnoreCase)
            ? VulkanPipelineCacheLoadStatus.Loaded
            : VulkanPipelineCacheLoadStatus.UuidMismatch;
    }

    /// <summary>Wraps <paramref name="blob"/> in the envelope <see cref="Parse"/> reads.</summary>
    public static byte[] Wrap(ReadOnlySpan<byte> blob)
    {
        byte[] file = new byte[EnvelopeSize + blob.Length];
        BinaryPrimitives.WriteUInt32LittleEndian(file, Magic);
        BinaryPrimitives.WriteUInt32LittleEndian(file.AsSpan(4), FormatVersion);
        BinaryPrimitives.WriteUInt64LittleEndian(file.AsSpan(8), (ulong)blob.Length);
        SHA256.HashData(blob, file.AsSpan(16, SHA256.HashSizeInBytes));
        blob.CopyTo(file.AsSpan(EnvelopeSize));
        return file;
    }

    /// <summary>Replaces <paramref name="path"/> with the enveloped <paramref name="blob"/>: a temporary file beside it, moved over it.</summary>
    /// <remarks>A reader sees the previous file or the new one, and a writer that dies leaves the previous file.
    /// Throws when the write or the move fails, with the temporary file removed.</remarks>
    public static void WriteAtomic(string path, ReadOnlySpan<byte> blob)
    {
        // Process and thread make the name unique among writers; one holds it for a single synchronous call.
        string temp = $"{path}.{Environment.ProcessId}.{Environment.CurrentManagedThreadId}{TempSuffix}";
        bool moved = false;
        try
        {
            using (FileStream stream = new(temp, FileMode.Create, FileAccess.Write, FileShare.None))
            {
                stream.Write(Wrap(blob));
            }
            AtomicFileReplace.Move(temp, path);
            moved = true;
        }
        finally
        {
            if (!moved) TryDelete(temp);
        }
        RemoveStaleTemps(path);
    }

    private static void RemoveStaleTemps(string path)
    {
        try
        {
            string? directory = Path.GetDirectoryName(path);
            if (string.IsNullOrEmpty(directory)) return;
            DateTime cutoff = DateTime.UtcNow - StaleTempAge;
            foreach (FileInfo temp in new DirectoryInfo(directory).EnumerateFiles(Path.GetFileName(path) + ".*" + TempSuffix))
            {
                // A name that vanished since the listing reads as year 1601, and its writer may be reusing it.
                temp.Refresh();
                if (temp.Exists && temp.LastWriteTimeUtc < cutoff) TryDelete(temp.FullName);
            }
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            Logs.Verbose($"VulkanPipelineCache: could not sweep temporary files beside {path}: {ex.Message}");
        }
    }

    private static void TryDelete(string file)
    {
        try
        {
            File.Delete(file);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            Logs.Verbose($"VulkanPipelineCache: could not delete {file}: {ex.Message}");
        }
    }
}
