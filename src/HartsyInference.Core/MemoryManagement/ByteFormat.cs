namespace HartsyInference.Core.MemoryManagement;

/// <summary>Byte-count formatting for VRAM/weight log lines.</summary>
public static class ByteFormat
{
    /// <summary>Whole mebibytes, e.g. "512 MB".</summary>
    public static string Mb(long bytes) => $"{bytes / (1024 * 1024)} MB";

    /// <summary>One-decimal mebibytes, e.g. "512.3 MB".</summary>
    public static string MbF1(long bytes) => $"{bytes / (1024.0 * 1024):F1} MB";
}
