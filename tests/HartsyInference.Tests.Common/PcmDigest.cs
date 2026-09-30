using System.Security.Cryptography;

namespace HartsyInference.Tests.Common;

/// <summary>SHA-256 over the raw bytes of float PCM, for "identical output" gates on audio refactors. Two buffers
/// with the same digest are bit-for-bit the same samples; a single ULP anywhere changes it, which is the point.</summary>
public static class PcmDigest
{
    /// <summary>Hex digest of one buffer.</summary>
    public static string Of(ReadOnlySpan<float> samples) =>
        Convert.ToHexString(SHA256.HashData(MemoryMarshal.AsBytes(samples)));

    /// <summary>Hex digest of several buffers as if concatenated, so a streamed result can be compared to a whole one.</summary>
    public static string Of(IEnumerable<float[]> chunks)
    {
        ArgumentNullException.ThrowIfNull(chunks);
        using IncrementalHash hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        foreach (float[] chunk in chunks)
        {
            hash.AppendData(MemoryMarshal.AsBytes(chunk.AsSpan()));
        }
        return Convert.ToHexString(hash.GetHashAndReset());
    }

    /// <summary>Joins chunks into one buffer.</summary>
    public static float[] Concat(IEnumerable<float[]> chunks)
    {
        ArgumentNullException.ThrowIfNull(chunks);
        List<float[]> list = [.. chunks];
        long total = 0;
        foreach (float[] chunk in list)
        {
            total += chunk.Length;
        }
        float[] joined = new float[total];
        int offset = 0;
        foreach (float[] chunk in list)
        {
            chunk.CopyTo(joined, offset);
            offset += chunk.Length;
        }
        return joined;
    }
}
