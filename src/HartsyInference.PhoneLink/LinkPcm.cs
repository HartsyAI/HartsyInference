using System.Buffers.Binary;

namespace HartsyInference.PhoneLink;

/// <summary>PCM16 little-endian sample packing. A straight copy on little-endian hosts, a per-sample swap elsewhere.</summary>
internal static class LinkPcm
{
    public static void Encode(ReadOnlySpan<short> samples, Span<byte> destination)
    {
        if (BitConverter.IsLittleEndian)
        {
            MemoryMarshal.AsBytes(samples).CopyTo(destination);
            return;
        }
        for (int i = 0; i < samples.Length; i++)
            BinaryPrimitives.WriteInt16LittleEndian(destination.Slice(i * 2), samples[i]);
    }

    public static void Decode(ReadOnlySpan<byte> source, Span<short> destination)
    {
        if (BitConverter.IsLittleEndian)
        {
            source.CopyTo(MemoryMarshal.AsBytes(destination));
            return;
        }
        for (int i = 0; i < destination.Length; i++)
            destination[i] = BinaryPrimitives.ReadInt16LittleEndian(source.Slice(i * 2));
    }
}
