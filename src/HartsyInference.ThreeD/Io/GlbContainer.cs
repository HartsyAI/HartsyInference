using System.Buffers.Binary;

namespace HartsyInference.ThreeD.Io;

/// <summary>Frames a glTF JSON chunk and a binary chunk into a GLB 2.0 container (JSON padded with spaces, BIN with zeros).</summary>
internal static class GlbContainer
{
    private const uint Magic = 0x46546C67;
    private const uint JsonChunk = 0x4E4F534A;
    private const uint BinChunk = 0x004E4942;

    public static byte[] Assemble(byte[] json, byte[] bin)
    {
        int jsonLength = Align(json.Length), binLength = Align(bin.Length);
        long total = 12L + 8 + jsonLength + 8 + binLength;
        if (total > uint.MaxValue)
            throw new InvalidOperationException("The GLB container would exceed 4 GiB.");
        byte[] glb = new byte[total];
        int p = 0;
        WriteU32(glb, ref p, Magic);
        WriteU32(glb, ref p, 2);
        WriteU32(glb, ref p, (uint)total);
        WriteU32(glb, ref p, (uint)jsonLength);
        WriteU32(glb, ref p, JsonChunk);
        json.CopyTo(glb, p);
        for (int i = json.Length; i < jsonLength; i++) glb[p + i] = 0x20;
        p += jsonLength;
        WriteU32(glb, ref p, (uint)binLength);
        WriteU32(glb, ref p, BinChunk);
        bin.CopyTo(glb, p);
        return glb;
    }

    /// <summary>Rounds up to a multiple of four bytes.</summary>
    public static int Align(int length) => (length + 3) & ~3;

    private static void WriteU32(byte[] dst, ref int offset, uint value)
    {
        BinaryPrimitives.WriteUInt32LittleEndian(dst.AsSpan(offset), value);
        offset += 4;
    }
}
