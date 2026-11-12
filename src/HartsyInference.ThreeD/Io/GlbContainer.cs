using System.Buffers.Binary;

namespace HartsyInference.ThreeD.Io;

/// <summary>Frames a glTF JSON chunk and a binary chunk into a GLB 2.0 container (JSON padded with spaces, BIN with zeros).</summary>
internal static class GlbContainer
{
    internal const uint Magic = 0x46546C67;
    internal const uint JsonChunk = 0x4E4F534A;
    internal const uint BinChunk = 0x004E4942;

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

    /// <summary>The chunks of a GLB: the JSON chunk's payload and where it sits, and where the binary chunk's payload sits (length 0 when there is none); the binary payload is not copied.</summary>
    public readonly record struct Chunks(byte[] Json, int JsonOffset, int JsonLength, int BinOffset, int BinLength);

    /// <summary>Splits a GLB 2.0 container into its JSON and binary chunks.</summary>
    /// <exception cref="HartsyInference.Core.Exceptions.HartsyInferenceException">Not a GLB 2.0, a chunk runs past the end, or there is no JSON chunk.</exception>
    public static Chunks Split(byte[] glb)
    {
        ArgumentNullException.ThrowIfNull(glb);
        if (glb.Length < 20 || BinaryPrimitives.ReadUInt32LittleEndian(glb) != Magic)
            throw new HartsyInference.Core.Exceptions.HartsyInferenceException("File is not a binary glTF (GLB) container.");
        if (BinaryPrimitives.ReadUInt32LittleEndian(glb.AsSpan(4)) != 2)
            throw new HartsyInference.Core.Exceptions.HartsyInferenceException("Only glTF 2.0 GLB containers are supported.");

        byte[]? json = null;
        int binOffset = 0, binLength = 0;
        int jsonOffset = 0, jsonLength = 0;
        long offset = 12;
        while (offset + 8 <= glb.Length)
        {
            long length = BinaryPrimitives.ReadUInt32LittleEndian(glb.AsSpan((int)offset));
            uint type = BinaryPrimitives.ReadUInt32LittleEndian(glb.AsSpan((int)offset + 4));
            long start = offset + 8;
            if (start + length > glb.Length)
                throw new HartsyInference.Core.Exceptions.HartsyInferenceException("GLB chunk runs past the end of the file.");
            byte[]? chunk = type == JsonChunk ? glb.AsSpan((int)start, (int)length).ToArray() : null;
            if (type == JsonChunk)
            {
                json = chunk;
                jsonOffset = (int)start;
                jsonLength = (int)length;
            }
            else if (type == BinChunk)
            {
                binOffset = (int)start;
                binLength = (int)length;
            }
            offset = start + length;
        }

        return json is null
            ? throw new HartsyInference.Core.Exceptions.HartsyInferenceException("GLB file has no JSON chunk.")
            : new Chunks(json, jsonOffset, jsonLength, binOffset, binLength);
    }

    /// <summary>Replaces the JSON chunk of <paramref name="glb"/> with <paramref name="json"/>, padded with spaces to four bytes, and fixes the chunk and container lengths.</summary>
    public static byte[] ReplaceJson(byte[] glb, byte[] json)
    {
        Chunks chunks = Split(glb);
        int padded = Align(json.Length);
        long total = (long)glb.Length - chunks.JsonLength + padded;
        if (total > uint.MaxValue)
            throw new InvalidOperationException("The GLB container would exceed 4 GiB.");
        byte[] result = new byte[total];
        int payload = chunks.JsonOffset;
        Buffer.BlockCopy(glb, 0, result, 0, payload);
        Buffer.BlockCopy(json, 0, result, payload, json.Length);
        Array.Fill(result, (byte)0x20, payload + json.Length, padded - json.Length);
        Buffer.BlockCopy(glb, payload + chunks.JsonLength, result, payload + padded, glb.Length - payload - chunks.JsonLength);
        BinaryPrimitives.WriteUInt32LittleEndian(result.AsSpan(8), (uint)total);
        BinaryPrimitives.WriteUInt32LittleEndian(result.AsSpan(payload - 8), (uint)padded);
        return result;
    }

    /// <summary>Rounds up to a multiple of four bytes.</summary>
    public static int Align(int length) => (length + 3) & ~3;

    private static void WriteU32(byte[] dst, ref int offset, uint value)
    {
        BinaryPrimitives.WriteUInt32LittleEndian(dst.AsSpan(offset), value);
        offset += 4;
    }
}
