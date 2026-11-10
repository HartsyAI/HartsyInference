using System.IO.Compression;
using System.Text;
using Xunit;

namespace HartsyInference.ThreeD.Tests.Parity;

/// <summary>Unit check of the npz reader on an archive written here, so the parity lane's fixture reader is verified without the fixtures.</summary>
public sealed class NpzArchiveTests
{
    [Fact]
    public void Open_ReadsFloatIntAndBoolEntriesWithTheirShapes()
    {
        string path = Path.Combine(Path.GetTempPath(), $"npz-reader-{Guid.NewGuid():N}.npz");
        try
        {
            using (ZipArchive zip = ZipFile.Open(path, ZipArchiveMode.Create))
            {
                WriteEntry(zip, "floats.npy", "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }", Bytes(BitConverter.GetBytes(1.5f), BitConverter.GetBytes(-2f), BitConverter.GetBytes(0.25f), BitConverter.GetBytes(4f), BitConverter.GetBytes(5f), BitConverter.GetBytes(6f)));
                WriteEntry(zip, "coords.npy", "{'descr': '<i4', 'fortran_order': False, 'shape': (1, 4), }", Bytes(BitConverter.GetBytes(7), BitConverter.GetBytes(8), BitConverter.GetBytes(9), BitConverter.GetBytes(-10)));
                WriteEntry(zip, "flags.npy", "{'descr': '|b1', 'fortran_order': False, 'shape': (3,), }", [1, 0, 1]);
            }

            NpzArchive archive = NpzArchive.Open(path);

            Assert.Equal(new float[] { 1.5f, -2f, 0.25f, 4f, 5f, 6f }, archive.Get("floats").RequireShape(2, 3).ToFloat32());
            Assert.Equal(new int[] { 7, 8, 9, -10 }, archive.Get("coords").RequireShape(1, 4).ToInt32());
            Assert.Equal(new bool[] { true, false, true }, archive.Get("flags").RequireShape(3).ToBool());
            Assert.Throws<InvalidDataException>(() => archive.Get("floats").RequireShape(3, 2));
            Assert.Throws<InvalidDataException>(() => archive.Get("missing"));
        }
        finally
        {
            File.Delete(path);
        }
    }

    // Writes a version-1.0 .npy entry; the header is padded so the data starts on a 64-byte boundary, as numpy does.
    private static void WriteEntry(ZipArchive zip, string name, string header, byte[] data)
    {
        int unpadded = 10 + header.Length + 1;
        string padded = header + new string(' ', (64 - unpadded % 64) % 64) + "\n";
        byte[] headerBytes = Encoding.Latin1.GetBytes(padded);
        using MemoryStream npy = new();
        npy.Write([0x93, (byte)'N', (byte)'U', (byte)'M', (byte)'P', (byte)'Y', 1, 0]);
        npy.Write(BitConverter.GetBytes((ushort)headerBytes.Length));
        npy.Write(headerBytes);
        npy.Write(data);
        using Stream entry = zip.CreateEntry(name).Open();
        npy.Position = 0;
        npy.CopyTo(entry);
    }

    private static byte[] Bytes(params byte[][] parts)
    {
        using MemoryStream stream = new();
        foreach (byte[] part in parts)
            stream.Write(part);
        return stream.ToArray();
    }
}
