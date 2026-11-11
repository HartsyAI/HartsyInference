using System.Runtime.InteropServices;
using System.Text;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Worlds;

namespace HartsyInference.Engine.Tests.Worlds;

public sealed class WorldCameraFileTests
{
    [Fact]
    public void Json_ObjectWithPosesAndIntrinsics_Parses()
    {
        WorldCameraPath path = WorldCameraFile.Parse(
            Encoding.UTF8.GetBytes(WorldTestFixtures.CameraJson(3)), ".json");

        Assert.Equal(3, path.Poses.Count);
        Assert.Equal(16, path.Poses[0].Length);
        Assert.Equal(new WorldIntrinsics { Fx = 320, Fy = 320, Cx = 320, Cy = 180 }, path.Intrinsics);
    }

    [Fact]
    public void Json_BareArrayOfFourByFourPoses_FlattensRowMajor()
    {
        const string json = "[[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]]";

        WorldCameraPath path = WorldCameraFile.Parse(Encoding.UTF8.GetBytes(json), ".JSON");

        Assert.Null(path.Intrinsics);
        Assert.Equal(16, path.Poses[0].Length);
        Assert.Equal(1f, path.Poses[0][0]);
        Assert.Equal(1f, path.Poses[0][15]);
    }

    [Fact]
    public void Npy_Float32_NByBy16_Parses()
    {
        float[] values = new float[2 * 16];
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = i;
        }

        WorldCameraPath path = WorldCameraFile.Parse(Npy("<f4", "2, 16", MemoryMarshal.AsBytes(values.AsSpan()).ToArray()), ".npy");

        Assert.Equal(2, path.Poses.Count);
        Assert.Equal(16f, path.Poses[1][0]);
        Assert.Equal(31f, path.Poses[1][15]);
    }

    [Fact]
    public void Npy_Float64_NByByFourByFour_Parses()
    {
        double[] values = new double[16];
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = i * 0.5;
        }

        WorldCameraPath path = WorldCameraFile.Parse(Npy("<f8", "1, 4, 4", MemoryMarshal.AsBytes(values.AsSpan()).ToArray()), ".npy");

        Assert.Single(path.Poses);
        Assert.Equal(7.5f, path.Poses[0][15]);
    }

    [Fact]
    public void Npy_FortranOrder_IsRefused()
    {
        byte[] bytes = Npy("<f4", "1, 16", new byte[64], fortranOrder: true);

        Assert.Throws<InvalidDataException>(() => WorldCameraFile.Parse(bytes, ".npy"));
    }

    [Fact]
    public void Npy_WrongShape_IsRefused()
    {
        byte[] bytes = Npy("<f4", "2, 9", new byte[72]);

        Assert.Throws<InvalidDataException>(() => WorldCameraFile.Parse(bytes, ".npy"));
    }

    [Fact]
    public void UnsupportedExtension_IsRefused()
    {
        Assert.Throws<InvalidDataException>(() => WorldCameraFile.Parse(new byte[] { 0x7b }, ".txt"));
    }

    [Fact]
    public void Intrinsics_TextParsesFourNumbers()
    {
        Assert.Equal(new WorldIntrinsics { Fx = 320.5, Fy = 321, Cx = 320, Cy = 180 }, WorldCameraFile.ParseIntrinsics("320.5, 321,320,180"));
    }

    [Theory]
    [InlineData("320,320,320")]
    [InlineData("a,b,c,d")]
    public void Intrinsics_TextThatIsNotFourNumbers_IsRefused(string text)
    {
        Assert.Throws<ArgumentException>(() => WorldCameraFile.ParseIntrinsics(text));
    }

    [Fact]
    public void MissingCameraFile_IsReportedAsMissing()
    {
        Assert.Throws<FileNotFoundException>(() => WorldCameraFile.Load(Path.Combine(Path.GetTempPath(), $"absent-{Guid.NewGuid():N}.npy")));
    }

    /// <summary>A version-1 NumPy file: the magic, the header padded to a 64-byte boundary, then the data.</summary>
    [Fact]
    public void IntrinsicsNpy_PerFrameMatrices_ReadFxFyCxCy()
    {
        float[] k = [800f, 0f, 844f, 0f, 830f, 463f, 0f, 0f, 1f, 700f, 0f, 640f, 0f, 710f, 360f, 0f, 0f, 1f];
        string path = TempFile(".npy", Npy("<f4", "2, 3, 3", MemoryMarshal.AsBytes(k.AsSpan()).ToArray()));
        try
        {
            IReadOnlyList<WorldIntrinsics> intrinsics = WorldCameraFile.LoadIntrinsics(path);

            Assert.Equal(2, intrinsics.Count);
            Assert.Equal(new WorldIntrinsics { Fx = 800, Fy = 830, Cx = 844, Cy = 463 }, intrinsics[0]);
            Assert.Equal(new WorldIntrinsics { Fx = 700, Fy = 710, Cx = 640, Cy = 360 }, intrinsics[1]);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void IntrinsicsNpy_SingleFloat64Vector_IsOneEntry()
    {
        double[] k = [320.5, 321, 320, 180];
        string path = TempFile(".npy", Npy("<f8", "4,", MemoryMarshal.AsBytes(k.AsSpan()).ToArray()));
        try
        {
            Assert.Equal([new WorldIntrinsics { Fx = 320.5, Fy = 321, Cx = 320, Cy = 180 }], WorldCameraFile.LoadIntrinsics(path));
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void IntrinsicsNpy_WrongShape_IsRefused()
    {
        string path = TempFile(".npy", Npy("<f4", "2, 3", new byte[24]));
        try
        {
            Assert.Throws<InvalidDataException>(() => WorldCameraFile.LoadIntrinsics(path));
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void IntrinsicsOption_ReadsAFileWhenOneExists_AndTextOtherwise()
    {
        string path = TempFile(".json", Encoding.UTF8.GetBytes(
            "[{\"fx\": 1, \"fy\": 2, \"cx\": 3, \"cy\": 4}, {\"fx\": 5, \"fy\": 6, \"cx\": 7, \"cy\": 8}]"));
        try
        {
            Assert.Equal(2, WorldCameraFile.ParseIntrinsicsOption(path).Count);
            Assert.Equal([new WorldIntrinsics { Fx = 9, Fy = 10, Cx = 11, Cy = 12 }], WorldCameraFile.ParseIntrinsicsOption("9,10,11,12"));
            Assert.Throws<ArgumentException>(() => WorldCameraFile.ParseIntrinsicsOption("missing-file.npy"));
        }
        finally
        {
            File.Delete(path);
        }
    }

    private static string TempFile(string extension, byte[] bytes)
    {
        string path = Path.Combine(Path.GetTempPath(), $"world-intrinsics-{Guid.NewGuid():N}{extension}");
        File.WriteAllBytes(path, bytes);
        return path;
    }

    private static byte[] Npy(string descr, string shape, byte[] data, bool fortranOrder = false)
    {
        string dict = "{'descr': '" + descr + "', 'fortran_order': " + (fortranOrder ? "True" : "False") + ", 'shape': (" + shape + "), }";
        int unpadded = 10 + dict.Length + 1;
        string header = dict + new string(' ', (64 - unpadded % 64) % 64) + "\n";
        byte[] headerBytes = Encoding.ASCII.GetBytes(header);
        List<byte> bytes = [0x93, (byte)'N', (byte)'U', (byte)'M', (byte)'P', (byte)'Y', 1, 0,
            (byte)(headerBytes.Length & 0xff), (byte)(headerBytes.Length >> 8)];
        bytes.AddRange(headerBytes);
        bytes.AddRange(data);
        return bytes.ToArray();
    }
}
