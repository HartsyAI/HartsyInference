using System.Text;
using HartsyInference.Core.Exceptions;
using HartsyInference.ModelAssets.Gguf;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

public sealed class GgufSplitDetectionTests : IDisposable
{
    private const uint TypeF32 = 0;
    private const uint TypeInt32 = 5;
    private const uint TypeUInt16 = 2;
    private const uint TypeString = 8;
    private const int Alignment = 32;

    private readonly string _dir = Path.Combine(Path.GetTempPath(), $"gguf_split_{Guid.NewGuid():N}");

    public GgufSplitDetectionTests() => Directory.CreateDirectory(_dir);

    public void Dispose() => Directory.Delete(_dir, recursive: true);

    [Fact]
    public void Load_CompleteSplit_IndexesEveryPartsTensors()
    {
        WritePart(1, 3, 3, ("blk.0.a", [1f, 2f, 3f, 4f]));
        WritePart(2, 3, 3, ("blk.1.b", [10f, 11f, 12f]));
        string last = WritePart(3, 3, 3, ("output.c", [20f, 21f]));

        using GgufLoader loader = new GgufLoader();
        loader.Load(last);

        Assert.Equal(3, loader.Descriptors.Count);
        Assert.Equal("llama", loader.Metadata.GetString("general.architecture"));
        Assert.Equal(new float[] { 1f, 2f, 3f, 4f }, loader.GetTensor("blk.0.a").AsSpan<float>().ToArray());
        Assert.Equal(new float[] { 10f, 11f, 12f }, loader.GetTensor("blk.1.b").AsSpan<float>().ToArray());
        Assert.Equal(new float[] { 20f, 21f }, loader.GetTensor("output.c").AsSpan<float>().ToArray());
    }

    [Fact]
    public void Load_IncompleteSplit_NamesTheMissingPart()
    {
        string first = WritePart(1, 3, 2, ("a", [1f]));
        WritePart(3, 3, 2, ("c", [3f]));

        using GgufLoader loader = new GgufLoader();
        UnsupportedModelException error = Assert.Throws<UnsupportedModelException>(() => loader.Load(first));

        Assert.Contains("incomplete", error.Message);
        Assert.Contains("00002", error.Message);
    }

    [Fact]
    public void Load_PartDeclaringAnotherPosition_IsRejected()
    {
        string first = WritePart(1, 2, 2, ("a", [1f]));
        WriteFile(2, 2, declaredNo: 0, declaredTensors: 2, ("b", [2f]));

        using GgufLoader loader = new GgufLoader();
        UnsupportedModelException error = Assert.Throws<UnsupportedModelException>(() => loader.Load(first));

        Assert.Contains("should be part 2 of 2", error.Message);
    }

    [Fact]
    public void Load_TensorTotalDisagreesWithParts_IsRejected()
    {
        string first = WritePart(1, 2, 5, ("a", [1f]));
        WritePart(2, 2, 5, ("b", [2f]));

        using GgufLoader loader = new GgufLoader();
        UnsupportedModelException error = Assert.Throws<UnsupportedModelException>(() => loader.Load(first));

        Assert.Contains("declares 5 tensors but its parts hold 2", error.Message);
    }

    [Fact]
    public void Load_TensorInTwoParts_IsRejected()
    {
        string first = WritePart(1, 2, 2, ("a", [1f]));
        WritePart(2, 2, 2, ("a", [2f]));

        using GgufLoader loader = new GgufLoader();
        UnsupportedModelException error = Assert.Throws<UnsupportedModelException>(() => loader.Load(first));

        Assert.Contains("appears twice", error.Message);
    }

    [Fact]
    public void Load_SplitCountOne_LoadsNormally()
    {
        string path = WriteFile(1, 1, declaredNo: 0, declaredTensors: 1, ("a", [1f]));

        using GgufLoader loader = new GgufLoader();
        loader.Load(path);

        Assert.Equal("llama", loader.Metadata.GetString("general.architecture"));
        Assert.Equal(new float[] { 1f }, loader.GetTensor("a").AsSpan<float>().ToArray());
    }

    private string WritePart(int index, int count, int declaredTensors, params (string Name, float[] Values)[] tensors) =>
        WriteFile(index, count, declaredNo: index - 1, declaredTensors, tensors);

    private string WriteFile(int index, int count, int declaredNo, int declaredTensors, params (string Name, float[] Values)[] tensors)
    {
        using MemoryStream infos = new();
        using MemoryStream payload = new();
        using (BinaryWriter info = new(infos, Encoding.UTF8, leaveOpen: true))
        using (BinaryWriter data = new(payload, Encoding.UTF8, leaveOpen: true))
        {
            foreach ((string name, float[] values) in tensors)
            {
                PadTo(data, Alignment);
                WriteString(info, name);
                info.Write(1u);
                info.Write((ulong)values.Length);
                info.Write(TypeF32);
                info.Write((ulong)data.BaseStream.Position);
                foreach (float value in values)
                    data.Write(value);
            }
        }

        using MemoryStream file = new();
        using (BinaryWriter header = new(file, Encoding.UTF8, leaveOpen: true))
        {
            header.Write(0x46554747u);
            header.Write(3u);
            header.Write((ulong)tensors.Length);
            header.Write(4UL);
            WriteString(header, "general.architecture");
            header.Write(TypeString);
            WriteString(header, "llama");
            WriteString(header, "split.no");
            header.Write(TypeUInt16);
            header.Write((ushort)declaredNo);
            WriteString(header, "split.count");
            header.Write(TypeUInt16);
            header.Write((ushort)count);
            WriteString(header, "split.tensors.count");
            header.Write(TypeInt32);
            header.Write(declaredTensors);
            header.Write(infos.ToArray());
            PadTo(header, Alignment);
            header.Write(payload.ToArray());
        }

        string path = Path.Combine(_dir, $"model-{index:D5}-of-{count:D5}.gguf");
        File.WriteAllBytes(path, file.ToArray());
        return path;
    }

    private static void PadTo(BinaryWriter writer, int alignment)
    {
        while (writer.BaseStream.Position % alignment != 0)
            writer.Write((byte)0);
    }

    private static void WriteString(BinaryWriter writer, string value)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(value);
        writer.Write((ulong)bytes.Length);
        writer.Write(bytes);
    }
}
