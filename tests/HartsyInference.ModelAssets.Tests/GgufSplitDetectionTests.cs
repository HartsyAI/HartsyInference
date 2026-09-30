using System.Text;
using HartsyInference.Core.Exceptions;
using HartsyInference.ModelAssets.Gguf;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

public sealed class GgufSplitDetectionTests : IDisposable
{
    private const uint TypeUInt16 = 2;
    private const uint TypeUInt32 = 4;
    private const uint TypeString = 8;

    private readonly string _dir = Path.Combine(Path.GetTempPath(), $"gguf_split_{Guid.NewGuid():N}");

    public GgufSplitDetectionTests() => Directory.CreateDirectory(_dir);

    public void Dispose() => Directory.Delete(_dir, recursive: true);

    [Fact]
    public void Load_SplitFile_IsRefusedWithMergeHint()
    {
        string path = WriteGguf("part.gguf", splitCount: (TypeUInt16, 3));

        using GgufLoader loader = new GgufLoader();
        UnsupportedModelException error = Assert.Throws<UnsupportedModelException>(() => loader.Load(path));

        Assert.Contains("split GGUF", error.Message);
        Assert.Contains("--merge", error.Message);
        Assert.Contains("part 1 of 3", error.Message);
    }

    [Fact]
    public void Load_SplitCountAsUInt32_IsAlsoDetected()
    {
        string path = WriteGguf("part32.gguf", splitCount: (TypeUInt32, 2));

        using GgufLoader loader = new GgufLoader();

        Assert.Throws<UnsupportedModelException>(() => loader.Load(path));
    }

    [Fact]
    public void Load_SplitCountOne_LoadsNormally()
    {
        string path = WriteGguf("single.gguf", splitCount: (TypeUInt16, 1));

        using GgufLoader loader = new GgufLoader();
        loader.Load(path);

        Assert.Equal("llama", loader.Metadata.GetString("general.architecture"));
    }

    [Fact]
    public void Load_NoSplitKeys_LoadsNormally()
    {
        string path = WriteGguf("plain.gguf", splitCount: null);

        using GgufLoader loader = new GgufLoader();
        loader.Load(path);

        Assert.Empty(loader.Descriptors);
    }

    private string WriteGguf(string name, (uint Type, uint Value)? splitCount)
    {
        string path = Path.Combine(_dir, name);
        using FileStream stream = new FileStream(path, FileMode.Create, FileAccess.Write);
        using BinaryWriter writer = new BinaryWriter(stream);
        writer.Write(0x46554747u);
        writer.Write(3u);
        writer.Write(0UL);
        writer.Write(splitCount is null ? 1UL : 3UL);
        WriteString(writer, "general.architecture");
        writer.Write(TypeString);
        WriteString(writer, "llama");
        if (splitCount is { } split)
        {
            WriteString(writer, "split.no");
            writer.Write(TypeUInt16);
            writer.Write((ushort)0);
            WriteString(writer, "split.count");
            writer.Write(split.Type);
            if (split.Type == TypeUInt16)
                writer.Write((ushort)split.Value);
            else
                writer.Write(split.Value);
        }
        return path;
    }

    private static void WriteString(BinaryWriter writer, string value)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(value);
        writer.Write((ulong)bytes.Length);
        writer.Write(bytes);
    }
}
