using HartsyInference.Core.Tensors;
using HartsyInference.Engine.Audio;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The on-disk sizing the audio switch check falls back to before a model has ever loaded: only weight files
/// count, a symbolic stand-in counts as the file it points at, and a runner that widens half-precision weights to F32
/// is sized from the checkpoint header rather than the file length. Temp folders only — a hub id would size the real
/// model cache, so that branch is left to the integration path.</summary>
public sealed class AudioWeightFootprintTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), $"hartsy-footprint-{Guid.NewGuid():N}");

    public AudioWeightFootprintTests() => Directory.CreateDirectory(_root);

    [Fact]
    public void Folder_CountsWeightFilesAtAnyDepth_AndNothingElse()
    {
        WriteBytes("model.safetensors", 3000);
        WriteBytes(Path.Combine("BiCodec", "model.pth"), 500);
        WriteBytes("tokenizer.json", 7000);
        WriteBytes("config.json", 100);
        WriteBytes("weights.bin.tmp", 9000);

        Assert.Equal(3500, AudioWeightFootprint.Estimate(_root, "tts"));
    }

    [Fact]
    public void SymbolicStandIn_CountsTheFileItPointsAt()
    {
        string target = WriteBytes("dia-1_6b_fp32.safetensors", 4096);
        string standIn = Path.Combine(_root, "repo", "pytorch_model.bin");
        Directory.CreateDirectory(Path.GetDirectoryName(standIn)!);
        File.CreateSymbolicLink(standIn, target);

        Assert.Equal(4096, AudioWeightFootprint.Estimate(standIn, "tts"));
        Assert.Equal(4096, AudioWeightFootprint.Estimate(Path.GetDirectoryName(standIn), "tts"));
    }

    [Fact]
    public void PromotingRunner_SizesHalfPrecisionTensorsAtF32_FromTheHeader()
    {
        // Named like the converted Dia stand-in: the format is sniffed, not read off the extension.
        string path = Path.Combine(_root, "pytorch_model.bin");
        using (Tensor half = new(new TensorShape(4, 8), DType.BF16))
        using (Tensor full = new(new TensorShape(16), DType.F32))
        {
            SafeTensorsWriter.Save(path, new Dictionary<string, Tensor> { ["proj.weight"] = half, ["norm.weight"] = full });
        }

        Assert.Equal(new FileInfo(path).Length, AudioWeightFootprint.Estimate(path, "tts"));
        Assert.Equal(4 * 8 * sizeof(float) + 16 * sizeof(float), AudioWeightFootprint.Estimate(path, "tts", promotesHalfToF32: true));
    }

    [Theory]
    [InlineData(null)]
    [InlineData("yue2")]
    public void NothingToSize_IsZero(string? source)
    {
        Assert.Equal(0, AudioWeightFootprint.Estimate(source, "music"));
    }

    private string WriteBytes(string relative, int length)
    {
        string path = Path.Combine(_root, relative);
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllBytes(path, new byte[length]);
        return path;
    }

    public void Dispose() => Directory.Delete(_root, recursive: true);
}
