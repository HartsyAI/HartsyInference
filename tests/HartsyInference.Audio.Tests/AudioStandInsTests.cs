using HartsyInference.Audio.Cache;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.Audio.Tests;

public sealed class AudioStandInsTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), $"standins-{Guid.NewGuid():N}");

    public AudioStandInsTests() => Directory.CreateDirectory(_root);

    public void Dispose() => Directory.Delete(_root, recursive: true);

    [Fact]
    public void Sync_LinksAnArtifactIntoEveryDeclaredPathAndRecordsIt()
    {
        string artifact = WriteArtifact("tts/Dia/dia_fp32.safetensors", "tts/nari-labs--Dia-1.6B-0626/pytorch_model.bin; tts/other--repo/weights.pth");

        Assert.Equal(2, AudioStandIns.Sync(_root));

        string linked = Path.Combine(_root, "tts/nari-labs--Dia-1.6B-0626/pytorch_model.bin");
        Assert.True(File.Exists(linked));
        Assert.Equal(File.ReadAllBytes(artifact), File.ReadAllBytes(linked));
        Assert.Null(new FileInfo(linked).LinkTarget); // a hard link, so size checks see the real file
        Assert.Contains("pytorch_model.bin", File.ReadAllText(Path.Combine(_root, ".hartsy-standins.json")));
        Assert.Equal(0, AudioStandIns.Sync(_root));
    }

    [Fact]
    public void Sync_NeverReplacesAnExistingFile()
    {
        string upstream = Path.Combine(_root, "stt/openai--whisper-base/model.safetensors");
        Directory.CreateDirectory(Path.GetDirectoryName(upstream)!);
        File.WriteAllText(upstream, "original");
        WriteArtifact("stt/Whisper/whisper-base_fp32.safetensors", "stt/openai--whisper-base/model.safetensors");

        Assert.Equal(0, AudioStandIns.Sync(_root));
        Assert.Equal("original", File.ReadAllText(upstream));
    }

    [Fact]
    public void Sync_RemovesLinksWhenTheArtifactIsDeleted()
    {
        string artifact = WriteArtifact("fx/Demucs/demucs_fp16.safetensors", "fx/demucs/htdemucs.th");
        AudioStandIns.Sync(_root);
        string linked = Path.Combine(_root, "fx/demucs/htdemucs.th");
        Assert.True(File.Exists(linked));

        File.Delete(artifact);
        AudioStandIns.Sync(_root);

        Assert.False(File.Exists(linked));
    }

    [Theory]
    [InlineData("../escape.bin")]
    [InlineData("/etc/passwd")]
    [InlineData("tts/./x.bin")]
    public void ReadDeclared_DropsPathsOutsideTheRoot(string declared)
    {
        string artifact = WriteArtifact("tts/X/x_fp32.safetensors", declared);
        Assert.Empty(AudioStandIns.ReadDeclared(artifact));
        Assert.Equal(0, AudioStandIns.Sync(_root));
    }

    private string WriteArtifact(string relative, string standsInFor)
    {
        string path = Path.Combine(_root, relative);
        using Tensor t = new(new TensorShape(4), DType.F32);
        SafeTensorsMerger.Write(path, [new("w", t)], metadata: new Dictionary<string, string> { [AudioStandIns.MetadataKey] = standsInFor });
        return path;
    }
}
