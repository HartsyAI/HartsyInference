using System.Text;
using System.Text.Json;
using HartsyInference.Audio.Cache;
using Xunit;

namespace HartsyInference.Audio.Tests.Cache;

/// <summary>A Hartsy repack standing in for an upstream file has a rewritten header, so the upstream file's pinned hash cannot
/// match it. <see cref="AudioStandIns.StandsIn"/> is what lets the cache skip the pin for such a file, and only for it.</summary>
public sealed class AudioStandInPinTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "hartsy-standin-pin-" + Guid.NewGuid().ToString("N"));

    public AudioStandInPinTests() => Directory.CreateDirectory(_root);

    public void Dispose() => Directory.Delete(_root, recursive: true);

    /// <summary>A one-tensor safetensors file, with <c>hartsy.stands_in_for</c> set when <paramref name="standsInFor"/> is given.</summary>
    private static void WriteSafetensors(string path, string? standsInFor)
    {
        Dictionary<string, object> header = new() { ["t"] = new { dtype = "F32", shape = new[] { 1 }, data_offsets = new[] { 0, 4 } } };
        if (standsInFor is not null)
        {
            header["__metadata__"] = new Dictionary<string, string> { [AudioStandIns.MetadataKey] = standsInFor };
        }
        byte[] json = JsonSerializer.SerializeToUtf8Bytes(header);
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        using FileStream file = File.Create(path);
        file.Write(BitConverter.GetBytes((long)json.Length));
        file.Write(json);
        file.Write(new byte[4]);
    }

    [Fact]
    public void AnArtifactDeclaringThisPathStandsInForIt()
    {
        string path = Path.Combine(_root, "tts", "higurashi-ai--Haru-4B", "model.safetensors");
        WriteSafetensors(path, "tts/higurashi-ai--Haru-4B/model.safetensors");
        Assert.True(AudioStandIns.StandsIn(_root, path));
    }

    [Fact]
    public void OneOfSeveralDeclaredPathsIsEnough()
    {
        string path = Path.Combine(_root, "tts", "acme--m", "model.safetensors");
        WriteSafetensors(path, "tts/other--x/a.safetensors;tts/acme--m/model.safetensors");
        Assert.True(AudioStandIns.StandsIn(_root, path));
    }

    [Fact]
    public void AnUpstreamFileWithNoDeclarationIsStillPinned()
    {
        string path = Path.Combine(_root, "tts", "acme--m", "model.safetensors");
        WriteSafetensors(path, standsInFor: null);
        Assert.False(AudioStandIns.StandsIn(_root, path));
    }

    [Fact]
    public void AnArtifactDeclaringAnotherPathDoesNotSkipThisOnesPin()
    {
        string path = Path.Combine(_root, "tts", "acme--m", "model.safetensors");
        WriteSafetensors(path, "tts/acme--m/other.safetensors");
        Assert.False(AudioStandIns.StandsIn(_root, path));
    }

    /// <summary>The file's own header is only trusted for a file that was on disk before the fetch: one just downloaded came from
    /// upstream, so a header claiming to stand in cannot exempt it from its pin.</summary>
    [Fact]
    public void OnlyAFileAlreadyOnDiskCanSkipItsPin()
    {
        string path = Path.Combine(_root, "tts", "higurashi-ai--Haru-4B", "model.safetensors");
        WriteSafetensors(path, "tts/higurashi-ai--Haru-4B/model.safetensors");
        Assert.False(AudioModelCache.PinApplies(wasOnDisk: true, _root, path));
        Assert.True(AudioModelCache.PinApplies(wasOnDisk: false, _root, path));

        string upstream = Path.Combine(_root, "tts", "acme--m", "model.safetensors");
        WriteSafetensors(upstream, standsInFor: null);
        Assert.True(AudioModelCache.PinApplies(wasOnDisk: true, _root, upstream));
    }

    [Fact]
    public void OnWindowsADeclarationDifferingOnlyInCaseNamesTheSameFile()
    {
        string path = Path.Combine(_root, "tts", "higurashi-ai--Haru-4B", "model.safetensors");
        WriteSafetensors(path, "tts/Higurashi-AI--Haru-4B/model.safetensors");
        Assert.Equal(OperatingSystem.IsWindows(), AudioStandIns.StandsIn(_root, path));
    }

    [Fact]
    public void AFileThatIsNotSafetensorsIsNeverAStandIn()
    {
        string path = Path.Combine(_root, "tts", "acme--m", "config.json");
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllText(path, "{}", Encoding.UTF8);
        Assert.False(AudioStandIns.StandsIn(_root, path));

        string broken = Path.Combine(_root, "tts", "acme--m", "broken.safetensors");
        File.WriteAllBytes(broken, [1, 2, 3]);
        Assert.False(AudioStandIns.StandsIn(_root, broken));
    }
}
