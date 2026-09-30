using System.Formats.Tar;
using System.IO.Compression;
using HartsyInference.Audio.Models.Denoise;
using HartsyInference.Core.Tensors;
using HartsyInference.Engine.Audio.Wake;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.API.Tests;

/// <summary>Installing the RNNoise denoiser from xiph's own release, to the path the wake stack loads it from.</summary>
public sealed class RnnoiseInstallerTests(ITestOutputHelper log) : IDisposable
{
    private readonly string _root = Directory.CreateTempSubdirectory("rnnoise-install-").FullName;

    /// <summary>An offline install takes only the pinned release: anything else is refused before it is unpacked.</summary>
    [Fact]
    public void InstallFromTarball_RefusesATarballThatIsNotThePinnedRelease()
    {
        string tarball = Path.Combine(_root, "rnnoise_data-other.tar.gz");
        using (FileStream file = File.Create(tarball))
        using (GZipStream gzip = new(file, CompressionLevel.Fastest))
        using (TarWriter writer = new(gzip, TarEntryFormat.Pax))
        {
            writer.WriteEntry(new PaxTarEntry(TarEntryType.RegularFile, RnnoiseCheckpoint.CheckpointMember)
            {
                DataStream = new MemoryStream([1, 2, 3]),
            });
        }

        InvalidDataException error = Assert.Throws<InvalidDataException>(() =>
            RnnoiseInstaller.InstallFromTarball(tarball, _root));

        Assert.Contains(RnnoiseInstaller.TarballSha256, error.Message);
        Assert.False(Directory.Exists(Path.Combine(_root, "denoise")));
    }

    /// <summary>The real thing, end to end: download from media.xiph.org, verify, convert, and have the wake stack
    /// load the result. Opt in with <c>HARTSY_ALLOW_NETWORK_TESTS=1</c> under <c>--filter Network=Real</c>; with
    /// <c>HARTSYINFERENCE_RNNOISE_WEIGHTS</c> set, the result is also compared with that file tensor by tensor.</summary>
    [Fact]
    [Trait("Network", "Real")]
    public async Task EnsureAsync_InstallsTheOfficialRelease_WhereTheWakeStackLoadsIt()
    {
        if (Environment.GetEnvironmentVariable("HARTSY_ALLOW_NETWORK_TESTS") != "1")
        {
            log.WriteLine("SKIPPED: set HARTSY_ALLOW_NETWORK_TESTS=1 to download xiph's ~59 MB model tarball");
            return;
        }

        string path = await RnnoiseInstaller.EnsureAsync(_root, CancellationToken.None);

        Assert.Equal(RnnoiseInstaller.WeightsPath(_root), path);
        string[] left = [.. Directory.GetFiles(Path.GetDirectoryName(path)!).Select(f => Path.GetFileName(f))];
        Assert.Equal(["rnnoise.safetensors"], left);
        using (SafeTensorsLoader loader = new())
        {
            loader.Load(path);
            Assert.Equal(RnnoiseInstaller.TarballSha256, loader.Metadata!["hartsy.source_sha256"]);
            Assert.Equal(RnnoiseCheckpoint.License, loader.Metadata["hartsy.license"]);
        }
        using WakeModelSet models = new(_root);
        Assert.True(models.LoadDenoiser());
        Assert.True(models.DenoiseAvailable);

        string? reference = Environment.GetEnvironmentVariable("HARTSYINFERENCE_RNNOISE_WEIGHTS");
        if (reference is not null && File.Exists(reference))
        {
            AssertSameTensors(reference, path);
            log.WriteLine($"installed file is tensor-identical to {reference}");
        }
    }

    private static void AssertSameTensors(string expectedPath, string actualPath)
    {
        using SafeTensorsLoader expected = new();
        expected.Load(expectedPath);
        using SafeTensorsLoader actual = new();
        actual.Load(actualPath);
        Assert.Equal(expected.Descriptors.Keys.Order(), actual.Descriptors.Keys.Order());
        foreach (string name in expected.Descriptors.Keys)
        {
            using Tensor a = expected.GetTensor(name);
            using Tensor b = actual.GetTensor(name);
            Assert.Equal(a.Shape, b.Shape);
            Assert.True(a.AsSpan<byte>().SequenceEqual(b.AsSpan<byte>()), $"'{name}' differs");
        }
    }

    public void Dispose() => Directory.Delete(_root, recursive: true);
}
