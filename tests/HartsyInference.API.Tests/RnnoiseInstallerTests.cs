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

    /// <summary>The wake stack scores on F32 by construction: with the int8 tables installed beside the weights for
    /// the voice front end, it still loads F32 and builds F32 streams.</summary>
    [Fact]
    public void WakeStack_LoadsFloat_EvenWithTheInt8TablesInstalled()
    {
        Directory.CreateDirectory(Path.Combine(_root, "denoise"));
        Save(RnnoiseInstaller.WeightsPath(_root), SyntheticFloatWeights());
        Save(RnnoiseInstaller.Int8TablesPath(_root), SyntheticInt8Tables());
        using (RnnoiseWeights int8 = RnnoiseWeights.LoadFile(RnnoiseInstaller.WeightsPath(_root), RnnoisePrecision.Int8))
        {
            Assert.Equal(RnnoisePrecision.Int8, int8.Precision);   // the tables are loadable, so the F32 below is a choice
        }

        using WakeModelSet models = new(_root);
        Assert.True(models.LoadDenoiser());
        Assert.Equal(RnnoisePrecision.Float, models.DenoisePrecision);
    }

    private static Dictionary<string, Tensor> SyntheticFloatWeights()
    {
        (string Name, long[] Shape)[] layout =
        [
            ("conv1.weight", [128, 65, 3]), ("conv1.bias", [128]), ("conv2.weight", [384, 128, 3]), ("conv2.bias", [384]),
            ("dense_out.weight", [32, 1536]), ("dense_out.bias", [32]), ("vad_dense.weight", [1, 1536]),
            ("vad_dense.bias", [1]),
        ];
        Dictionary<string, Tensor> tensors = new(StringComparer.Ordinal);
        foreach ((string name, long[] shape) in layout) tensors[name] = Filled(new TensorShape(shape), 0.01f);
        for (int layer = 1; layer <= 3; layer++)
        {
            foreach (string kind in new[] { "ih", "hh" })
            {
                tensors[$"gru{layer}.weight_{kind}_l0"] = Filled(new TensorShape(1152, 384), 0.01f);
                tensors[$"gru{layer}.bias_{kind}_l0"] = Filled(new TensorShape(1152), 0.01f);
            }
        }
        return tensors;
    }

    private static Dictionary<string, Tensor> SyntheticInt8Tables()
    {
        Dictionary<string, Tensor> tables = new(StringComparer.Ordinal);
        foreach ((string name, int count) in RnnoiseInt8Tables.Arrays)
        {
            bool int8 = name.EndsWith("_weights_int8", StringComparison.Ordinal);
            if (!int8)
            {
                tables[name] = Filled(new TensorShape(count), 1e-5f);
                continue;
            }
            Tensor weights = new(new TensorShape(count), DType.I8);
            weights.AsSpan<sbyte>().Fill(1);
            tables[name] = weights;
        }
        return tables;
    }

    private static Tensor Filled(TensorShape shape, float value)
    {
        Tensor tensor = new(shape, DType.F32);
        tensor.AsSpan<float>().Fill(value);
        return tensor;
    }

    private static void Save(string path, Dictionary<string, Tensor> tensors)
    {
        try
        {
            SafeTensorsWriter.Save(path, tensors);
        }
        finally
        {
            foreach (Tensor tensor in tensors.Values) tensor.Dispose();
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
