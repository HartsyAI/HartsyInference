using System.Formats.Tar;
using System.IO.Compression;
using HartsyInference.Audio.Models.Denoise;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Turning xiph's model tarball into <c>rnnoise.safetensors</c>.
///
/// <para>The unit tests stand a safetensors payload in for the tarball's <c>.pth</c> (the repacker sniffs content,
/// not extension), which exercises extraction, validation and the replace-only-when-loadable rule without a torch
/// pickle in the repo. The real tarball is covered by the integration test at the bottom.</para></summary>
public sealed class RnnoiseCheckpointTests(ITestOutputHelper log) : IDisposable
{
    private readonly string _dir = Directory.CreateTempSubdirectory("rnnoise-ckpt-").FullName;

    /// <summary>Every tensor RnnoiseWeights binds, at the shipped model's shapes (cond 128, GRU 384).</summary>
    private static readonly (string Name, long[] Shape)[] Layout =
    [
        ("conv1.weight", [128, 65, 3]), ("conv1.bias", [128]),
        ("conv2.weight", [384, 128, 3]), ("conv2.bias", [384]),
        ("gru1.weight_ih_l0", [1152, 384]), ("gru1.weight_hh_l0", [1152, 384]),
        ("gru1.bias_ih_l0", [1152]), ("gru1.bias_hh_l0", [1152]),
        ("gru2.weight_ih_l0", [1152, 384]), ("gru2.weight_hh_l0", [1152, 384]),
        ("gru2.bias_ih_l0", [1152]), ("gru2.bias_hh_l0", [1152]),
        ("gru3.weight_ih_l0", [1152, 384]), ("gru3.weight_hh_l0", [1152, 384]),
        ("gru3.bias_ih_l0", [1152]), ("gru3.bias_hh_l0", [1152]),
        ("dense_out.weight", [32, 1536]), ("dense_out.bias", [32]),
        ("vad_dense.weight", [1, 1536]), ("vad_dense.bias", [1]),
    ];

    [Fact]
    public void ExtractMember_CopiesOnlyTheNamedFile()
    {
        byte[] wanted = [1, 2, 3, 4, 5];
        string tarball = WriteTarball(("src/rnnoise_data.c", [9, 9]), (RnnoiseCheckpoint.CheckpointMember, wanted));
        string extracted = Path.Combine(_dir, "out.pth");

        RnnoiseCheckpoint.ExtractMember(tarball, RnnoiseCheckpoint.CheckpointMember, extracted);

        Assert.Equal(wanted, File.ReadAllBytes(extracted));
    }

    [Fact]
    public void ExtractMember_Throws_WhenTheCheckpointIsAbsent()
    {
        string tarball = WriteTarball(("models/rnnoise10Gb_15.pth", [1]));
        Assert.Throws<InvalidDataException>(() =>
            RnnoiseCheckpoint.ExtractMember(tarball, RnnoiseCheckpoint.CheckpointMember, Path.Combine(_dir, "x")));
    }

    [Fact]
    public void ConvertTarball_WritesLoadableWeights_WithMetadata_AndLeavesNoTemporaries()
    {
        string checkpoint = WriteCheckpoint(Layout);
        string tarball = WriteTarball((RnnoiseCheckpoint.CheckpointMember, File.ReadAllBytes(checkpoint)));
        File.Delete(checkpoint);
        string output = Path.Combine(_dir, "denoise", "rnnoise.safetensors");

        Dictionary<string, string> metadata = new() { ["hartsy.license"] = RnnoiseCheckpoint.License };
        RnnoiseCheckpoint.ConvertTarball(tarball, output, metadata);

        using SafeTensorsLoader loader = new();
        loader.Load(output);
        Assert.Equal(RnnoiseCheckpoint.License, loader.Metadata!["hartsy.license"]);
        Dictionary<string, Tensor> tensors = loader.GetAllTensors();
        Assert.Equal(RnnoiseWeights.TensorCount, tensors.Count);
        // Values survive the round trip, not just names: the last element of each tensor encodes its index.
        for (int i = 0; i < Layout.Length; i++)
            Assert.Equal((float)i, tensors[Layout[i].Name].AsSpan<float>()[^1]);
        foreach (Tensor tensor in tensors.Values) tensor.Dispose();
        string[] left = [.. Directory.GetFiles(Path.GetDirectoryName(output)!).Select(f => Path.GetFileName(f))];
        Assert.Equal(["rnnoise.safetensors"], left);
    }

    /// <summary>A checkpoint of another width (upstream's own trainer defaults to a 256-wide GRU) must fail the
    /// install, and must not replace a working file already in place.</summary>
    [Fact]
    public void ConvertCheckpoint_RejectsAnotherArchitecture_AndKeepsTheExistingFile()
    {
        (string Name, long[] Shape)[] narrow = [.. Layout];
        narrow[Array.FindIndex(narrow, t => t.Name == "gru2.weight_hh_l0")] = ("gru2.weight_hh_l0", [768, 256]);
        string checkpoint = WriteCheckpoint(narrow);
        string output = Path.Combine(_dir, "rnnoise.safetensors");
        File.WriteAllBytes(output, [42]);

        HartsyInferenceException error = Assert.Throws<HartsyInferenceException>(() =>
            RnnoiseCheckpoint.ConvertCheckpoint(checkpoint, output));

        Assert.Contains("gru2.weight_hh_l0", error.Message);
        Assert.Equal([42], File.ReadAllBytes(output));
        Assert.False(File.Exists(output + ".staging"));
    }

    [Fact]
    public void ConvertCheckpoint_RejectsExtraTensors()
    {
        (string Name, long[] Shape)[] extra = [.. Layout, ("gru4.weight_ih_l0", [1152, 384])];
        string checkpoint = WriteCheckpoint(extra);
        Assert.Throws<InvalidDataException>(() =>
            RnnoiseCheckpoint.ConvertCheckpoint(checkpoint, Path.Combine(_dir, "rnnoise.safetensors")));
    }

    /// <summary>The real tarball through the C# path, compared tensor by tensor with the offline converter's output
    /// (<c>tools/convert_rnnoise.py</c>). Byte equality, because both read the same float32 storage and neither
    /// should touch a value; file hashes would differ anyway, since the two writers order tensors differently.</summary>
    [Fact]
    [Trait("Category", "Integration")]
    public void ConvertTarball_RealXiphTarball_MatchesTheOfflineConverter()
    {
        string? tarball = Environment.GetEnvironmentVariable("HARTSYINFERENCE_RNNOISE_TARBALL");
        if (tarball is null || !File.Exists(tarball))   // tier-lint: guarded
        {
            log.WriteLine("SKIPPED: set HARTSYINFERENCE_RNNOISE_TARBALL to xiph's rnnoise_data-*.tar.gz");
            return;
        }
        string output = Path.Combine(_dir, "rnnoise.safetensors");
        RnnoiseCheckpoint.ConvertTarball(tarball, output);

        string? reference = Environment.GetEnvironmentVariable("HARTSYINFERENCE_RNNOISE_WEIGHTS");
        if (reference is null || !File.Exists(reference))
        {
            log.WriteLine("converted and loaded; set HARTSYINFERENCE_RNNOISE_WEIGHTS to compare with the offline copy");
            return;
        }
        using SafeTensorsLoader ours = new();
        ours.Load(output);
        using SafeTensorsLoader theirs = new();
        theirs.Load(reference);
        Dictionary<string, Tensor> a = ours.GetAllTensors();
        Dictionary<string, Tensor> b = theirs.GetAllTensors();
        Assert.Equal(b.Keys.Order(), a.Keys.Order());
        foreach ((string name, Tensor tensor) in a)
        {
            Assert.Equal(b[name].Shape, tensor.Shape);
            Assert.Equal(b[name].DType, tensor.DType);
            Assert.True(tensor.AsSpan<byte>().SequenceEqual(b[name].AsSpan<byte>()), $"'{name}' differs");
        }
        log.WriteLine($"{a.Count} tensors byte-identical to {reference}");
        foreach (Tensor tensor in a.Values.Concat(b.Values)) tensor.Dispose();
    }

    private string WriteCheckpoint((string Name, long[] Shape)[] layout)
    {
        Dictionary<string, Tensor> tensors = new(StringComparer.Ordinal);
        for (int i = 0; i < layout.Length; i++)
        {
            Tensor tensor = new(new TensorShape(layout[i].Shape), DType.F32);
            Fill(i, layout[i].Shape).CopyTo(tensor.AsSpan<float>());
            tensors[layout[i].Name] = tensor;
        }
        string path = Path.Combine(_dir, $"{Guid.NewGuid():N}.pth");
        SafeTensorsWriter.Save(path, tensors);
        foreach (Tensor tensor in tensors.Values) tensor.Dispose();
        return path;
    }

    private static float[] Fill(int index, long[] shape)
    {
        long count = 1;
        foreach (long dim in shape) count *= dim;
        float[] values = new float[count];
        for (int i = 0; i < values.Length; i++) values[i] = (i % 97) * 1e-3f;
        values[^1] = index;
        return values;
    }

    private string WriteTarball(params (string Name, byte[] Data)[] entries)
    {
        string path = Path.Combine(_dir, $"{Guid.NewGuid():N}.tar.gz");
        using FileStream file = File.Create(path);
        using GZipStream gzip = new(file, CompressionLevel.Fastest);
        using TarWriter writer = new(gzip, TarEntryFormat.Pax);
        foreach ((string name, byte[] data) in entries)
            writer.WriteEntry(new PaxTarEntry(TarEntryType.RegularFile, name) { DataStream = new MemoryStream(data) });
        return path;
    }

    public void Dispose() => Directory.Delete(_dir, recursive: true);
}
