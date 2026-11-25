using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Dcp;
using HartsyInference.ModelAssets.Duo;
using Xunit;

namespace HartsyInference.ModelAssets.Tests.Duo;

public sealed class DcpCheckpointTests : IDisposable
{
    private readonly string _folder = Path.Combine(Path.GetTempPath(), $"dcp-{Guid.NewGuid():N}");

    public void Dispose()
    {
        if (Directory.Exists(_folder)) Directory.Delete(_folder, recursive: true);
    }

    private static Dictionary<string, DcpFixture.Tensor1> Sample() => new()
    {
        ["models.generator.proj_in.lora_A.default.weight"] = DcpFixture.F32([2, 3], 1f, 2.5f, -3f, 0.1f, 65519f, 1e-8f),
        ["models.generator.proj_in.lora_B.default.weight"] = DcpFixture.Bf16([3, 1], 1f, -2f, 0.5f),
        ["models.generator.role_embeddings.weight"] = DcpFixture.F32([2, 2], 0f, 0f, 0f, 0f),
    };

    [Fact]
    public void Metadata_ListsEveryTensorWithItsDTypeShapeAndRange()
    {
        DcpFixture.Write(_folder, Sample());
        using DcpCheckpoint checkpoint = DcpCheckpoint.Open(_folder);

        Assert.Equal(3, checkpoint.Entries.Count);
        DcpTensorEntry a = checkpoint.Entries["models.generator.proj_in.lora_A.default.weight"];
        Assert.Equal(DType.F32, a.DType);
        Assert.Equal(new long[] { 2, 3 }, a.Shape);
        Assert.Equal(0, a.Offset);
        Assert.Equal("__0_0.distcp", a.RelativePath);
        DcpTensorEntry b = checkpoint.Entries["models.generator.proj_in.lora_B.default.weight"];
        Assert.Equal(DType.BF16, b.DType);
        Assert.Equal(a.Length, b.Offset);
        Assert.Equal(new Dictionary<string, int> { ["models.generator."] = 3 }, checkpoint.FamilyPrefixes());
    }

    [Fact]
    public void ReadTensor_SlicesEachStandaloneZipAndConvertsOnRequest()
    {
        DcpFixture.Write(_folder, Sample());
        using DcpCheckpoint checkpoint = DcpCheckpoint.Open(_folder);

        foreach (string key in new[] { "models.generator.proj_in.lora_B.default.weight", "models.generator.proj_in.lora_A.default.weight" })
        {
            using Tensor tensor = checkpoint.ReadTensor(key);
            Assert.Equal(checkpoint.Entries[key].DType, tensor.DType);
        }

        using Tensor a = checkpoint.ReadTensor("models.generator.proj_in.lora_A.default.weight");
        Assert.Equal(new TensorShape(2, 3), a.Shape);
        Assert.Equal(new float[] { 1f, 2.5f, -3f, 0.1f, 65519f, 1e-8f }, a.AsSpan<float>().ToArray());
        using Tensor b = checkpoint.ReadTensor("models.generator.proj_in.lora_B.default.weight", DType.F32);
        Assert.Equal(new float[] { 1f, -2f, 0.5f }, b.AsSpan<float>().ToArray());
        Assert.Throws<KeyNotFoundException>(() => checkpoint.ReadTensor("missing"));
    }

    [Fact]
    public void Open_RefusesAFolderWithoutMetadata()
    {
        Directory.CreateDirectory(_folder);
        Assert.Throws<FileNotFoundException>(() => DcpCheckpoint.Open(_folder));
    }

    [Fact]
    public void Metadata_RefusesATensorStoredInMoreThanItsWholeShape()
    {
        string key = "models.generator.proj_in.lora_A.default.weight";
        DcpFixture.Write(_folder, Sample(), chunkOverride: key);
        NotSupportedException error = Assert.Throws<NotSupportedException>(() => DcpCheckpoint.Open(_folder));
        Assert.Contains(key, error.Message);
    }

    [Fact]
    public void ReadTensor_RefusesARecordThatDisagreesWithTheMetadataOrRunsPastTheFile()
    {
        DcpFixture.Write(_folder, Sample());
        string path = Path.Combine(_folder, "__0_0.distcp");
        using (DcpCheckpoint checkpoint = DcpCheckpoint.Open(_folder))
        {
            // Truncating the data file leaves the metadata promising bytes that are gone.
            File.WriteAllBytes(path, File.ReadAllBytes(path)[..32]);
        }
        using DcpCheckpoint truncated = DcpCheckpoint.Open(_folder);
        Assert.Throws<EndOfStreamException>(() => truncated.ReadTensor("models.generator.role_embeddings.weight"));
    }

    [Fact]
    public void Parse_RejectsAGlobalThatIsNotCheckpointMetadata()
    {
        PickleBuilder p = new();
        p.Global("os", "system").Tuple(b => b.Str("calc")).Op((byte)'R');
        InvalidDataException error = Assert.Throws<InvalidDataException>(() => DcpMetadata.Parse(p.Finish()));
        Assert.Contains("os.system", error.Message);
    }

    [Theory]
    [InlineData("builtins", "eval")]
    [InlineData("subprocess", "Popen")]
    [InlineData("torch", "load")]
    [InlineData("torch.distributed", "rpc")]
    public void IsAllowedGlobal_RejectsCallablesOutsideTheMetadataVocabulary(string module, string name)
        => Assert.False(DcpMetadataUnpickler.IsAllowedGlobal(module, name));

    [Theory]
    [InlineData("torch", "Size")]
    [InlineData("torch", "float32")]
    [InlineData("torch.distributed.checkpoint.metadata", "MetadataIndex")]
    [InlineData("torch.distributed.checkpoint.filesystem", "_StorageInfo")]
    [InlineData("collections", "OrderedDict")]
    public void IsAllowedGlobal_AcceptsTheClassesAMetadataFileHolds(string module, string name)
        => Assert.True(DcpMetadataUnpickler.IsAllowedGlobal(module, name));

    [Fact]
    public void Parse_RefusesAStorageFileOutsideTheFolder()
    {
        DcpFixture.Write(_folder, Sample(), relativePath: "../escape.distcp");
        InvalidDataException error = Assert.Throws<InvalidDataException>(() => DcpCheckpoint.Open(_folder));
        Assert.Contains("outside", error.Message);
    }

    [Fact]
    public void Parse_ReportsATruncatedOrCorruptedFileAsInvalidDataNotAnInternalError()
    {
        DcpFixture.Write(_folder, Sample());
        byte[] metadata = File.ReadAllBytes(Path.Combine(_folder, ".metadata"));
        foreach (int cut in new[] { 5, 40, metadata.Length / 2, metadata.Length - 1 })
            Assert.Throws<InvalidDataException>(() => DcpMetadata.Parse(metadata[..cut]));

        byte[] corrupted = [.. metadata];
        for (int i = 60; i < corrupted.Length; i += 7) corrupted[i] ^= 0x5a;
        Assert.ThrowsAny<Exception>(() => DcpMetadata.Parse(corrupted));
    }

    [Fact]
    public void Parse_ReadsTensorPropertiesFromTheTupleStateTorchWrites()
    {
        DcpFixture.Write(_folder, Sample());
        DcpMetadata parsed = DcpMetadata.Parse(File.ReadAllBytes(Path.Combine(_folder, ".metadata")));
        Assert.Equal(DType.BF16, parsed.Tensors["models.generator.proj_in.lora_B.default.weight"].DType);
    }

    [Fact]
    public void Parse_RefusesAFileThatIsNotMetadata()
    {
        PickleBuilder p = new();
        p.Str("not metadata");
        Assert.Throws<InvalidDataException>(() => DcpMetadata.Parse(p.Finish()));
    }

    [DuoRealWeightsFact("HARTSY_DUO_DCP_DIR")]
    [Trait("Category", "Integration")]
    public void RealCheckpoint_ListsTheDocumentedTensorTable()
    {
        using DcpCheckpoint checkpoint = DcpCheckpoint.Open(Environment.GetEnvironmentVariable("HARTSY_DUO_DCP_DIR")!);
        Assert.Equal(635, checkpoint.Entries.Count);
        Assert.Equal(695_937_536L, checkpoint.Entries.Values.Sum(e => e.Shape.Aggregate(1L, (x, y) => x * y)));
        Assert.All(checkpoint.Entries.Values, e => Assert.Equal("__0_0.distcp", e.RelativePath));
    }

    [DuoRealWeightsFact("HARTSY_DUO_DCP_DIR")]
    [Trait("Category", "Integration")]
    public void RealCheckpoint_EveryLoraModuleMapsOntoABackboneWeight()
    {
        using DuoLoraSource lora = DuoLoraSource.FromDcp(Environment.GetEnvironmentVariable("HARTSY_DUO_DCP_DIR")!);
        Assert.Equal(317, lora.Modules.Count);
        Assert.All(lora.Modules, module => Assert.True(DuoLoraKeyMap.TryMap(module, out _), $"'{module}' has no backbone target"));
        Assert.All(lora.Describe(), t => Assert.Equal(DType.F32, t.DType));
    }
}
