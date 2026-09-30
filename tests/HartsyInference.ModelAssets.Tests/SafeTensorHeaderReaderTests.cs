using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

public sealed class SafeTensorHeaderReaderTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), $"st_header_{Guid.NewGuid():N}");

    public SafeTensorHeaderReaderTests() => Directory.CreateDirectory(_dir);

    public void Dispose() => Directory.Delete(_dir, recursive: true);

    private string PathOf(string name) => Path.Combine(_dir, name);

    [Fact]
    public void Read_ValidFile_ReturnsAbsoluteOffsets()
    {
        string path = PathOf("ok.safetensors");
        ShardTestFiles.WriteShard(path, ShardTestFiles.F32("a", 1, 2), ShardTestFiles.F32("b", 3));

        SafeTensorHeader header = SafeTensorHeaderReader.Read(path, SafeTensorHeaderReader.DefaultMaxHeaderBytes);

        Assert.Equal(2, header.Tensors.Count);
        Assert.Equal(header.DataStart, header.Tensors["a"].DataOffset);
        Assert.Equal(header.DataStart + 8, header.Tensors["b"].DataOffset);
        Assert.Equal(4, header.Tensors["b"].ByteLength);
        Assert.Equal(new FileInfo(path).Length, header.FileLength);
    }

    [Fact]
    public void Read_ExposesMetadata()
    {
        string path = PathOf("meta.safetensors");
        ShardTestFiles.WriteRaw(path,
            """{"__metadata__":{"format":"pt"},"t":{"dtype":"U8","shape":[2],"data_offsets":[0,2]}}""", 2);

        SafeTensorHeader header = SafeTensorHeaderReader.Read(path, SafeTensorHeaderReader.DefaultMaxHeaderBytes);

        Assert.Equal("pt", header.Metadata!["format"]);
        Assert.Single(header.Tensors);
    }

    [Fact]
    public void Read_OversizedHeader_IsRejectedBeforeAllocating()
    {
        string path = PathOf("big.safetensors");
        File.WriteAllBytes(path, [.. BitConverter.GetBytes((ulong)1 << 40), .. new byte[64]]);

        HartsyInferenceException error = Assert.Throws<HartsyInferenceException>(
            () => SafeTensorHeaderReader.Read(path, 1024));

        Assert.Contains("limit", error.Message);
    }

    [Fact]
    public void Read_HeaderLongerThanFile_IsRejected()
    {
        string path = PathOf("short.safetensors");
        File.WriteAllBytes(path, [.. BitConverter.GetBytes((ulong)500), .. new byte[20]]);

        HartsyInferenceException error = Assert.Throws<HartsyInferenceException>(
            () => SafeTensorHeaderReader.Read(path, SafeTensorHeaderReader.DefaultMaxHeaderBytes));

        Assert.Contains("Invalid safetensors header length", error.Message);
    }

    [Fact]
    public void Read_TensorPastEndOfFile_ReportsTruncation()
    {
        string path = PathOf("trunc.safetensors");
        ShardTestFiles.WriteRaw(path, """{"t":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}}""", 8);

        HartsyInferenceException error = Assert.Throws<HartsyInferenceException>(
            () => SafeTensorHeaderReader.Read(path, SafeTensorHeaderReader.DefaultMaxHeaderBytes));

        Assert.Contains("truncated", error.Message);
    }

    [Fact]
    public void Read_OverlappingTensors_AreRejected()
    {
        string path = PathOf("overlap.safetensors");
        ShardTestFiles.WriteRaw(path,
            """{"a":{"dtype":"U8","shape":[8],"data_offsets":[0,8]},"b":{"dtype":"U8","shape":[8],"data_offsets":[4,12]}}""", 12);

        HartsyInferenceException error = Assert.Throws<HartsyInferenceException>(
            () => SafeTensorHeaderReader.Read(path, SafeTensorHeaderReader.DefaultMaxHeaderBytes));

        Assert.Contains("overlap", error.Message);
    }

    [Fact]
    public void Read_TensorsListedOutOfOrder_AreAccepted()
    {
        string path = PathOf("unsorted.safetensors");
        ShardTestFiles.WriteRaw(path,
            """{"b":{"dtype":"U8","shape":[4],"data_offsets":[4,8]},"a":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}}""", 8);

        SafeTensorHeader header = SafeTensorHeaderReader.Read(path, SafeTensorHeaderReader.DefaultMaxHeaderBytes);

        Assert.Equal(header.DataStart, header.Tensors["a"].DataOffset);
    }

    [Fact]
    public void Read_ByteLengthDisagreeingWithShape_IsRejected()
    {
        string path = PathOf("badlen.safetensors");
        ShardTestFiles.WriteRaw(path, """{"t":{"dtype":"F32","shape":[4],"data_offsets":[0,12]}}""", 12);

        HartsyInferenceException error = Assert.Throws<HartsyInferenceException>(
            () => SafeTensorHeaderReader.Read(path, SafeTensorHeaderReader.DefaultMaxHeaderBytes));

        Assert.Contains("needs 16", error.Message);
    }

    [Fact]
    public void Read_DataLessStub_IsAcceptedWhenByteLengthCheckIsOff()
    {
        string path = PathOf("stub.safetensors");
        ShardTestFiles.WriteRaw(path, """{"t":{"dtype":"F32","shape":[5376,96],"data_offsets":[0,0]}}""", 0);

        SafeTensorHeader header = SafeTensorHeaderReader.Read(
            path, SafeTensorHeaderReader.DefaultMaxHeaderBytes, verifyByteLength: false);

        Assert.Equal(0, header.Tensors["t"].ByteLength);
        Assert.Throws<HartsyInferenceException>(() => SafeTensorHeaderReader.Read(path));
        using SafeTensorsLoader loader = new SafeTensorsLoader();
        loader.Load(path);
        Assert.Single(loader.Descriptors);
    }

    [Fact]
    public void Read_DuplicateTensorName_IsRejected()
    {
        string path = PathOf("dup.safetensors");
        ShardTestFiles.WriteRaw(path,
            """{"t":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"t":{"dtype":"U8","shape":[1],"data_offsets":[1,2]}}""", 2);

        HartsyInferenceException error = Assert.Throws<HartsyInferenceException>(
            () => SafeTensorHeaderReader.Read(path, SafeTensorHeaderReader.DefaultMaxHeaderBytes));

        Assert.Contains("twice", error.Message);
    }

    [Theory]
    [InlineData("""{"t":{"dtype":"QQ","shape":[1],"data_offsets":[0,1]}}""", "Unsupported safetensors dtype")]
    [InlineData("""{"t":{"dtype":"U8","shape":[-1],"data_offsets":[0,1]}}""", "dimension")]
    [InlineData("""{"t":{"dtype":"U8","shape":[1],"data_offsets":[1,0]}}""", "malformed data_offsets")]
    [InlineData("""{"t":{"dtype":"U8","shape":[1,1,1,1,1,1,1],"data_offsets":[0,1]}}""", "rank 7")]
    [InlineData("""{"t":{"dtype":"U8","shape":[4611686018427387904,4],"data_offsets":[0,1]}}""", "overflows")]
    [InlineData("""[1,2]""", "not a JSON object")]
    [InlineData("""{"t":{"dtype":"U8","shape":[1]}}""", "data_offsets")]
    public void Read_MalformedTensor_IsRejected(string headerJson, string expected)
    {
        string path = PathOf("malformed.safetensors");
        ShardTestFiles.WriteRaw(path, headerJson, 8);

        HartsyInferenceException error = Assert.Throws<HartsyInferenceException>(
            () => SafeTensorHeaderReader.Read(path, SafeTensorHeaderReader.DefaultMaxHeaderBytes));

        Assert.Contains(expected, error.Message);
    }

    [Theory]
    [InlineData("F8_E8M0")]
    [InlineData("U16")]
    [InlineData("U32")]
    [InlineData("U64")]
    [InlineData("F8_E4M3FNUZ")]
    [InlineData("F8_E5M2FNUZ")]
    public void Read_NewDtypes_Parse(string name)
    {
        Assert.True(SafeTensorDTypes.TryParse(name, out DType dtype));
        Assert.Equal(name, dtype.Name);
        string path = PathOf("dtype.safetensors");
        long bytes = dtype.SizeInBytes * 3;
        ShardTestFiles.WriteRaw(path,
            $$$"""{"t":{"dtype":"{{{name}}}","shape":[3],"data_offsets":[0,{{{bytes}}}]}}""", bytes);

        SafeTensorHeader header = SafeTensorHeaderReader.Read(path, SafeTensorHeaderReader.DefaultMaxHeaderBytes);

        Assert.Equal(dtype, header.Tensors["t"].DType);
    }

    [Fact]
    public void Loader_SharesTheValidator()
    {
        string path = PathOf("shared.safetensors");
        ShardTestFiles.WriteRaw(path,
            """{"a":{"dtype":"U8","shape":[8],"data_offsets":[0,8]},"b":{"dtype":"U8","shape":[8],"data_offsets":[4,12]}}""", 12);

        using SafeTensorsLoader loader = new SafeTensorsLoader();

        Assert.Throws<HartsyInferenceException>(() => loader.Load(path));
    }

    [Fact]
    public void Loader_RefusesToMaterialiseFnuz()
    {
        string path = PathOf("fnuz.safetensors");
        ShardTestFiles.WriteRaw(path, """{"t":{"dtype":"F8_E4M3FNUZ","shape":[2],"data_offsets":[0,2]}}""", 2);

        using SafeTensorsLoader loader = new SafeTensorsLoader();
        loader.Load(path);
        UnsupportedModelException error = Assert.Throws<UnsupportedModelException>(() => loader.GetTensor("t"));

        Assert.Contains("FNUZ", error.Message);
    }

    [Fact]
    public void Loader_LoadsE8M0AsOneByteElements()
    {
        string path = PathOf("e8m0.safetensors");
        ShardTestFiles.WriteRaw(path, """{"t":{"dtype":"F8_E8M0","shape":[4],"data_offsets":[0,4]}}""", 4);

        using SafeTensorsLoader loader = new SafeTensorsLoader();
        loader.Load(path);
        Tensor tensor = loader.GetTensor("t");

        Assert.Equal(DType.F8E8M0, tensor.DType);
        Assert.Equal(4, tensor.Shape.ElementCount);
    }
}
