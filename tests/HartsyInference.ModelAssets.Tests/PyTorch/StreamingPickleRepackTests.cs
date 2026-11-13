using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.PyTorch;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.ModelAssets.Tests.PyTorch;

public sealed class StreamingPickleRepackTests : IDisposable
{
    private readonly string _folder = Path.Combine(Path.GetTempPath(), $"repack-{Guid.NewGuid():N}");
    private readonly string _source;

    public StreamingPickleRepackTests()
    {
        Directory.CreateDirectory(_folder);
        _source = Path.Combine(_folder, "source.bin");
        File.WriteAllBytes(_source, Convert.FromBase64String(PickleLazyReadTests.Checkpoint));
    }

    public void Dispose() => Directory.Delete(_folder, recursive: true);

    [Fact]
    public void Repack_KeepsNamesOrderShapesDtypesAndValuesAndVerifyAgrees()
    {
        string output = Path.Combine(_folder, "out", "repacked.safetensors");

        PickleRepackSummary summary = StreamingPickleRepack.Repack(_source, output, metadata: new Dictionary<string, string> { ["format"] = "pt" });

        Assert.Equal(2, summary.TensorCount);
        Assert.Equal(40, summary.TensorBytes);
        Assert.True(summary.Streamed);
        using SafeTensorsLoader loader = new();
        loader.Load(output);
        Assert.Equal(["a.weight", "b"], loader.Descriptors.Keys);
        Assert.Equal(new TensorShape(2, 3), loader.Descriptors["a.weight"].Shape);
        Assert.Equal(DType.F32, loader.Descriptors["a.weight"].DType);
        Assert.Equal("pt", loader.Metadata!["format"]);
        // The transposed view is stored in row-major order, as the pickle loader returns it.
        Assert.Equal(new float[] { 4f, 5f, 6f, 7f }, loader.GetTensor("b").AsSpan<float>().ToArray());
        PickleRepackVerification verification = StreamingPickleRepack.Verify(_source, output);
        Assert.True(verification.Equal);
        Assert.Equal(2, verification.TensorCount);
        Assert.Empty(verification.Differences);
    }

    [Fact]
    public void Verify_ReportsAChangedValueAndANameAndShapeDifference()
    {
        string output = Path.Combine(_folder, "repacked.safetensors");
        StreamingPickleRepack.Repack(_source, output);
        byte[] bytes = File.ReadAllBytes(output);
        bytes[^1] ^= 0x01;
        string corrupted = Path.Combine(_folder, "corrupted.safetensors");
        File.WriteAllBytes(corrupted, bytes);

        PickleRepackVerification changed = StreamingPickleRepack.Verify(_source, corrupted);

        Assert.False(changed.Equal);
        Assert.Contains(changed.Differences, d => d.Contains("'b' differs in value", StringComparison.Ordinal));

        using Tensor other = new(new TensorShape(3), DType.F32);
        string renamed = Path.Combine(_folder, "renamed.safetensors");
        SafeTensorsWriter.Save(renamed, new Dictionary<string, Tensor> { ["a.weight"] = other, ["c"] = other });
        PickleRepackVerification different = StreamingPickleRepack.Verify(_source, renamed);

        Assert.False(different.Equal);
        Assert.Contains(different.Differences, d => d.Contains("names or their order differ", StringComparison.Ordinal));
        Assert.Contains(different.Differences, d => d.Contains("'a.weight' is F32 [2, 3]", StringComparison.Ordinal));
        Assert.Contains(different.Differences, d => d.Contains("'b' is missing", StringComparison.Ordinal));
    }

    [Fact]
    public void StreamWriter_WritesIntegerTensorsUnderTheirOwnDtype()
    {
        string output = Path.Combine(_folder, "ints.safetensors");
        using Tensor steps = new(new TensorShape(2), DType.I64);
        steps.AsSpan<long>()[0] = 7;
        steps.AsSpan<long>()[1] = -3;

        SafeTensorsStreamWriter.Write(output, [new SafeTensorsStreamWriter.Entry("steps", DType.I64, steps.Shape, () => (steps, false))]);

        using SafeTensorsLoader loader = new();
        loader.Load(output);
        Assert.Equal(DType.I64, loader.Descriptors["steps"].DType);
        Assert.Equal(new long[] { 7, -3 }, loader.GetTensor("steps").AsSpan<long>().ToArray());
    }
}
