using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Core.Tests.Memory;

/// <summary>Named checkpoint reads: what is refused, and who owns a widened copy.</summary>
public sealed class CheckpointTensorsTests : IDisposable
{
    private readonly Tensor _f32 = new(new TensorShape(2, 3), DType.F32);
    private readonly Tensor _f16 = new(new TensorShape(4), DType.F16);
    private readonly Tensor _fp8 = new(new TensorShape(2, 2), DType.F8E4M3);
    private readonly Dictionary<string, Tensor> _weights;

    public CheckpointTensorsTests()
    {
        foreach (ref Half v in _f16.AsSpan<Half>()) v = (Half)1.5f;
        _fp8.QuantInfo = new QuantWeightInfo { Format = "fp8_block" };
        _weights = new Dictionary<string, Tensor> { ["a"] = _f32, ["b"] = _f16, ["q"] = _fp8 };
    }

    public void Dispose()
    {
        foreach (Tensor t in _weights.Values) t.Dispose();
    }

    [Fact]
    public void Require_ReturnsTheBorrowedTensor_WithOrWithoutAShape()
    {
        Assert.Same(_f32, CheckpointTensors.Require(_weights, "Toy", "a"));
        Assert.Same(_f32, CheckpointTensors.Require(_weights, "Toy", "a", 2, 3));
    }

    [Fact]
    public void AMissingTensor_NamesTheFamilyAndTheKey()
    {
        KeyNotFoundException error = Assert.Throws<KeyNotFoundException>(() => CheckpointTensors.Require(_weights, "Toy", "nope", 1));
        Assert.Contains("Toy", error.Message);
        Assert.Contains("'nope'", error.Message);
    }

    [Theory]
    [InlineData(new long[] { 3, 2 })]
    [InlineData(new long[] { 2, 3, 1 })]
    [InlineData(new long[] { 6 })]
    public void AnotherShape_IsRefusedEvenWithTheSameElementCount(long[] shape)
    {
        InvalidDataException error = Assert.Throws<InvalidDataException>(() => CheckpointTensors.Require(_weights, "Toy", "a", shape));
        Assert.Contains("'a'", error.Message);
        Assert.Contains(string.Join(", ", shape), error.Message);
    }

    [Fact]
    public void RequireF32_BorrowsAnF32Tensor_AndOwnsOnlyAWidenedCopy()
    {
        List<Tensor> owned = [];

        Assert.Same(_f32, CheckpointTensors.RequireF32(_weights, "Toy", "a", owned, 2, 3));
        Assert.Empty(owned);

        Tensor widened = CheckpointTensors.RequireF32(_weights, "Toy", "b", owned, 4);
        Assert.Same(widened, Assert.Single(owned));
        Assert.Equal(DType.F32, widened.DType);
        Assert.All(widened.AsReadOnlySpan<float>().ToArray(), v => Assert.Equal(1.5f, v));
        widened.Dispose();
        Assert.Equal((Half)1.5f, _f16.AsReadOnlySpan<Half>()[0]);
    }

    [Fact]
    public void RequireF32_RefusesAQuantizedTensor_InsteadOfDroppingItsScales()
    {
        List<Tensor> owned = [];

        NotSupportedException error = Assert.Throws<NotSupportedException>(() => CheckpointTensors.RequireF32(_weights, "Toy", "q", owned));

        Assert.Contains("fp8_block", error.Message);
        Assert.Empty(owned);
    }

    [Fact]
    public void OptionalF32_IsNullOnlyWhenAbsent_AndStillChecksTheShape()
    {
        List<Tensor> owned = [];

        Assert.Null(CheckpointTensors.OptionalF32(_weights, "Toy", "nope", owned, 4));
        Assert.NotNull(CheckpointTensors.OptionalF32(_weights, "Toy", "b", owned, 4));
        Assert.Throws<InvalidDataException>(() => CheckpointTensors.OptionalF32(_weights, "Toy", "b", owned, 5));
        foreach (Tensor t in owned) t.Dispose();
    }
}
