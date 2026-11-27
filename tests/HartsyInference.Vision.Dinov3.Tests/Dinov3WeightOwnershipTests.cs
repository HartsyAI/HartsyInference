using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Vision.Dinov2;
using HartsyInference.Vision.Dinov3;
using Xunit;

namespace HartsyInference.Vision.Dinov3.Tests;

/// <summary>What <see cref="Dinov3VisionEncoder"/> borrows from a checkpoint, what it copies, and when the copies die.</summary>
public sealed class Dinov3WeightOwnershipTests : IDisposable
{
    private static readonly IBackend Backend = new CpuBackend();
    private readonly Dinov2Preset _preset = Dinov3Fixture.TinyPreset(imageSize: 64);
    private readonly Dictionary<string, Tensor> _full;
    private readonly Dictionary<string, Tensor> _half = new(StringComparer.Ordinal);

    public Dinov3WeightOwnershipTests()
    {
        _full = Dinov3Fixture.Tiny(_preset, seed: 51);
        foreach ((string key, Tensor value) in _full) _half[key] = value.CastTo(DType.F16);
    }

    public void Dispose()
    {
        Dinov3Fixture.Dispose(_full);
        Dinov3Fixture.Dispose(_half);
    }

    [Fact]
    public void AnF32Checkpoint_IsBorrowed_AndOutlivesTheEncoder()
    {
        Dinov3VisionEncoder encoder = new(_preset);
        encoder.LoadWeights(_full);
        List<Tensor> read = [.. encoder.EnumerateWeights()];

        encoder.Dispose();
        encoder.Dispose();

        Assert.Equal(_full.Count, read.Count);
        Assert.All(read, t => Assert.Contains(t, _full.Values));
        Assert.All(_full.Values, t => Assert.True(Alive(t)));
    }

    [Fact]
    public void A16BitCheckpoint_IsWidened_AndTheCopiesDieWithTheEncoder()
    {
        Dinov3VisionEncoder encoder = new(_preset);
        encoder.LoadWeights(_half);
        List<Tensor> copies = [.. encoder.EnumerateWeights()];

        Assert.Equal(_half.Count, copies.Count);
        Assert.All(copies, t => Assert.Equal(DType.F32, t.DType));
        Assert.All(copies, t => Assert.DoesNotContain(t, _half.Values));
        _ = Tokens(encoder);

        encoder.Dispose();

        Assert.All(copies, t => Assert.False(Alive(t)));
        Assert.All(_half.Values, t => Assert.True(Alive(t)));
        Assert.Throws<ObjectDisposedException>(() => Tokens(encoder));
        Assert.Throws<ObjectDisposedException>(() => encoder.LoadWeights(_half));
    }

    [Fact]
    public void AReload_FreesTheFirstCopies_AndTheEncoderStillRuns()
    {
        using Dinov3VisionEncoder encoder = new(_preset);
        encoder.LoadWeights(_half);
        List<Tensor> first = [.. encoder.EnumerateWeights()];
        float[] before = Tokens(encoder);

        encoder.LoadWeights(_half);

        Assert.All(first, t => Assert.False(Alive(t)));
        Assert.Equal(before, Tokens(encoder));
    }

    [Fact]
    public void AFailedLoad_FreesItsCopies_AndLeavesTheEncoderWithoutWeights()
    {
        using Dinov3VisionEncoder encoder = new(_preset);
        encoder.LoadWeights(_half);
        List<Tensor> first = [.. encoder.EnumerateWeights()];
        Dictionary<string, Tensor> broken = new(_half);
        broken.Remove("norm.bias");

        Assert.Throws<KeyNotFoundException>(() => encoder.LoadWeights(broken));

        Assert.All(first, t => Assert.False(Alive(t)));
        Assert.Empty(encoder.EnumerateWeights());
        Assert.Throws<InvalidOperationException>(() => Tokens(encoder));
    }

    [Fact]
    public void AQuantizedTensor_IsRefused_InsteadOfLosingItsScales()
    {
        const string Key = "layer.0.attention.q_proj.weight";
        using Dinov3VisionEncoder encoder = new(_preset);
        using Tensor fp8 = new(new TensorShape(128, 128), DType.F8E4M3) { QuantInfo = new QuantWeightInfo { Format = "fp8_block" } };

        NotSupportedException error = Assert.Throws<NotSupportedException>(
            () => encoder.LoadWeights(new Dictionary<string, Tensor>(_full) { [Key] = fp8 }));

        Assert.Contains($"'{Key}'", error.Message);
    }

    private static float[] Tokens(Dinov3VisionEncoder encoder)
    {
        using Tensor pixels = new(new TensorShape(1, 3, 64, 64), DType.F32);
        Span<float> values = pixels.AsSpan<float>();
        for (int i = 0; i < values.Length; i++) values[i] = MathF.Sin(i);
        using Tensor tokens = encoder.Encode(Backend, pixels);
        return tokens.AsReadOnlySpan<float>().ToArray();
    }

    private static bool Alive(Tensor t)
    {
        try
        {
            _ = t.AsReadOnlySpan<byte>().Length;
            return true;
        }
        catch (ObjectDisposedException)
        {
            return false;
        }
    }
}
