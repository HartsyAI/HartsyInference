using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Gguf;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>Lazily produced tensors converted several at a time still land in registration order with their own data, and a
/// producer failure is reported without leaking what was produced ahead.</summary>
public sealed class GgufWriterProduceAheadTests : IDisposable
{
    private readonly string _path = Path.Combine(Path.GetTempPath(), $"hartsy-ahead-{Guid.NewGuid():N}.gguf");

    public void Dispose()
    {
        try { File.Delete(_path); } catch (IOException) { }
    }

    private static Tensor Filled(int index)
    {
        Thread.Sleep((7 - index % 7) * 3);   // later tensors often finish first
        Tensor t = new(new TensorShape(4, 8), DType.F32);
        t.AsSpan<float>().Fill(index);
        return t;
    }

    [Theory]
    [InlineData(1)]
    [InlineData(4)]
    public void TensorsKeepTheirOrderAndData(int ahead)
    {
        using (GgufWriter writer = new(_path) { ProduceAhead = ahead })
        {
            writer.SetMetadata("general.architecture", "test");
            for (int i = 0; i < 12; i++)
            {
                int index = i;
                writer.AddTensor($"t{index:D2}", new TensorShape(4, 8), DType.F32, () => Filled(index));
            }
            writer.Flush();
        }
        using GgufLoader loader = new();
        loader.Load(_path);
        for (int i = 0; i < 12; i++)
        {
            using Tensor t = loader.GetTensor($"t{i:D2}");
            Assert.All(t.AsReadOnlySpan<float>().ToArray(), v => Assert.Equal(i, v));
        }
    }

    [Fact]
    public void AProducerFailureSurfaces()
    {
        using GgufWriter writer = new(_path) { ProduceAhead = 3 };
        writer.SetMetadata("general.architecture", "test");
        for (int i = 0; i < 6; i++)
        {
            int index = i;
            writer.AddTensor($"t{index}", new TensorShape(4, 8), DType.F32,
                () => index == 2 ? throw new InvalidDataException("bad tensor 2") : Filled(index));
        }
        Assert.Equal("bad tensor 2", Assert.Throws<InvalidDataException>(writer.Flush).Message);
    }
}
