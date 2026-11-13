using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.PyTorch;
using Xunit;

namespace HartsyInference.ModelAssets.Tests.PyTorch;

public sealed class PickleLazyReadTests : IDisposable
{
    // torch.save({"a.weight": tensor([[1, 2.5, -3], [0.1, 65519, 1e-8]]), "b": a transposed view of [4, 5, 6, 7] (stride (1, 2))}), written by torch 2.x.
    internal const string Checkpoint =
        "UEsDBAAACAgAAAAAAAAAAAAAAAAAAAAAAAAQABIAYXJjaGl2ZS9kYXRhLnBrbEZCDgBaWlpaWlpaWlpaWlpaWoACfXEAKFgIAAAAYS53ZWlnaHRxAWN0b3Jj" +
        "aC5fdXRpbHMKX3JlYnVpbGRfdGVuc29yX3YyCnECKChYBwAAAHN0b3JhZ2VxA2N0b3JjaApGbG9hdFN0b3JhZ2UKcQRYAQAAADBxBVgDAAAAY3B1cQZLBnRx" +
        "B1FLAEsCSwOGcQhLA0sBhnEJiWNjb2xsZWN0aW9ucwpPcmRlcmVkRGljdApxCilScQt0cQxScQ1YAQAAAGJxDmgCKChoA2gEWAEAAAAxcQ9oBksEdHEQUUsA" +
        "SwJLAoZxEUsBSwKGcRKJaAopUnETdHEUUnEVdS5QSwcIPIV5pesAAADrAAAAUEsDBAAACAgAAAAAAAAAAAAAAAAAAAAAAAARABYAYXJjaGl2ZS9ieXRlb3Jk" +
        "ZXJGQhIAWlpaWlpaWlpaWlpaWlpaWlpabGl0dGxlUEsHCIU94xkGAAAABgAAAFBLAwQAAAgIAAAAAAAAAAAAAAAAAAAAAAAADgA+AGFyY2hpdmUvZGF0YS8w" +
        "RkI6AFpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWloAAIA/AAAgQAAAQMDNzMw9AO9/R3fMKzJQSwcI" +
        "xLW7zRgAAAAYAAAAUEsDBAAACAgAAAAAAAAAAAAAAAAAAAAAAAAOACwAYXJjaGl2ZS9kYXRhLzFGQigAWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpa" +
        "WlpaWlpaWlpaWgAAgEAAAMBAAACgQAAA4EBQSwcIYcMlHRAAAAAQAAAAUEsDBAAACAgAAAAAAAAAAAAAAAAAAAAAAAAPADMAYXJjaGl2ZS92ZXJzaW9uRkIv" +
        "AFpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaMwpQSwcI0Z5nVQIAAAACAAAAUEsDBAAACAgAAAAAAAAAAAAAAAAAAAAA" +
        "AAAeADIAYXJjaGl2ZS8uZGF0YS9zZXJpYWxpemF0aW9uX2lkRkIuAFpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlowNDk3" +
        "MDcyOTkzNzk5MTM3NDYyMDAwMDkzOTc2MzYzOTYxMjIwMTA4UEsHCKTxa58oAAAAKAAAAFBLAQIAAAAACAgAAAAAAAA8hXml6wAAAOsAAAAQAAAAAAAAAAAA" +
        "AAAAAAAAAABhcmNoaXZlL2RhdGEucGtsUEsBAgAAAAAICAAAAAAAAIU94xkGAAAABgAAABEAAAAAAAAAAAAAAAAAOwEAAGFyY2hpdmUvYnl0ZW9yZGVyUEsB" +
        "AgAAAAAICAAAAAAAAMS1u80YAAAAGAAAAA4AAAAAAAAAAAAAAAAAlgEAAGFyY2hpdmUvZGF0YS8wUEsBAgAAAAAICAAAAAAAAGHDJR0QAAAAEAAAAA4AAAAA" +
        "AAAAAAAAAAAAKAIAAGFyY2hpdmUvZGF0YS8xUEsBAgAAAAAICAAAAAAAANGeZ1UCAAAAAgAAAA8AAAAAAAAAAAAAAAAAoAIAAGFyY2hpdmUvdmVyc2lvblBL" +
        "AQIAAAAACAgAAAAAAACk8WufKAAAACgAAAAeAAAAAAAAAAAAAAAAABIDAABhcmNoaXZlLy5kYXRhL3NlcmlhbGl6YXRpb25faWRQSwYGLAAAAAAAAAAeAy0A" +
        "AAAAAAAAAAAGAAAAAAAAAAYAAAAAAAAAfgEAAAAAAAC4AwAAAAAAAFBLBgcAAAAANgUAAAAAAAABAAAAUEsFBgAAAAAGAAYAfgEAALgDAAAAAA==";

    private readonly string _path = Path.Combine(Path.GetTempPath(), $"lazy-{Guid.NewGuid():N}.bin");

    public PickleLazyReadTests() => File.WriteAllBytes(_path, Convert.FromBase64String(Checkpoint));

    public void Dispose() => File.Delete(_path);

    [Fact]
    public void OpenLazy_ListsTensorsWithoutReadingThemAndReadTensorGivesEachOne()
    {
        using PytorchPickleLoader loader = new();
        loader.OpenLazy(_path);

        Assert.Equal(2, loader.Descriptors.Count);
        Assert.Equal(new TensorShape(2, 3), loader.Descriptors["a.weight"].Shape);
        Assert.Equal(DType.F32, loader.Descriptors["a.weight"].DType);
        Assert.Empty(loader.GetAllTensors());

        using Tensor a = loader.ReadTensor("a.weight");
        Assert.Equal(new float[] { 1f, 2.5f, -3f, 0.1f, 65519f, 1e-8f }, a.AsSpan<float>().ToArray());
        // A transposed view comes back in row-major order, as with Load.
        using Tensor b = loader.ReadTensor("b");
        Assert.Equal(new float[] { 4f, 5f, 6f, 7f }, b.AsSpan<float>().ToArray());
    }

    [Fact]
    public void ReadTensor_GivesTheSameValuesInAnyOrderAndForARepeatedTensor()
    {
        using PytorchPickleLoader loader = new();
        loader.OpenLazy(_path);
        PytorchTensorDescriptor a = loader.Descriptors["a.weight"], b = loader.Descriptors["b"];
        Assert.NotEqual(a.StorageKey, b.StorageKey);
        Assert.Equal(0, a.StorageByteOffset);

        foreach (string name in new[] { "b", "a.weight", "a.weight", "b", "a.weight" })
        {
            using Tensor t = loader.ReadTensor(name);
            Assert.Equal(name == "b" ? new float[] { 4f, 5f, 6f, 7f } : [1f, 2.5f, -3f, 0.1f, 65519f, 1e-8f], t.AsSpan<float>().ToArray());
        }
    }

    [Fact]
    public void ReadTensor_ConvertsToHalfOneTensorAtATime()
    {
        using PytorchPickleLoader loader = new();
        loader.OpenLazy(_path);

        using Tensor a = loader.ReadTensor("a.weight", DType.F16);

        Assert.Equal(DType.F16, a.DType);
        Assert.Equal(new TensorShape(2, 3), a.Shape);
        Half[] expected = [(Half)1f, (Half)2.5f, (Half)(-3f), (Half)0.1f, (Half)65519f, (Half)1e-8f];
        Assert.Equal(expected, a.AsSpan<Half>().ToArray());
    }

    [Fact]
    public void ReadTensor_RejectsAnUnknownNameAndALoaderThatWasNotOpenedLazily()
    {
        using PytorchPickleLoader lazy = new();
        lazy.OpenLazy(_path);
        Assert.Throws<KeyNotFoundException>(() => lazy.ReadTensor("missing"));
        Assert.Throws<InvalidOperationException>(() => lazy.OpenLazy(_path));
        Assert.Throws<InvalidOperationException>(() => lazy.Load(_path));

        using PytorchPickleLoader whole = new();
        whole.Load(_path);
        Assert.Throws<InvalidOperationException>(() => whole.ReadTensor("b"));
        Assert.Throws<InvalidOperationException>(() => whole.OpenLazy(_path));
    }

    [Fact]
    public void OpenLazy_RefusesAFileThatIsNotAZipCheckpoint()
    {
        File.WriteAllBytes(_path, [0x80, 0x02, 0x7d, 0x2e]);
        using PytorchPickleLoader loader = new();
        Assert.Throws<NotSupportedException>(() => loader.OpenLazy(_path));
    }
}
