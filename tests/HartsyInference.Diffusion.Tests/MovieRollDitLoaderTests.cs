using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The released MovieRoll DiT file (<c>yunyangge/MovieRoll</c> <c>distilled/t2v/model.safetensors</c>, 33,291,800,480 bytes, sha256
/// 410694c3e24f6c5d3a53485fbebdfe8e9cc18165f103623eee21f6ff3b91b169), read for its header and for one tensor. The file holds FP32 weights.
/// <c>MOVIEROLL_DIT_PATH</c> names the file. Without it the tests return early, and with <c>HARTSY_REQUIRE_REAL_WEIGHTS=1</c> they fail instead.
/// The first test reads the header only: every tensor is FP32, and the byte lengths account for the whole file apart from a header under 8 MB.
/// The second casts the smallest tensor to F16 and back, and checks the round trip against the FP32 values within the F16 rounding of one part
/// in 1024.</summary>
[Trait("Category", "Integration")]
public sealed class MovieRollDitLoaderTests(ITestOutputHelper output)
{
    private const long MaxHeaderBytes = 8L << 20;
    private const float F16RelativeBound = 1f / 1024f;

    private static string? DitPath()
    {
        string? path = Environment.GetEnvironmentVariable("MOVIEROLL_DIT_PATH");
        if (!string.IsNullOrEmpty(path) && File.Exists(path))
        {
            return path;
        }
        Assert.False(Environment.GetEnvironmentVariable("HARTSY_REQUIRE_REAL_WEIGHTS") == "1",
            "MOVIEROLL_DIT_PATH does not name the released MovieRoll DiT file.");
        return null;
    }

    [Fact]
    public void Header_IsFp32AndAccountsForTheWholeFile()
    {
        if (DitPath() is not string path)
        {
            return;
        }
        using SafeTensorsLoader loader = new();
        loader.Load(path);

        long fileLength = new FileInfo(path).Length;
        long dataBytes = 0;
        foreach (SafeTensorDescriptor descriptor in loader.Descriptors.Values)
        {
            Assert.Equal(DType.F32, descriptor.DType);
            dataBytes += descriptor.ByteLength;
        }
        long headerBytes = fileLength - dataBytes;
        output.WriteLine($"tensors {loader.Descriptors.Count}, data bytes {dataBytes}, file bytes {fileLength}, header bytes {headerBytes}");
        Assert.InRange(headerBytes, 8L, MaxHeaderBytes);
    }

    [Fact]
    public void SmallestTensor_CastsToF16AndBackWithinRounding()
    {
        if (DitPath() is not string path)
        {
            return;
        }
        using SafeTensorsLoader loader = new();
        loader.Load(path);

        SafeTensorDescriptor smallest = loader.Descriptors.Values.OrderBy(d => d.ByteLength).First();
        using Tensor source = loader.GetTensor(smallest.Name);
        using Tensor half = source.CastTo(DType.F16);
        using Tensor roundTrip = half.CastTo(DType.F32);

        ReadOnlySpan<float> expected = source.AsReadOnlySpan<float>();
        ReadOnlySpan<float> actual = roundTrip.AsReadOnlySpan<float>();
        Assert.Equal(expected.Length, actual.Length);
        float worst = 0f;
        for (int i = 0; i < expected.Length; i++)
        {
            float scale = Math.Max(Math.Abs(expected[i]), 1e-6f);
            worst = Math.Max(worst, Math.Abs(expected[i] - actual[i]) / scale);
        }
        output.WriteLine($"{smallest.Name} shape {smallest.Shape}: {expected.Length} values, worst relative round trip {worst:E3}");
        Assert.True(worst <= F16RelativeBound, $"{smallest.Name}: relative round trip {worst:E3} exceeds {F16RelativeBound:E3}.");
    }
}
