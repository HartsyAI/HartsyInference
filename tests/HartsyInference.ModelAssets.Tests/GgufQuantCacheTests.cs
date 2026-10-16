using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Gguf;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>A quant cache is reused only when it is the cache the caller would write: same metadata, same tensors in the same
/// dtypes. Anything else is rebuilt, since quantization keeps the shapes that a later check could have caught.</summary>
public sealed unsafe class GgufQuantCacheTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), $"hartsy-quant-cache-{Guid.NewGuid():N}");
    private readonly Tensor _weight = Ramp(4, 64);

    public GgufQuantCacheTests() => Directory.CreateDirectory(_dir);

    [Fact]
    public void AMatchingCache_IsMappedAsIs()
    {
        string path = Path.Combine(_dir, "cache.gguf");
        Open(path, DType.Q8_0, version: 1).Dispose();
        DateTime written = File.GetLastWriteTimeUtc(path);
        using GgufLoader again = Open(path, DType.Q8_0, version: 1);
        Assert.Equal(written, File.GetLastWriteTimeUtc(path));
        Assert.Equal(DType.Q8_0, again.Descriptors["w"].DType);
    }

    [Fact]
    public void ACacheFromAnotherVersion_IsRebuilt()
    {
        string path = Path.Combine(_dir, "cache.gguf");
        Open(path, DType.Q8_0, version: 1).Dispose();
        using GgufLoader rebuilt = Open(path, DType.Q8_0, version: 2);
        Assert.True(rebuilt.Metadata.TryGetValue("test.version", out object? version));
        Assert.Equal("2", Convert.ToString(version, System.Globalization.CultureInfo.InvariantCulture));
    }

    [Fact]
    public void ACacheWithOtherDtypes_IsRebuilt()
    {
        string path = Path.Combine(_dir, "cache.gguf");
        Open(path, DType.Q8_0, version: 1).Dispose();
        using GgufLoader rebuilt = Open(path, DType.F16, version: 1);
        Assert.Equal(DType.F16, rebuilt.Descriptors["w"].DType);
    }

    [Fact]
    public void Mismatch_NamesWhatDiffers()
    {
        string path = Path.Combine(_dir, "cache.gguf");
        using GgufLoader cache = Open(path, DType.Q8_0, version: 1);
        Assert.Null(GgufQuantCache.Mismatch(cache, Entries(DType.Q8_0), Metadata(1)));
        Assert.Contains("test.version", GgufQuantCache.Mismatch(cache, Entries(DType.Q8_0), Metadata(3)), StringComparison.Ordinal);
        Assert.Contains("'w'", GgufQuantCache.Mismatch(cache, Entries(DType.Q4_0), Metadata(1)), StringComparison.Ordinal);
        Assert.Contains("'extra' is missing", GgufQuantCache.Mismatch(cache,
            [.. Entries(DType.Q8_0), new GgufCacheEntry("extra", _weight.Shape, DType.F32, () => _weight.Reshape(_weight.Shape))],
            Metadata(1)), StringComparison.Ordinal);
    }

    [Fact]
    public void APartialCacheFromADeadWriter_IsRemoved_AndAWritersOwnTemporaryNeverLingers()
    {
        string path = Path.Combine(_dir, "cache.gguf");
        // Process ids are positive ints well below int.MaxValue on every platform the engine runs on.
        string abandoned = $"{path}.{int.MaxValue}.{Guid.NewGuid():N}.tmp";
        File.WriteAllBytes(abandoned, [1, 2, 3]);
        Open(path, DType.Q8_0, version: 1).Dispose();
        Assert.False(File.Exists(abandoned));
        Assert.Equal([Path.GetFileName(path)], Directory.EnumerateFiles(_dir).Select(Path.GetFileName));
    }

    public void Dispose()
    {
        _weight.Dispose();
        Directory.Delete(_dir, recursive: true);
    }

    private GgufLoader Open(string path, DType dtype, int version) =>
        GgufQuantCache.Open(path, () => Entries(dtype), "test", "[Test]", Metadata(version));

    private IEnumerable<GgufCacheEntry> Entries(DType dtype) =>
        [new GgufCacheEntry("w", _weight.Shape, dtype, () => GgufQuantCache.Convert(_weight, dtype))];

    private static Dictionary<string, object> Metadata(int version) => new() { ["test.version"] = version };

    private static Tensor Ramp(int rows, int cols)
    {
        Tensor t = new(new TensorShape(rows, cols), DType.F32);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++) p[i] = (i % 17 - 8) * 0.125f;
        return t;
    }
}
