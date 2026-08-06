using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.PyTorch;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Verifies the offline repack (<c>tools/repack</c>) loses nothing: the single
/// <c>kokoro-82m.safetensors</c> must carry the exact same keys, shapes, dtypes, and bytes as the
/// original <c>kokoro-v1_0.pth</c> after the runtime flatten + inner-<c>module.</c> strip. Point
/// <c>HARTSYINFERENCE_KOKORO_REPACK_TEST</c> at a directory holding both files to run; self-skips
/// otherwise so CI stays self-contained.</summary>
public sealed class KokoroRepackParityTests
{
    private static string? Dir => Environment.GetEnvironmentVariable("HARTSYINFERENCE_KOKORO_REPACK_TEST");

    [Fact]
    public unsafe void RepackedSafetensors_BitwiseMatchesOriginalPth()
    {
        string? dir = Dir;
        if (dir is null) { Assert.True(true, "set HARTSYINFERENCE_KOKORO_REPACK_TEST to run"); return; }
        string pthPath = Path.Combine(dir, "kokoro-v1_0.pth");
        string stPath = Path.Combine(dir, "kokoro-82m.safetensors");
        Assert.True(File.Exists(pthPath) && File.Exists(stPath), $"missing kokoro-v1_0.pth / kokoro-82m.safetensors in {dir}");

        // Reproduce the exact pre-repack runtime path: recursive flatten + strip the inner "module."
        // wrapper that nn.DataParallel baked in (bert.module.embeddings.weight -> bert.embeddings.weight).
        using PytorchPickleLoader pth = new();
        pth.Load(pthPath, recursiveFlatten: true);
        const string wrap = "module.";
        Dictionary<string, Tensor> expected = new(StringComparer.Ordinal);
        foreach ((string key, Tensor tensor) in pth.GetAllTensors())
        {
            int dot = key.IndexOf('.');
            string mapped = dot >= 0 && key.AsSpan(dot + 1).StartsWith(wrap)
                ? string.Concat(key.AsSpan(0, dot + 1), key.AsSpan(dot + 1 + wrap.Length))
                : key;
            expected[mapped] = tensor;
        }

        using SafeTensorsLoader st = new();
        st.Load(stPath);
        Dictionary<string, Tensor> actual = st.GetAllTensors();

        Assert.Equal(expected.Count, actual.Count);
        foreach ((string key, Tensor want) in expected)
        {
            Assert.True(actual.TryGetValue(key, out Tensor? got) && got is not null, $"repacked file missing key '{key}'");
            Assert.Equal(want.DType, got!.DType);
            Assert.Equal(want.Shape.ToString(), got.Shape.ToString());
            ReadOnlySpan<byte> a = want.AsSpan<byte>();
            ReadOnlySpan<byte> b = got.AsSpan<byte>();
            Assert.True(a.SequenceEqual(b), $"tensor bytes differ for key '{key}'");
        }
    }
}
