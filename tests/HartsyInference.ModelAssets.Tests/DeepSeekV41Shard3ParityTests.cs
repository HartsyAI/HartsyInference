using System.Runtime.InteropServices;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Core.Tensors.Quant;
using HartsyInference.Cpu;
using HartsyInference.ModelAssets.BlockScale;
using HartsyInference.ModelAssets.Quant;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>M1: layers.0 of official shard 3 (dense F8_E4M3 with 32x32 E8M0 blocks, routed experts MXFP4 with 1x32 E8M0) decoded by the host
/// codecs against tests/python-reference/dump_deepseek_v41_shard3_ref.py, which applies the official convert.py dequant semantics.
/// Reads 64-row windows through the binder and codecs; only the touched pages of the shard are ever resident. The shard can be the real
/// download or a sparse replica holding the real bytes of those windows. Fixtures come from HARTSY_DSV41_SHARD3_FIXTURES.</summary>
[Trait("Category", "Integration")]
[Trait("Category", "RealWeights")]
public sealed class DeepSeekV41Shard3ParityTests
{
    private const string ShardName = "model-00003-of-00048.safetensors";
    private const long RssLimitBytes = 3L << 30;
    private const double LinearRelativeTolerance = 1e-4;

    private readonly ITestOutputHelper _output;

    public DeepSeekV41Shard3ParityTests(ITestOutputHelper output) => _output = output;

    private static string ShardPath => Path.Combine(
        Environment.GetEnvironmentVariable("HARTSY_DSV41_FLASH_DIR")
        ?? Path.Combine(TestPaths.ModelsDir, "DeepSeek-V4.1-Flash"), ShardName);

    private static string FixtureDir => Environment.GetEnvironmentVariable("HARTSY_DSV41_SHARD3_FIXTURES")
        ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "dsv41-ref", "shard3_fixtures");

    private sealed record Window(string Name, string Kind, long RowOffset, int Rows, int Cols, int Batch)
    {
        public string Key => Name[..Name.IndexOf('@')];
    }

    [Fact]
    public void Bind_Shard3Inventory_PairsEveryLayer0WeightWithItsScale()
    {
        if (!RealWeightGate.Require(_output.WriteLine, ShardPath)) return;
        using ShardedSafeTensorSet set = ShardedSafeTensorSet.OpenFiles([ShardPath]);

        QuantBindingSet bindings = QuantCompanionBinder.Bind(set.Inventory, QuantFlavor.Official);

        Assert.Empty(bindings.PerTensorFp8);
        int dense = 0, experts = 0;
        foreach (QuantBinding binding in bindings.Bindings.Values)
        {
            if (binding.Encoding == QuantEncoding.Fp8E4M3BlockE8M0)
            {
                Assert.Equal(new BlockGeometry(32, 32), binding.Geometry);
                dense++;
            }
            else
            {
                Assert.Equal(QuantEncoding.Mxfp4E8M0, binding.Encoding);
                Assert.Equal(new BlockGeometry(1, 32), binding.Geometry);
                experts++;
            }
            Assert.Equal(DType.F8E8M0, binding.ScaleDType);
        }
        _output.WriteLine($"{dense} dense fp8 and {experts} mxfp4 bindings from {set.Inventory.Count} tensors");
        Assert.Equal(8, dense);
        Assert.Equal(384 * 3, experts);
    }

    [Fact]
    public void DenseFp8Windows_MatchOfficialDequantAndLinear()
    {
        if (!RealWeightGate.Require(_output.WriteLine, ShardPath, Path.Combine(FixtureDir, "manifest.tsv"))) return;
        RunWindows("dense");
    }

    [Fact]
    public void RoutedExpertMxfp4Windows_MatchOfficialDequantAndLinear()
    {
        if (!RealWeightGate.Require(_output.WriteLine, ShardPath, Path.Combine(FixtureDir, "manifest.tsv"))) return;
        RunWindows("fp4");
    }

    private void RunWindows(string kind)
    {
        using ShardedSafeTensorSet set = ShardedSafeTensorSet.OpenFiles([ShardPath]);
        QuantBindingSet bindings = QuantCompanionBinder.Bind(set.Inventory, QuantFlavor.Official);
        using IBackend backend = new CpuBackend();
        List<Window> windows = ReadManifest().Where(w => w.Kind == kind).ToList();
        Assert.NotEmpty(windows);

        foreach (Window window in windows)
        {
            QuantBinding binding = bindings.Bindings[window.Key + ".weight"];
            QuantRecipe recipe = binding.ToRecipe(set.GetTensor);
            Tensor weight = set.GetTensor(binding.WeightKey);
            float[] expected = ReadFloats(window.Name + ".w.f32");
            float[] actual = new float[window.Rows * window.Cols];

            DequantWindow(recipe, weight, window.RowOffset, window.Rows, actual);
            if (window.Key.EndsWith(".wo_a", StringComparison.Ordinal)) RoundToBf16(actual);
            AssertBitExact(window.Name, expected, actual);

            if (kind == "dense") AssertSlicedRecipeAgrees(recipe, weight, window, expected);
            double relative = LinearRelativeError(backend, window, actual);
            _output.WriteLine($"{window.Name}: dequant bit-exact, linear max relative error {relative:E2}");
            Assert.True(relative <= LinearRelativeTolerance,
                $"{window.Name}: linear relative error {relative:E3} exceeds {LinearRelativeTolerance:E0}.");
        }
        Assert.True(ResidentBytes() < RssLimitBytes, $"RSS {ResidentBytes() >> 20} MiB exceeds {RssLimitBytes >> 20} MiB.");
    }

    private static void DequantWindow(QuantRecipe recipe, Tensor weight, long rowOffset, long rows, Span<float> dest)
    {
        ReadOnlySpan<byte> packed = weight.AsSpan<byte>();
        if (recipe.Encoding == QuantEncoding.Mxfp4E8M0) Mxfp4E8M0Codec.DequantRows(packed, recipe, rowOffset, rows, dest);
        else Fp8BlockE8M0Codec.DequantRows(packed, recipe, rowOffset, rows, dest);
    }

    private static void AssertSlicedRecipeAgrees(QuantRecipe recipe, Tensor weight, Window window, float[] expected)
    {
        QuantRecipe sliced = recipe.SliceRows(window.RowOffset, window.Rows, window.Key + ".weight");
        ReadOnlySpan<byte> packed = weight.AsSpan<byte>().Slice((int)(window.RowOffset * window.Cols), window.Rows * window.Cols);
        float[] viaSlice = new float[expected.Length];
        Fp8BlockE8M0Codec.DequantRows(packed, sliced, 0, window.Rows, viaSlice);
        if (window.Key.EndsWith(".wo_a", StringComparison.Ordinal)) RoundToBf16(viaSlice);
        AssertBitExact(window.Name + " (sliced recipe)", expected, viaSlice);
    }

    private double LinearRelativeError(IBackend backend, Window window, float[] weights)
    {
        float[] x = ReadFloats(window.Name + ".x.f32");
        float[] expected = ReadFloats(window.Name + ".y.f32");
        using Tensor input = new Tensor(new TensorShape(window.Batch, window.Cols), DType.F32);
        using Tensor weight = new Tensor(new TensorShape(window.Rows, window.Cols), DType.F32);
        using Tensor output = new Tensor(new TensorShape(window.Batch, window.Rows), DType.F32);
        x.CopyTo(input.AsSpan<float>());
        weights.CopyTo(weight.AsSpan<float>());

        backend.Linear(output, input, weight, null);

        ReadOnlySpan<float> got = output.AsSpan<float>();
        double maxDiff = 0, maxAbs = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            maxDiff = Math.Max(maxDiff, Math.Abs((double)got[i] - expected[i]));
            maxAbs = Math.Max(maxAbs, Math.Abs((double)expected[i]));
        }
        return maxDiff / Math.Max(maxAbs, 1e-30);
    }

    private static void AssertBitExact(string name, float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            uint want = BitConverter.SingleToUInt32Bits(expected[i]);
            uint got = BitConverter.SingleToUInt32Bits(actual[i]);
            if (want != got) Assert.Fail($"{name}: element {i} is {actual[i]:R} (0x{got:X8}), reference {expected[i]:R} (0x{want:X8}).");
        }
    }

    // torch .bfloat16() rounds to nearest even; TensorCasts.F32ToBf16Bits truncates.
    private static void RoundToBf16(Span<float> values)
    {
        for (int i = 0; i < values.Length; i++)
        {
            uint bits = BitConverter.SingleToUInt32Bits(values[i]);
            bits += 0x7FFFu + ((bits >> 16) & 1u);
            values[i] = BitConverter.UInt32BitsToSingle(bits & 0xFFFF0000u);
        }
    }

    private static List<Window> ReadManifest()
    {
        List<Window> windows = new();
        foreach (string line in File.ReadLines(Path.Combine(FixtureDir, "manifest.tsv")).Skip(1))
        {
            string[] f = line.Split('\t');
            windows.Add(new Window(f[0], f[1], long.Parse(f[2]), int.Parse(f[3]), int.Parse(f[4]), int.Parse(f[5])));
        }
        return windows;
    }

    private static float[] ReadFloats(string fileName) =>
        MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(Path.Combine(FixtureDir, fileName))).ToArray();

    private static long ResidentBytes()
    {
        foreach (string line in File.ReadLines("/proc/self/status"))
        {
            if (line.StartsWith("VmRSS:", StringComparison.Ordinal))
                return long.Parse(line.Split(' ', StringSplitOptions.RemoveEmptyEntries)[1]) * 1024;
        }
        throw new InvalidOperationException("VmRSS not found in /proc/self/status.");
    }
}
