using System.Runtime.InteropServices;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Core.Tensors.Quant;
using HartsyInference.Cuda;
using HartsyInference.ModelAssets.BlockScale;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Device EXL3 2-bit MCG trellis decode against the host <see cref="Exl3Codec"/>: BF16 must equal the host F32 result rounded once, bit for bit.</summary>
[Collection("CudaSerial")]
public sealed class CudaExl3DequantTests
{
    private readonly ITestOutputHelper _output;

    public CudaExl3DequantTests(ITestOutputHelper output) => _output = output;

    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(dir))
            dir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return dir;
    }

    private static byte[] Fixture(string name) => File.ReadAllBytes(Path.Combine(AppContext.BaseDirectory, "Fixtures", "Exl3", name));

    private static Tensor Wrap(byte[] data, DType dtype, params long[] dims)
    {
        Tensor t = new(new TensorShape(dims), dtype);
        data.CopyTo(t.AsSpan<byte>());
        return t;
    }

    private static ExpertMatrix Build(int inDim, int outDim, byte[] trellis, byte[] suh, byte[] svh)
    {
        Tensor mcg = new(new TensorShape(1), DType.I32);
        mcg.AsSpan<int>()[0] = unchecked((int)Exl3Format.McgMultiplier);
        QuantRecipe recipe = new()
        {
            Encoding = QuantEncoding.Exl3Trellis, Geometry = new BlockGeometry(16, 16), ScaleDType = DType.F16, LogicalRows = outDim, LogicalCols = inDim,
            Exl3 = new Exl3Companions(Wrap(suh, DType.F16, inDim), Wrap(svh, DType.F16, outDim), mcg, Exl3Format.SupportedBits),
        };
        return new ExpertMatrix(Wrap(trellis, DType.I16, inDim / 16, outDim / 16, 32), recipe);
    }

    private static ExpertMatrix Random(int inDim, int outDim, int seed)
    {
        Random rng = new(seed);
        byte[] trellis = new byte[Exl3Format.PackedBytes(inDim, outDim, 2)];
        rng.NextBytes(trellis);
        return Build(inDim, outDim, trellis, Signs(rng, inDim), Signs(rng, outDim));
    }

    private static byte[] Signs(Random rng, int count)
    {
        Half[] h = new Half[count];
        for (int i = 0; i < count; i++) h[i] = (Half)(float)((rng.NextDouble() < 0.5 ? -1 : 1) * (0.25 + rng.NextDouble()));
        byte[] bytes = new byte[count * 2];
        MemoryMarshal.AsBytes(h.AsSpan()).CopyTo(bytes);
        return bytes;
    }

    private static ExpertMatrix FixtureMatrix() => Build(256, 384, Fixture("trellis.i16"), Fixture("suh.f16"), Fixture("svh.f16"));

    private static float[] HostFloats(ExpertMatrix matrix)
    {
        QuantRecipe recipe = matrix.Recipe!;
        float[] dense = new float[recipe.LogicalRows * recipe.LogicalCols];
        Exl3Codec.DequantRows(matrix.Weight.AsReadOnlySpan<byte>(), recipe, 0, recipe.LogicalRows, dense);
        return dense;
    }

    private void AssertDeviceMatchesHost(string name, params ExpertMatrix[] matrices)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: no CUDA device"); return; }
        using CudaBackend backend = new(0, PtxDir());
        _output.WriteLine($"device: {backend.Capabilities.Name}");
        Assert.True(backend.SupportsRecipeDequant, "dequant_recipe_to_bf16.ptx did not load with the EXL3 kernel");
        ExpertMatrix w1 = matrices[0], w2 = matrices[1 % matrices.Length], w3 = matrices[2 % matrices.Length];
        using CudaExpertCache cache = new(backend, 256L << 20, [new ExpertBank(0, 1, key => new ExpertWeights(key, w1, w2, w3))]);
        using ExpertLease lease = cache.Acquire([new ExpertKey(0, 0)]);
        ExpertWeights weights = lease.Get(new ExpertKey(0, 0));
        ExpertMatrix[] resident = [weights.W1, weights.W2, weights.W3];
        for (int m = 0; m < resident.Length; m++)
        {
            float[] host = HostFloats(resident[m]);
            ushort[] expected = new ushort[host.Length];
            for (int i = 0; i < host.Length; i++) expected[i] = CudaQuantWorkspaceTests.ToBf16(host[i]);
            using QuantWorkspaceLease dense = backend.QuantWorkspace.Dequantize(lease, resident[m]);
            CudaQuantWorkspaceTests.AssertBitExact($"{name} w{m + 1}", expected, CudaQuantWorkspaceTests.ReadBack(backend, dense));
        }
    }

    [Trait("Category", "GpuIntegration")]
    [Fact]
    public void Fixture_DeviceBf16_EqualsHostF32RoundedOnce() => AssertDeviceMatchesHost("fixture 256x384", FixtureMatrix());

    [Trait("Category", "GpuIntegration")]
    [Theory]
    [InlineData(128, 128)]
    [InlineData(384, 256)]
    [InlineData(640, 384)]
    public void RandomTrellis_DeviceBf16_EqualsHostF32RoundedOnce(int inDim, int outDim) =>
        AssertDeviceMatchesHost($"random {inDim}x{outDim}", Random(inDim, outDim, 11), Random(inDim, outDim, 12), Random(inDim, outDim, 13));

    [Trait("Category", "GpuIntegration")]
    [Fact]
    public void Fixture_DeviceBf16_GapToUpstreamFusedFp16_IsReported()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: no CUDA device"); return; }
        using CudaBackend backend = new(0, PtxDir());
        ExpertMatrix fixture = FixtureMatrix();
        using CudaExpertCache cache = new(backend, 64L << 20, [new ExpertBank(0, 1, key => new ExpertWeights(key, fixture, fixture, fixture))]);
        using ExpertLease lease = cache.Acquire([new ExpertKey(0, 0)]);
        ExpertMatrix resident = lease.Get(new ExpertKey(0, 0)).W1;
        using QuantWorkspaceLease dense = backend.QuantWorkspace.Dequantize(lease, resident);
        ushort[] bf16 = CudaQuantWorkspaceTests.ReadBack(backend, dense);
        // Upstream's fused kernel writes W as [in, out]; the device writes M[o, i] = W[i, o].
        Half[] fused = new Half[256 * 384];
        Fixture("w_fused.f16").CopyTo(MemoryMarshal.AsBytes(fused.AsSpan()));
        double max = 0, worst = 0;
        foreach (Half h in fused) max = Math.Max(max, Math.Abs((double)h));
        for (int o = 0; o < 384; o++)
        {
            for (int i = 0; i < 256; i++)
            {
                double device = BitConverter.Int32BitsToSingle(bf16[o * 256 + i] << 16);
                worst = Math.Max(worst, Math.Abs(device - (double)fused[i * 384 + o]));
            }
        }
        _output.WriteLine($"max|W|={max:G6} worst |device BF16 - upstream fused fp16|={worst:G6} ({worst / max:P3} of max|W|)");
        // BF16 has 8 significant bits, so its own rounding alone can reach 2^-9 of an element; the sum with the fp16 fused gap stays under 2^-7 of max|W|.
        Assert.True(worst <= max / 128.0, $"device BF16 differs from upstream fused fp16 by {worst / max:P3} of max|W|");
    }
}
