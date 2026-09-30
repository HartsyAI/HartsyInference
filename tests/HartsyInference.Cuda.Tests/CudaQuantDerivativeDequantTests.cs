using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Core.Tensors.Quant;
using HartsyInference.Cuda;
using HartsyInference.ModelAssets.BlockScale;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Device dequant of ModelOpt NVFP4, Quark U8-scale MXFP4 and MLX affine expert matrices against the host codecs.</summary>
[Collection("CudaSerial")]
[Trait("Category", "GpuIntegration")]
public sealed unsafe class CudaQuantDerivativeDequantTests
{
    private readonly ITestOutputHelper _output;

    public CudaQuantDerivativeDequantTests(ITestOutputHelper output) => _output = output;

    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(dir))
            dir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return dir;
    }

    private static ExpertMatrix BuildNvfp4(int rows, int cols, int blockRows, int scaleColOffset, float global, int seed)
    {
        Random rng = new(seed);
        BlockGeometry geometry = new(blockRows, 16);
        (long scaleRows, long scaleCols) = geometry.ScaleShape(rows, cols);
        Tensor scale = new(new TensorShape(scaleRows, scaleCols + scaleColOffset), DType.F8E4M3);
        Span<byte> scaleBytes = scale.AsSpan<byte>();
        rng.NextBytes(scaleBytes);
        // NaN (0x7F, 0xFF), the smallest subnormal, negative zero and the largest finite value.
        byte[] specials = [0x7F, 0xFF, 0x01, 0x80, 0x7E, 0x00];
        for (int i = 0; i < specials.Length; i++) scaleBytes[Math.Min(scaleColOffset + i * 2, scaleBytes.Length - 1)] = specials[i];
        Tensor packed = new(new TensorShape(rows, cols / 2), DType.I8);
        rng.NextBytes(packed.AsSpan<byte>());
        packed.AsSpan<byte>()[0] = 0x88;
        Tensor globalScale = new(new TensorShape(1), DType.F32);
        globalScale.AsSpan<float>()[0] = global;
        QuantRecipe recipe = new()
        {
            Encoding = QuantEncoding.Nvfp4, Geometry = geometry, ScaleDType = DType.F8E4M3, LogicalRows = rows, LogicalCols = cols,
            Scale = scale, GlobalScale = globalScale, ScaleColOffset = scaleColOffset,
        };
        return new ExpertMatrix(packed, recipe);
    }

    private static ExpertMatrix BuildQuark(int rows, int cols, int seed)
    {
        Random rng = new(seed);
        BlockGeometry geometry = new(1, 32);
        (long scaleRows, long scaleCols) = geometry.ScaleShape(rows, cols);
        Tensor scale = new(new TensorShape(scaleRows, scaleCols), DType.U8);
        Span<byte> scaleBytes = scale.AsSpan<byte>();
        for (int i = 0; i < scaleBytes.Length; i++) scaleBytes[i] = (byte)rng.Next(100, 150);
        scaleBytes[0] = 255;
        Tensor packed = new(new TensorShape(rows, cols / 2), DType.U8);
        rng.NextBytes(packed.AsSpan<byte>());
        QuantRecipe recipe = new()
        {
            Encoding = QuantEncoding.Mxfp4E8M0, Geometry = geometry, ScaleDType = DType.U8, LogicalRows = rows, LogicalCols = cols, Scale = scale,
        };
        return new ExpertMatrix(packed, recipe);
    }

    private static ExpertMatrix BuildAffine(int bits, int rows, int cols, int scaleColOffset, int seed)
    {
        Random rng = new(seed);
        BlockGeometry geometry = new(1, 64);
        (long scaleRows, long scaleCols) = geometry.ScaleShape(rows, cols);
        TensorShape scaleShape = new(scaleRows, scaleCols + scaleColOffset);
        Tensor scale = new(scaleShape, DType.F32);
        Tensor bias = new(scaleShape, DType.F32);
        Span<float> s = scale.AsSpan<float>(), b = bias.AsSpan<float>();
        for (int i = 0; i < s.Length; i++)
        {
            s[i] = (float)((rng.NextDouble() - 0.4) * 0.05);
            b[i] = (float)((rng.NextDouble() - 0.5) * 0.4);
        }
        // Non-finite and cancelling values exercise the unfused multiply-then-add.
        s[scaleColOffset] = float.NaN;
        b[Math.Min(scaleColOffset + 1, b.Length - 1)] = float.PositiveInfinity;
        s[s.Length - 1] = 3.0e38f;
        Tensor packed = new(new TensorShape(rows, bits == 4 ? cols / 2 : cols), DType.U8);
        rng.NextBytes(packed.AsSpan<byte>());
        QuantRecipe recipe = new()
        {
            Encoding = bits == 4 ? QuantEncoding.AffineInt4 : QuantEncoding.AffineInt8, Geometry = geometry, ScaleDType = DType.F32,
            LogicalRows = rows, LogicalCols = cols, Scale = scale, Bias = bias, ScaleColOffset = scaleColOffset,
        };
        return new ExpertMatrix(packed, recipe);
    }

    private static ushort[] HostReference(ExpertMatrix matrix)
    {
        QuantRecipe recipe = matrix.Recipe!;
        float[] dense = new float[recipe.LogicalRows * recipe.LogicalCols];
        ReadOnlySpan<byte> packed = matrix.Weight.AsReadOnlySpan<byte>();
        switch (recipe.Encoding)
        {
            case QuantEncoding.Nvfp4: ModelOptNvfp4Codec.DequantRows(packed, recipe, 0, recipe.LogicalRows, dense); break;
            case QuantEncoding.Mxfp4E8M0: Mxfp4E8M0Codec.DequantRows(packed, recipe, 0, recipe.LogicalRows, dense); break;
            default: AffineIntCodec.DequantRows(packed, recipe, 0, recipe.LogicalRows, dense); break;
        }
        ushort[] bf16 = new ushort[dense.Length];
        for (int i = 0; i < dense.Length; i++) bf16[i] = CudaQuantWorkspaceTests.ToBf16(dense[i]);
        return bf16;
    }

    private void AssertDeviceMatchesHost(string name, Func<int, ExpertMatrix> build)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: no CUDA device"); return; }
        using CudaBackend backend = new(0, PtxDir());
        _output.WriteLine($"device: {backend.Capabilities.Name}");
        ExpertMatrix w1 = build(1), w2 = build(2), w3 = build(3);
        using CudaExpertCache cache = new(backend, 64L << 20, [new ExpertBank(0, 1, key => new ExpertWeights(key, w1, w2, w3))]);
        using ExpertLease lease = cache.Acquire([new ExpertKey(0, 0)]);
        ExpertWeights weights = lease.Get(new ExpertKey(0, 0));
        ExpertMatrix[] matrices = [weights.W1, weights.W2, weights.W3];
        for (int m = 0; m < matrices.Length; m++)
        {
            using QuantWorkspaceLease dense = backend.QuantWorkspace.Dequantize(lease, matrices[m]);
            CudaQuantWorkspaceTests.AssertBitExact($"{name} w{m + 1}", HostReference(matrices[m]), CudaQuantWorkspaceTests.ReadBack(backend, dense));
        }
    }

    [Theory]
    [InlineData(8, 64, 1, 0, 0.0031f)]
    [InlineData(4, 96, 1, 3, 1.0f)]
    [InlineData(64, 2304, 1, 0, 0.00042f)]
    [InlineData(16, 64, 4, 1, 7.5f)]
    [InlineData(8, 64, 1, 0, 1.0e37f)]
    public void Nvfp4ModelOpt_IsBitExactAgainstHostCodec(int rows, int cols, int blockRows, int offset, float global) =>
        AssertDeviceMatchesHost(
            $"nvfp4 {rows}x{cols} br{blockRows} off{offset} g{global}", seed => BuildNvfp4(rows, cols, blockRows, offset, global, seed));

    [Theory]
    [InlineData(8, 96)]
    [InlineData(64, 2304)]
    public void QuarkU8ScaleMxfp4_IsBitExactAgainstHostCodec(int rows, int cols) =>
        AssertDeviceMatchesHost($"quark {rows}x{cols}", seed => BuildQuark(rows, cols, seed));

    [Theory]
    [InlineData(4, 8, 128, 0)]
    [InlineData(4, 4, 192, 2)]
    [InlineData(4, 64, 2048, 0)]
    [InlineData(8, 8, 128, 0)]
    [InlineData(8, 4, 192, 1)]
    [InlineData(8, 32, 1024, 0)]
    public void MlxAffine_IsBitExactAgainstHostCodec(int bits, int rows, int cols, int offset) =>
        AssertDeviceMatchesHost($"mlx int{bits} {rows}x{cols} off{offset}", seed => BuildAffine(bits, rows, cols, offset, seed));

    [Fact]
    public void Dequantize_RefusesRecipesTheDeviceCannotRead()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: no CUDA device"); return; }
        using CudaBackend backend = new(0, PtxDir());
        ExpertMatrix bf16Scales = BuildAffine(4, 4, 128, 0, 9);
        QuantRecipe r = bf16Scales.Recipe!;
        Tensor bfScale = new(r.Scale!.Shape, DType.BF16);
        QuantRecipe wrong = new()
        {
            Encoding = r.Encoding, Geometry = r.Geometry, ScaleDType = DType.BF16, LogicalRows = r.LogicalRows, LogicalCols = r.LogicalCols,
            Scale = bfScale, Bias = new Tensor(r.Scale.Shape, DType.BF16),
        };
        Assert.Throws<NotSupportedException>(() => backend.QuantWorkspace.Dequantize(new ExpertMatrix(bf16Scales.Weight, wrong)));

        ExpertMatrix nvfp4 = BuildNvfp4(4, 64, 1, 0, 1.0f, 10);
        QuantRecipe q = nvfp4.Recipe!;
        QuantRecipe noGlobal = q with { GlobalScale = null };
        Assert.Throws<NotSupportedException>(() => backend.QuantWorkspace.Dequantize(new ExpertMatrix(nvfp4.Weight, noGlobal)));

        QuantRecipe exl3 = q with { Encoding = QuantEncoding.Exl3Trellis };
        Assert.Throws<NotSupportedException>(() => backend.QuantWorkspace.Dequantize(new ExpertMatrix(nvfp4.Weight, exl3)));
    }
}
