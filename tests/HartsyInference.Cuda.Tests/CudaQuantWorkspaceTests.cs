using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Core.Tensors.Quant;
using HartsyInference.Cuda;
using HartsyInference.ModelAssets.BlockScale;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Device dequant of MXFP4-E8M0 and FP8-block-E8M0 expert matrices against the host codecs, through the expert cache upload path.</summary>
[Collection("CudaSerial")]
public sealed unsafe class CudaQuantWorkspaceTests
{
    private readonly ITestOutputHelper _output;

    public CudaQuantWorkspaceTests(ITestOutputHelper output) => _output = output;

    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(dir))
            dir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return dir;
    }

    // The kernel rounds to nearest even and emits the canonical quiet NaN.
    internal static ushort ToBf16(float value)
    {
        uint bits = BitConverter.SingleToUInt32Bits(value);
        if (float.IsNaN(value)) return 0x7FC0;
        bits += 0x7FFFu + ((bits >> 16) & 1u);
        return (ushort)(bits >> 16);
    }

    internal sealed record Case(QuantEncoding Encoding, int BlockRows, int BlockCols, int Rows, int Cols, int ScaleColOffset);

    internal static ExpertMatrix Build(Case c, int seed)
    {
        Random rng = new(seed);
        BlockGeometry geometry = new(c.BlockRows, c.BlockCols);
        (long scaleRows, long scaleCols) = geometry.ScaleShape(c.Rows, c.Cols);
        Tensor scale = new(new TensorShape(scaleRows, scaleCols + c.ScaleColOffset), DType.F8E8M0);
        Span<byte> scaleBytes = scale.AsSpan<byte>();
        for (int i = 0; i < scaleBytes.Length; i++) scaleBytes[i] = (byte)rng.Next(96, 160);
        // Subnormal 2^-127, NaN, and the extremes that overflow or underflow a float product.
        byte[] specials = [0, 255, 1, 254, 127, 0, 255];
        for (int i = 0; i < specials.Length; i++)
            scaleBytes[Math.Min(c.ScaleColOffset + i * 3, scaleBytes.Length - 1)] = specials[i];

        bool fp4 = c.Encoding == QuantEncoding.Mxfp4E8M0;
        Tensor packed = new(fp4 ? new TensorShape(c.Rows, c.Cols / 2) : new TensorShape(c.Rows, c.Cols), fp4 ? DType.I8 : DType.F8E4M3);
        rng.NextBytes(packed.AsSpan<byte>());
        Span<byte> data = packed.AsSpan<byte>();
        if (!fp4 && data.Length > 4)
        {
            data[0] = 0x7F; data[1] = 0xFF; data[2] = 0x80; data[3] = 0x00;
        }
        else if (fp4 && data.Length > 2)
        {
            data[0] = 0x88; data[1] = 0x8F;
        }
        QuantRecipe recipe = new()
        {
            Encoding = c.Encoding, Geometry = geometry, ScaleDType = DType.F8E8M0,
            LogicalRows = c.Rows, LogicalCols = c.Cols, Scale = scale, ScaleColOffset = c.ScaleColOffset,
        };
        return new ExpertMatrix(packed, recipe);
    }

    internal static ushort[] HostReference(ExpertMatrix matrix)
    {
        QuantRecipe recipe = matrix.Recipe!;
        float[] dense = new float[recipe.LogicalRows * recipe.LogicalCols];
        ReadOnlySpan<byte> packed = matrix.Weight.AsReadOnlySpan<byte>();
        if (recipe.Encoding == QuantEncoding.Mxfp4E8M0) Mxfp4E8M0Codec.DequantRows(packed, recipe, 0, recipe.LogicalRows, dense);
        else Fp8BlockE8M0Codec.DequantRows(packed, recipe, 0, recipe.LogicalRows, dense);
        ushort[] bf16 = new ushort[dense.Length];
        for (int i = 0; i < dense.Length; i++) bf16[i] = ToBf16(dense[i]);
        return bf16;
    }

    internal static ushort[] ReadBack(CudaBackend backend, QuantWorkspaceLease lease)
    {
        backend.Sync();
        ushort[] host = new ushort[lease.Rows * lease.Cols];
        fixed (ushort* destination = host) CudaMemory.CopyDeviceToHost(destination, lease.DevicePointer, (nuint)lease.Bytes);
        return host;
    }

    internal static void AssertBitExact(string name, ushort[] expected, ushort[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            if (expected[i] != actual[i]) Assert.Fail($"{name}: element {i} is 0x{actual[i]:X4}, host reference 0x{expected[i]:X4}.");
        }
    }

    private static ExpertBank BankOf(ExpertMatrix w1, ExpertMatrix w2, ExpertMatrix w3) =>
        new(0, 1, key => new ExpertWeights(key, w1, w2, w3));

    [Trait("Category", "GpuIntegration")]
    [Theory]
    [InlineData(QuantEncoding.Mxfp4E8M0, 1, 32, 8, 96, 0)]
    [InlineData(QuantEncoding.Mxfp4E8M0, 1, 32, 4, 64, 3)]
    [InlineData(QuantEncoding.Mxfp4E8M0, 1, 32, 64, 2304, 0)]
    [InlineData(QuantEncoding.Mxfp4E8M0, 4, 16, 16, 64, 1)]
    [InlineData(QuantEncoding.Fp8E4M3BlockE8M0, 32, 32, 64, 96, 0)]
    [InlineData(QuantEncoding.Fp8E4M3BlockE8M0, 128, 128, 256, 256, 0)]
    [InlineData(QuantEncoding.Fp8E4M3BlockE8M0, 1, 32, 8, 96, 2)]
    [InlineData(QuantEncoding.Fp8E4M3BlockE8M0, 32, 32, 40, 70, 1)]
    public void DeviceDequant_IsBitExactAgainstHostCodecAfterExpertUpload(QuantEncoding encoding, int br, int bc, int rows, int cols, int offset)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: no CUDA device"); return; }
        using CudaBackend backend = new(0, PtxDir());
        _output.WriteLine($"device: {backend.Capabilities.Name}");
        Case c = new(encoding, br, bc, rows, cols, offset);
        ExpertMatrix w1 = Build(c, 1), w2 = Build(c, 2), w3 = Build(c, 3);
        using CudaExpertCache cache = new(backend, 64L << 20, [BankOf(w1, w2, w3)]);

        using ExpertLease lease = cache.Acquire([new ExpertKey(0, 0)]);
        ExpertWeights weights = lease.Get(new ExpertKey(0, 0));
        ExpertMatrix[] matrices = [weights.W1, weights.W2, weights.W3];
        for (int m = 0; m < matrices.Length; m++)
        {
            using QuantWorkspaceLease dense = backend.QuantWorkspace.Dequantize(lease, matrices[m]);
            AssertBitExact($"w{m + 1} {c}", HostReference(matrices[m]), ReadBack(backend, dense));
        }
    }

    [Trait("Category", "GpuIntegration")]
    [Fact]
    public void Workspace_IsABoundedRingOfTwoAndRefusesOversizeMatrices()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: no CUDA device"); return; }
        using CudaBackend backend = new(0, PtxDir());
        Case c = new(QuantEncoding.Mxfp4E8M0, 1, 32, 8, 96, 0);
        ExpertMatrix w = Build(c, 5);
        using CudaExpertCache cache = new(backend, 64L << 20, [BankOf(w, w, w)]);
        using ExpertLease lease = cache.Acquire([new ExpertKey(0, 0)]);
        ExpertMatrix resident = lease.Get(new ExpertKey(0, 0)).W1;

        using QuantWorkspaceLease first = backend.QuantWorkspace.Dequantize(resident);
        using QuantWorkspaceLease second = backend.QuantWorkspace.Dequantize(resident);
        Assert.NotEqual(first.DevicePointer, second.DevicePointer);
        Assert.Equal(2, backend.QuantWorkspace.Rented);
        Assert.Throws<InvalidOperationException>(() => backend.QuantWorkspace.Dequantize(resident));

        first.Dispose();
        using QuantWorkspaceLease third = backend.QuantWorkspace.Dequantize(resident);
        Assert.Equal(first.DevicePointer, third.DevicePointer);
        ushort[] expected = HostReference(resident);
        AssertBitExact("second", expected, ReadBack(backend, second));
        AssertBitExact("third", expected, ReadBack(backend, third));

        lease.Dispose();
        cache.Dispose();
        Case big = new(QuantEncoding.Fp8E4M3BlockE8M0, 32, 32, 4096, 4096, 0);
        Assert.True(big.Rows * big.Cols * 2L > CudaQuantWorkspace.DefaultSlotBytes);
        ExpertMatrix oversize = Build(big, 6);
        using CudaExpertCache bigCache = new(backend, 128L << 20, [BankOf(oversize, oversize, oversize)]);
        using ExpertLease bigLease = bigCache.Acquire([new ExpertKey(0, 0)]);
        Assert.Throws<NotSupportedException>(() => backend.QuantWorkspace.Dequantize(bigLease.Get(new ExpertKey(0, 0)).W1));
    }

    [Trait("Category", "GpuIntegration")]
    [Fact]
    public void Dequantize_RefusesMatricesThatAreNotResidentOrHaveNoRecipe()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: no CUDA device"); return; }
        using CudaBackend backend = new(0, PtxDir());
        ExpertMatrix cold = Build(new Case(QuantEncoding.Mxfp4E8M0, 1, 32, 8, 96, 0), 7);
        ExpertMatrix dense0 = new(new Tensor(new TensorShape(4, 4), DType.F32));
        Assert.Throws<InvalidOperationException>(() => backend.QuantWorkspace.Dequantize(cold));
        using CudaExpertCache cache = new(backend, 64L << 20, [BankOf(cold, cold, cold)]);
        ExpertLease lease = cache.Acquire([new ExpertKey(0, 0)]);
        ExpertMatrix held = lease.Get(new ExpertKey(0, 0)).W1;
        Assert.Throws<ArgumentException>(() => backend.QuantWorkspace.Dequantize(lease, dense0));
        lease.Dispose();
        Assert.Throws<InvalidOperationException>(() => backend.QuantWorkspace.Dequantize(lease, held));
        Assert.Throws<InvalidOperationException>(() => new CudaExpertCache(backend, 1L << 20));
        ExpertMatrix dense = dense0;
        Assert.Throws<NotSupportedException>(() => backend.QuantWorkspace.Dequantize(dense));
    }
}
