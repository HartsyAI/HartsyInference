using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;
using HartsyInference.ModelAssets.Nvfp4;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>A resident nvfp4 weight that stands in for an F16 one (<see cref="QuantWeightInfo.DenseDType"/>) must give
/// <c>Linear</c> output bit-identical to the F16 weight the eager open makes, on every route an unpacking GEMM takes:
/// F32 and F16 activations, the small-m GEMV, a bias, a damped alpha, padded scales, and the cached weight cast.</summary>
[Collection("CudaSerial")]
[Trait("Category", "GpuIntegration")]
public sealed unsafe class Nvfp4DenseTwinParityTests
{
    private readonly ITestOutputHelper _output;

    public Nvfp4DenseTwinParityTests(ITestOutputHelper output) => _output = output;

    [Theory]
    [InlineData(64, 256, 256, "F32", "F32", "none", 1.0f)]
    [InlineData(64, 256, 512, "F32", "F32", "F32", 1.0f)]
    [InlineData(64, 384, 512, "F16", "F16", "BF16", 0.03125f)]
    [InlineData(77, 256, 256, "F32", "F16", "F16", 1.0f)]
    [InlineData(1, 256, 512, "F32", "F32", "BF16", 1.0f)]   // the small-m GEMV
    [InlineData(2, 512, 256, "F32", "F32", "none", 1.0f)]
    [InlineData(64, 100, 48, "F32", "F32", "none", 1.0f)]   // rows pad to 128, block columns 3 -> 4
    [InlineData(32, 848, 3360, "F16", "F16", "none", 1.0f)] // Boogu's k/v: both pads at a real shape
    public void Linear_MatchesTheEagerF16Weight_BitForBit(int m, int n, int k, string input, string output,
        string bias, float damp)
    {
        using CudaBackend? backend = TryOpenBackend();
        if (backend is null) return;   // tier-lint: guarded
        using Operands ops = Operands.Create(m, n, k, ParseDType(input), ParseDType(output), bias, seed: m * 7919 + n * 31 + k);
        using Tensor eager = CheckpointConvertUtils.DequantNvfp4ToF16(ops.Packed, ops.BlockScale, ((float*)ops.GlobalScale.DataPointer)[0]);
        using Tensor resident = ops.AttachResident(DType.F16);
        eager.Fp8ScaleFactor = damp;
        resident.Fp8ScaleFactor = damp;
        try
        {
            uint[] expected = RunLinear(backend, ops, eager);
            uint[] got = RunLinear(backend, ops, resident);
            AssertBitIdentical(expected, got, $"m={m} n={n} k={k} {input}->{output} bias={bias} damp={damp}");
        }
        finally
        {
            backend.FreeWeights([eager, resident]);
        }
    }

    /// <summary>A recipe that widens its dense DiT weights to F32 (HiDream) gets an F32 twin: the F16 unpack, widened.</summary>
    [Theory]
    [InlineData(64, 256, 512, "none")]
    [InlineData(2, 384, 256, "F32")]
    public void Linear_MatchesTheEagerF32Weight_BitForBit(int m, int n, int k, string bias)
    {
        using CudaBackend? backend = TryOpenBackend();
        if (backend is null) return;   // tier-lint: guarded
        using Operands ops = Operands.Create(m, n, k, DType.F32, DType.F32, bias, seed: m * 131 + n + k);
        using Tensor eagerF16 = CheckpointConvertUtils.DequantNvfp4ToF16(ops.Packed, ops.BlockScale, ((float*)ops.GlobalScale.DataPointer)[0]);
        using Tensor eager = eagerF16.CastTo(DType.F32);
        using Tensor resident = ops.AttachResident(DType.F32);
        try
        {
            uint[] expected = RunLinear(backend, ops, eager);
            uint[] got = RunLinear(backend, ops, resident);
            AssertBitIdentical(expected, got, $"F32 twin m={m} n={n} k={k} bias={bias}");
        }
        finally
        {
            backend.FreeWeights([eager, resident]);
        }
    }

    /// <summary>A weight with more rows than the output is wide (an untrimmed quantizer pad) is refused, not written past the output.</summary>
    [Fact]
    public void OutputNarrowerThanTheWeight_IsRefused()
    {
        using CudaBackend? backend = TryOpenBackend();
        if (backend is null) return;   // tier-lint: guarded
        using Operands ops = Operands.Create(16, 848, 256, DType.F16, DType.F16, "none", seed: 848);
        using Tensor resident = ops.AttachResident(DType.F16);
        using Tensor eager = CheckpointConvertUtils.DequantNvfp4ToF16(ops.Packed, ops.BlockScale, ((float*)ops.GlobalScale.DataPointer)[0]);
        using Tensor narrow = new Tensor(new TensorShape(16, 840), DType.F16);
        try
        {
            Assert.Throws<ArgumentException>(() => backend.Linear(narrow, ops.Input, resident, null));
            Assert.Throws<ArgumentException>(() => backend.Linear(narrow, ops.Input, eager, null));
        }
        finally
        {
            backend.FreeWeights([eager, resident]);
        }
    }

    /// <summary>The negative control: without a dense dtype an F32 activation meets a BF16 unpack, so the comparison above can fail.</summary>
    [Fact]
    public void WithoutADenseDType_AnF32ActivationDoesNotMatchTheEagerWeight()
    {
        using CudaBackend? backend = TryOpenBackend();
        if (backend is null) return;   // tier-lint: guarded
        using Operands ops = Operands.Create(64, 256, 256, DType.F32, DType.F32, "none", seed: 99);
        using Tensor eager = CheckpointConvertUtils.DequantNvfp4ToF16(ops.Packed, ops.BlockScale, ((float*)ops.GlobalScale.DataPointer)[0]);
        using Tensor resident = ops.AttachResident(denseDType: null);
        try
        {
            uint[] expected = RunLinear(backend, ops, eager);
            uint[] got = RunLinear(backend, ops, resident);
            long mismatches = 0;
            for (long i = 0; i < expected.LongLength; i++) if (expected[i] != got[i]) mismatches++;
            _output.WriteLine($"no dense dtype: {mismatches} of {expected.LongLength} output words differ");
            Assert.True(mismatches > 0, "an undeclared resident weight matched the F16 twin, so the parity cases prove nothing");
        }
        finally
        {
            backend.FreeWeights([eager, resident]);
        }
    }

    [Fact]
    public void CachedWeightCast_MatchesTheEagerF16Weight_BitForBit()
    {
        using CudaBackend? backend = TryOpenBackend();
        if (backend is null) return;   // tier-lint: guarded
        using Operands ops = Operands.Create(64, 256, 512, DType.F32, DType.F32, "BF16", seed: 4711);
        using Tensor eager = CheckpointConvertUtils.DequantNvfp4ToF16(ops.Packed, ops.BlockScale, ((float*)ops.GlobalScale.DataPointer)[0]);
        using Tensor resident = ops.AttachResident(DType.F16);
        try
        {
            // Preloaded, so the second call reads the cached F16 cast rather than unpacking again.
            KnobStore.Set(EngineKnobs.ResidentCastCache, true);
            backend.PreloadWeights([eager, resident]);
            uint[] expected = RunLinear(backend, ops, eager);
            for (int pass = 1; pass <= 2; pass++)
            {
                uint[] got = RunLinear(backend, ops, resident);
                bool cached = GpuTransferHelper.TryGetWeightCast(resident, DType.F16, out ulong _);
                _output.WriteLine($"pass {pass}: F16 cast cached={cached}");
                AssertBitIdentical(expected, got, $"cached-cast pass {pass}");
            }
        }
        finally
        {
            KnobStore.Clear(EngineKnobs.ResidentCastCache);
            backend.FreeWeights([eager, resident]);
        }
    }

    /// <summary>With <c>vram.residentCastCache</c> off (the default) a preloaded resident weight keeps no dense copy and still matches its twin.</summary>
    [Fact]
    public void CastCacheOff_TheDefault_KeepsNoDenseCopy_AndMatchesTheEagerWeight()
    {
        using CudaBackend? backend = TryOpenBackend();
        if (backend is null) return;   // tier-lint: guarded
        using Operands ops = Operands.Create(64, 256, 512, DType.F32, DType.F32, "none", seed: 5150);
        using Tensor eager = CheckpointConvertUtils.DequantNvfp4ToF16(ops.Packed, ops.BlockScale, ((float*)ops.GlobalScale.DataPointer)[0]);
        using Tensor resident = ops.AttachResident(DType.F16);
        KnobStore.Set(EngineKnobs.ResidentCastCache, false);
        try
        {
            backend.PreloadWeights([eager, resident]);
            uint[] expected = RunLinear(backend, ops, eager);
            uint[] got = RunLinear(backend, ops, resident);
            Assert.False(GpuTransferHelper.TryGetWeightCast(resident, DType.F16, out ulong _), "the dense copy was cached anyway");
            AssertBitIdentical(expected, got, "cast cache off");
        }
        finally
        {
            KnobStore.Clear(EngineKnobs.ResidentCastCache);
            backend.FreeWeights([eager, resident]);
        }
    }

    private void AssertBitIdentical(uint[] expected, uint[] got, string label)
    {
        long mismatches = 0;
        for (long i = 0; i < expected.LongLength; i++) if (expected[i] != got[i]) mismatches++;
        _output.WriteLine($"{label}: {mismatches} of {expected.LongLength} output words differ");
        Assert.Equal(0L, mismatches);
    }

    private static uint[] RunLinear(CudaBackend backend, Operands ops, Tensor weight)
    {
        using Tensor result = new Tensor(new TensorShape(ops.M, ops.N), ops.OutputDType);
        backend.Linear(result, ops.Input, weight, ops.Bias);
        backend.Sync();
        uint[] words = new uint[(long)ops.M * ops.N];
        if (ops.OutputDType == DType.F32)
        {
            ReadOnlySpan<float> values = result.AsReadOnlySpan<float>();
            for (int i = 0; i < words.Length; i++) words[i] = BitConverter.SingleToUInt32Bits(values[i]);
        }
        else
        {
            ReadOnlySpan<Half> values = result.AsReadOnlySpan<Half>();
            for (int i = 0; i < words.Length; i++) words[i] = BitConverter.HalfToUInt16Bits(values[i]);
        }
        return words;
    }

    private static DType ParseDType(string name) => name switch
    {
        "F32" => DType.F32,
        "F16" => DType.F16,
        "BF16" => DType.BF16,
        _ => throw new ArgumentOutOfRangeException(nameof(name), name, null),
    };

    private CudaBackend? TryOpenBackend()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: no CUDA device"); return null; }
        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDir)) ptxDir = Path.Combine(RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        // Without the module the resident weight would take the host fallback and this would gate nothing.
        if (!File.Exists(Path.Combine(ptxDir, "dequant_nvfp4_to_f16.ptx")))
        {
            _output.WriteLine($"SKIPPED: dequant_nvfp4_to_f16.ptx not in {ptxDir}");
            return null;
        }
        CudaBackend backend = new CudaBackend(0, ptxDir);
        _output.WriteLine($"device: {backend.Context.DeviceName}");
        return backend;
    }

    /// <summary>A packed nvfp4 weight in the on-disk U8 form plus one Linear's activation and bias.</summary>
    private sealed class Operands : IDisposable
    {
        public required Tensor Packed { get; init; }
        public required Tensor BlockScale { get; init; }
        public required Tensor GlobalScale { get; init; }
        public required Tensor Input { get; init; }
        public Tensor? Bias { get; init; }
        public required int M { get; init; }
        public required int N { get; init; }
        public required DType OutputDType { get; init; }

        public static Operands Create(int m, int n, int k, DType input, DType output, string bias, int seed)
        {
            Random rng = new Random(seed);
            int paddedRows = (n + 127) / 128 * 128;
            int paddedCols = (k / Nvfp4Codec.GroupSize + 3) / 4 * 4;
            Tensor packed = new Tensor(new TensorShape(n, k / 2), DType.U8);
            rng.NextBytes(new Span<byte>(packed.DataPointer, n * (k / 2)));
            // Real quantizers saturate at 448, so 0x7F/0xFF (480 on the device, NaN in the eager table) never occur.
            Tensor blockScale = new Tensor(new TensorShape(paddedRows, paddedCols), DType.F8E4M3);
            byte* scales = (byte*)blockScale.DataPointer;
            for (long i = 0; i < (long)paddedRows * paddedCols; i++) scales[i] = (byte)(rng.Next(0x00, 0x7F) | (rng.Next(2) << 7));
            Tensor globalScale = new Tensor(new TensorShape(1), DType.F32);
            ((float*)globalScale.DataPointer)[0] = 0.0037f;

            Tensor activation = new Tensor(new TensorShape(m, k), input);
            for (long i = 0; i < (long)m * k; i++) Store(activation, i, (float)(rng.NextDouble() * 6.0 - 3.0));
            Tensor? biasTensor = null;
            if (bias != "none")
            {
                biasTensor = new Tensor(new TensorShape(n), ParseDType(bias));
                for (long i = 0; i < n; i++) Store(biasTensor, i, (float)(rng.NextDouble() - 0.5));
            }
            return new Operands
            {
                Packed = packed, BlockScale = blockScale, GlobalScale = globalScale, Input = activation, Bias = biasTensor,
                M = m, N = n, OutputDType = output,
            };
        }

        public Tensor AttachResident(DType? denseDType = default)
        {
            if (!Nvfp4Codec.TryAttachResident(Packed, BlockScale, GlobalScale, hasPreQuantScale: false, out Tensor resident,
                    denseDType))
                throw new InvalidOperationException("TryAttachResident refused a weight the resident path must serve.");
            return resident;
        }

        public void Dispose()
        {
            Packed.Dispose();
            BlockScale.Dispose();
            GlobalScale.Dispose();
            Input.Dispose();
            Bias?.Dispose();
        }

        private static void Store(Tensor tensor, long index, float value)
        {
            if (tensor.DType == DType.F32) ((float*)tensor.DataPointer)[index] = value;
            else if (tensor.DType == DType.F16) ((Half*)tensor.DataPointer)[index] = (Half)value;
            else ((ushort*)tensor.DataPointer)[index] = TensorCasts.F32ToBf16Bits(value);
        }
    }
}
