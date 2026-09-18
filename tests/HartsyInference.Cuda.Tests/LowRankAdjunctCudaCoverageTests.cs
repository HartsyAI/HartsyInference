using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Gguf;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>The CUDA half of the LoRA-adjunct coverage ledger: every GEMM entry a packed weight can reach on this
/// backend, each pinned against the dense merge it must reproduce.
/// <para>CUDA is where this actually runs in production — it is the only backend with resident quantized GEMMs, so it
/// is the only one where a LoRA'd block-quantized weight reaches a kernel at all. It is also where the entries
/// multiply: <c>Linear</c> alone returns early from a dozen decode-shaped fused GEMVs, <c>LinearGelu</c> folds the
/// activation into the dequant epilogue, <c>LinearMulti</c> takes the grouped int8 chain past <c>LinearImpl</c>
/// entirely, and a captured step graph replays whatever the capture recorded. A miss in any one of them is a
/// generation that succeeds with part of the LoRA applied.</para></summary>
[Collection("CudaSerial")]
[Trait("Category", "GpuIntegration")]
public sealed unsafe class LowRankAdjunctCudaCoverageTests
{
    private const int Rows = 256, Cols = 256, Rank = 8, Batch = 32;
    private const float DownValue = 0.02f, UpValue = 0.01f;

    private readonly ITestOutputHelper _output;

    public LowRankAdjunctCudaCoverageTests(ITestOutputHelper output) => _output = output;

    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(dir))
            dir = Path.Combine(RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return dir;
    }

    /// <remarks>The quants here are the ones our own quantizer can PRODUCE. The classic codecs a community H3 or Wan
    /// build actually ships (Q4_0/Q5_0/Q5_1) cannot be written at all, which is precisely why their LoRA has to ride
    /// on the weight rather than merge into it; covering them needs a real downloaded file, not a synthetic one.</remarks>
    [Theory]
    [InlineData("Q8_0")]
    [InlineData("Q4_K")]
    [InlineData("Q6_K")]
    public void Linear_OnABlockQuantizedBase_MatchesTheDequantizedMerge(string quant)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: no CUDA device"); return; }
        using CudaBackend backend = new CudaBackend(0, PtxDir());
        using Fixture fixture = new Fixture(QuantOf(quant), DType.F16);

        using Tensor actual = new Tensor(new TensorShape(Batch, Rows), DType.F16);
        backend.Linear(actual, fixture.Input, fixture.Patched, null);
        using Tensor expected = new Tensor(new TensorShape(Batch, Rows), DType.F16);
        backend.Linear(expected, fixture.Input, fixture.Merged, null);

        AssertClose(backend, expected, actual, tolerance: 0.02f);
    }

    /// <summary>The decode-shaped route: m=1 with F32 activations returns from a fused GEMV long before the general
    /// GEMM, so an adjunct added inside that branch would be skipped for exactly the shape LLM decode uses.</summary>
    [Fact]
    public void FusedDecodeGemv_AppliesTheAdjunct()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: no CUDA device"); return; }
        using CudaBackend backend = new CudaBackend(0, PtxDir());
        using Fixture fixture = new Fixture(DType.Q8_0, DType.F32, batch: 1);

        using Tensor actual = new Tensor(new TensorShape(1, Rows), DType.F32);
        backend.Linear(actual, fixture.Input, fixture.Patched, null);
        using Tensor expected = new Tensor(new TensorShape(1, Rows), DType.F32);
        backend.Linear(expected, fixture.Input, fixture.Merged, null);

        AssertClose(backend, expected, actual, tolerance: 0.02f);
    }

    /// <summary>MiniMax-H3's chunked projections: the window narrows the up matrix and keeps the down matrix whole.</summary>
    [Fact]
    public void LinearWeightRows_AppliesTheWindowedAdjunct()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: no CUDA device"); return; }
        using CudaBackend backend = new CudaBackend(0, PtxDir());
        using Fixture fixture = new Fixture(DType.Q8_0, DType.F16);

        const int offset = 64, count = 96;
        using Tensor actual = new Tensor(new TensorShape(Batch, count), DType.F16);
        backend.LinearWeightRows(actual, fixture.Input, fixture.Patched, null, offset, count);
        using Tensor full = new Tensor(new TensorShape(Batch, Rows), DType.F16);
        backend.Linear(full, fixture.Input, fixture.Merged, null);

        backend.Sync();
        ReadOnlySpan<Half> a = new ReadOnlySpan<Half>((void*)actual.DataPointer, Batch * count);
        ReadOnlySpan<Half> f = new ReadOnlySpan<Half>((void*)full.DataPointer, Batch * Rows);
        for (int row = 0; row < Batch; row++)
        {
            for (int column = 0; column < count; column++)
            {
                Assert.Equal((float)f[(row * Rows) + offset + column], (float)a[(row * count) + column], 2);
            }
        }
    }

    /// <summary>The fused-epilogue entry. GELU(base + delta) is not GELU(base) + delta, so the backend must un-fuse
    /// the activation when the weight carries an adjunct rather than adding it after.</summary>
    [Fact]
    public void LinearGelu_AppliesTheAdjunctBeforeTheActivation()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: no CUDA device"); return; }
        using CudaBackend backend = new CudaBackend(0, PtxDir());
        using Fixture fixture = new Fixture(Int8ConvRotWeight(), DType.F16);

        using Tensor actual = new Tensor(new TensorShape(Batch, Rows), DType.F16);
        backend.LinearGelu(actual, fixture.Input, fixture.Patched, null);
        using Tensor expected = new Tensor(new TensorShape(Batch, Rows), DType.F16);
        backend.LinearGelu(expected, fixture.Input, fixture.Merged, null);

        AssertClose(backend, expected, actual, tolerance: 0.03f);
    }

    /// <summary>The grouped resident-int8 chain bypasses LinearImpl for every eligible op, so an adjunct-carrying op
    /// has to fall out of the group — and the ops that stay in it must be unaffected.</summary>
    [Fact]
    public void LinearMulti_AppliesTheAdjunctOnlyToTheOpThatCarriesOne()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: no CUDA device"); return; }
        using CudaBackend backend = new CudaBackend(0, PtxDir());
        using Fixture fixture = new Fixture(Int8ConvRotWeight(), DType.F16);
        using Tensor plainWeight = Int8ConvRotWeight();

        using Tensor patchedOut = new Tensor(new TensorShape(Batch, Rows), DType.F16);
        using Tensor plainOut = new Tensor(new TensorShape(Batch, Rows), DType.F16);
        backend.LinearMulti(fixture.Input,
        [
            new LinearOp(patchedOut, fixture.Patched, null),
            new LinearOp(plainOut, plainWeight, null),
        ]);

        using Tensor expectedPatched = new Tensor(new TensorShape(Batch, Rows), DType.F16);
        backend.Linear(expectedPatched, fixture.Input, fixture.Merged, null);
        using Tensor expectedPlain = new Tensor(new TensorShape(Batch, Rows), DType.F16);
        backend.Linear(expectedPlain, fixture.Input, plainWeight, null);

        AssertClose(backend, expectedPatched, patchedOut, tolerance: 0.03f);
        AssertClose(backend, expectedPlain, plainOut, tolerance: 0.0f);
    }

    /// <summary>YuE2's low-VRAM AR path reaches the GEMM through QuantizedMatMul rather than Linear.</summary>
    [Fact]
    public void QuantizedMatMul_AppliesTheAdjunct()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: no CUDA device"); return; }
        using CudaBackend backend = new CudaBackend(0, PtxDir());
        using Fixture fixture = new Fixture(DType.Q8_0, DType.F16);

        using Tensor actual = new Tensor(new TensorShape(Batch, Rows), DType.F16);
        backend.QuantizedMatMul(actual, fixture.Input, fixture.Patched, null);
        using Tensor expected = new Tensor(new TensorShape(Batch, Rows), DType.F16);
        backend.Linear(expected, fixture.Input, fixture.Merged, null);

        AssertClose(backend, expected, actual, tolerance: 0.02f);
    }

    /// <summary>A captured DiT step replays whatever the capture recorded, so the adjunct's own launches have to be
    /// inside the capture — and the factors resident before it, since a host-to-device copy is not a replayable node.</summary>
    [Fact]
    public void CapturedStepGraph_ReplaysTheAdjunct()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: no CUDA device"); return; }
        using CudaBackend backend = new CudaBackend(0, PtxDir());
        if (!backend.StepGraphSupported) { _output.WriteLine("SKIPPED: no step-graph support"); return; }
        using Fixture fixture = new Fixture(DType.Q8_0, DType.F16);
        // Both the weight AND its adjunct factors must be resident before capture — a host-to-device copy of a
        // non-resident tensor inside a capture is not a replayable node. PreloadWeights follows the factors for
        // exactly this reason.
        backend.PreloadWeights([fixture.Patched]);

        // The activation has to be device-resident too, which is why the capture reads a warmed-up copy of it
        // rather than the host tensor (the pattern every graph owner in this suite follows).
        using Tensor residentInput = new Tensor(fixture.Input.Shape, DType.F16);
        backend.Scale(residentInput, fixture.Input, 1.0f);
        using Tensor output = new Tensor(new TensorShape(Batch, Rows), DType.F16);
        backend.Linear(output, residentInput, fixture.Patched, null);
        backend.Sync();

        backend.StepGraphBegin();
        backend.Linear(output, residentInput, fixture.Patched, null);
        backend.StepGraphEndAndLaunch();
        backend.Sync();
        // Reading the output would consume the device buffer the graph baked an address for, so drop the graph first.
        backend.StepGraphReset();

        using Tensor expected = new Tensor(new TensorShape(Batch, Rows), DType.F16);
        backend.Linear(expected, fixture.Input, fixture.Merged, null);
        AssertClose(backend, expected, output, tolerance: 0.02f);
        backend.FreeWeights([fixture.Patched]);
    }

    private static DType QuantOf(string name) => name switch
    {
        "Q8_0" => DType.Q8_0,
        "Q4_K" => DType.Q4_K,
        "Q6_K" => DType.Q6_K,
        _ => throw new ArgumentOutOfRangeException(nameof(name), name, "Unknown quant in the coverage table."),
    };

    /// <summary>A ConvRot-free int8_tensorwise weight — the format that keeps the resident IMMA path, which is what
    /// the fused-GELU and grouped entries exercise.</summary>
    private static Tensor Int8ConvRotWeight()
    {
        Random rng = new Random(20260918);
        Tensor w = new Tensor(new TensorShape(Rows, Cols), DType.I8);
        sbyte* p = (sbyte*)w.DataPointer;
        for (long i = 0; i < (long)Rows * Cols; i++) p[i] = (sbyte)rng.Next(-127, 128);
        Tensor rowScale = new Tensor(new TensorShape(Rows), DType.F32);
        float* s = (float*)rowScale.DataPointer;
        for (int i = 0; i < Rows; i++) s[i] = 0.003f;
        w.QuantInfo = new QuantWeightInfo { Format = "int8_tensorwise", RowScale = rowScale, ConvRotGroupSize = 0 };
        return w;
    }

    private static void AssertClose(CudaBackend backend, Tensor expected, Tensor actual, float tolerance)
    {
        backend.Sync();
        int count = (int)expected.ElementCount;
        for (int i = 0; i < count; i++)
        {
            float e = Read(expected, i), a = Read(actual, i);
            Assert.True(MathF.Abs(e - a) <= tolerance + (tolerance * MathF.Abs(e)),
                $"element {i}: expected {e}, got {a}");
        }
    }

    private static float Read(Tensor t, int index) => t.DType == DType.F32
        ? ((float*)t.DataPointer)[index]
        : (float)((Half*)t.DataPointer)[index];

    /// <summary>A packed base weight, the same weight carrying a one-term adjunct, and the dense merge of the two.</summary>
    private sealed class Fixture : IDisposable
    {
        private readonly List<Tensor> _owned = [];

        public Fixture(DType quant, DType activation, int batch = Batch)
            : this(Quantize(quant), activation, batch)
        {
        }

        public Fixture(Tensor packed, DType activation, int batch = Batch)
        {
            _owned.Add(packed);
            Input = Ramp(batch, Cols, activation);
            _owned.Add(Input);

            Tensor down = Filled(Rank, Cols, DownValue);
            Tensor up = Filled(Rows, Rank, UpValue);
            _owned.Add(down);
            _owned.Add(up);
            Patched = packed.WithLowRankAdjunct(new LowRankAdjunct
            {
                Terms = [new LowRankAdjunctTerm { Down = down, Up = up, Scale = 1.0f }],
            });
            _owned.Add(Patched);

            using Tensor dense = Dequantize(packed);
            Tensor merged = dense.CastTo(DType.F32);
            Span<float> values = merged.AsSpan<float>();
            float delta = Rank * DownValue * UpValue;
            for (int i = 0; i < values.Length; i++) values[i] += delta;
            Merged = merged.CastTo(activation == DType.F32 ? DType.F32 : DType.F16);
            merged.Dispose();
            _owned.Add(Merged);
        }

        public Tensor Input { get; }

        public Tensor Patched { get; }

        public Tensor Merged { get; }

        private static Tensor Quantize(DType quant)
        {
            using Tensor dense = Ramp(Rows, Cols, DType.F32);
            return GgufQuantizer.Quantize(dense, quant);
        }

        private static Tensor Dequantize(Tensor packed)
        {
            if (packed.DType.IsQuantized)
            {
                return GgufDequantizer.Dequantize(packed, DType.F32);
            }
            QuantWeightInfo info = packed.QuantInfo!;
            using Tensor bf16 = Core.Tensors.Int8ConvRotCodec.DequantToBf16(packed, info.RowScale!, info.ConvRotGroupSize);
            return bf16.CastTo(DType.F32);
        }

        private static Tensor Ramp(long rows, long cols, DType dtype)
        {
            Tensor f32 = new Tensor(new TensorShape(rows, cols), DType.F32);
            Span<float> values = f32.AsSpan<float>();
            for (int i = 0; i < values.Length; i++) values[i] = MathF.Sin(i * 0.013f) * 0.5f;
            if (dtype == DType.F32) return f32;
            Tensor cast = f32.CastTo(dtype);
            f32.Dispose();
            return cast;
        }

        private static Tensor Filled(long rows, long cols, float value)
        {
            Tensor t = new Tensor(new TensorShape(rows, cols), DType.F32);
            t.AsSpan<float>().Fill(value);
            return t;
        }

        public void Dispose()
        {
            for (int i = _owned.Count - 1; i >= 0; i--)
            {
                _owned[i].Dispose();
            }
        }
    }
}
