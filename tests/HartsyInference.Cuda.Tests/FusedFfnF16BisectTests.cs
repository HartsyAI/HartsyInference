using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Op-level bisection for the Ideogram4 fused-FFN F16 defect (fused output degenerate under
/// HARTSY_DIT_F16=1, bit-identical under F32): reproduces the exact production op sequence — fp8 weights
/// with a per-tensor scale, F16 activations, the sandwich-damp on w3 (split) vs whole-w13 damp + gate
/// undamp (fused) — and compares INTERMEDIATES (the fused GEMM's halves vs the split GEMMs) so the first
/// diverging op is identified, not just the end-to-end mismatch.</summary>
[Collection("CudaSerial")]
[Trait("Category", "GpuIntegration")]
public sealed unsafe class FusedFfnF16BisectTests
{
    private readonly ITestOutputHelper _output;
    public FusedFfnF16BisectTests(ITestOutputHelper output) => _output = output;

    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(dir))
            dir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return dir;
    }

    private static Tensor MakeFp8(int rows, int cols, float scale, int seed)
    {
        Tensor f32 = new Tensor(new TensorShape(rows, cols), DType.F32);
        float* p = (float*)f32.DataPointer;
        Random rng = new Random(seed);
        for (int i = 0; i < rows * cols; i++) p[i] = (float)(rng.NextDouble() * 2.0 - 1.0) / scale;
        Tensor fp8 = f32.CastTo(DType.F8E4M3);
        f32.Dispose();
        fp8.Fp8ScaleFactor = scale;
        return fp8;
    }

    private static Tensor MakeF16(int b, int l, int k, int seed)
    {
        Tensor t = new Tensor(new TensorShape(b, l, k), DType.F16);
        ushort* p = (ushort*)t.DataPointer;
        Random rng = new Random(seed);
        for (long i = 0; i < t.ElementCount; i++) p[i] = BitConverter.HalfToUInt16Bits((Half)(rng.NextDouble() * 2.0 - 1.0));
        return t;
    }

    private static float[] ToF32(Tensor t)
    {
        Tensor f = t.DType == DType.F32 ? t : t.CastTo(DType.F32);
        float[] r = new float[f.ElementCount];
        float* p = (float*)f.DataPointer;
        for (long i = 0; i < r.Length; i++) r[i] = p[i];
        if (!ReferenceEquals(f, t)) f.Dispose();
        return r;
    }

    private static double MaxRelDiff(float[] a, float[] b)
    {
        double m = 0;
        for (int i = 0; i < a.Length; i++)
        {
            double denom = Math.Max(1e-3, Math.Abs(b[i]));
            m = Math.Max(m, Math.Abs(a[i] - b[i]) / denom);
        }
        return m;
    }

    [Fact]
    public void FusedW13_F16Damped_HalvesMatchSplitGemms()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }

        const int B = 1, L = 64, Hidden = 128, Inner = 96;
        const float Damp = 1.0f / 64.0f;

        using Tensor w1 = MakeFp8(Inner, Hidden, scale: 2.0f, seed: 1);
        using Tensor w3 = MakeFp8(Inner, Hidden, scale: 2.0f, seed: 2);
        using Tensor x = MakeF16(B, L, Hidden, seed: 3);

        // fused = [w1; w3] with the shared scale, then the WHOLE tensor damped (the production F16 branch)
        Tensor w13 = HartsyInference.ModelAssets.CheckpointConverters.Utils.CheckpointConvertUtils.ConcatRowsHost(w1, w3);
        w13.Fp8ScaleFactor *= Damp;
        // split reference: w3 damped only (production split branch)
        w3.Fp8ScaleFactor *= Damp;

        using CudaBackend cuda = new CudaBackend(0, PtxDir());

        // Split GEMMs
        using Tensor gateRef = new Tensor(new TensorShape(B, L, Inner), DType.F16);
        using Tensor upRef = new Tensor(new TensorShape(B, L, Inner), DType.F16);
        cuda.Linear(gateRef, x, w1, null);       // undamped gate
        cuda.Linear(upRef, x, w3, null);         // damped up
        cuda.Sync();

        // Fused GEMM + slices + gate undamp
        using Tensor both = new Tensor(new TensorShape(B, L, 2 * Inner), DType.F16);
        cuda.Linear(both, x, w13, null);
        using Tensor gateF = new Tensor(new TensorShape(B, L, Inner), DType.F16);
        using Tensor upF = new Tensor(new TensorShape(B, L, Inner), DType.F16);
        cuda.SliceLastDim(gateF, both, 0);
        cuda.SliceLastDim(upF, both, Inner);
        using Tensor gateUndamped = new Tensor(new TensorShape(B, L, Inner), DType.F16);
        cuda.Scale(gateUndamped, gateF, 1.0f / Damp);
        cuda.Sync();

        float[] gr = ToF32(gateRef);
        float[] ur = ToF32(upRef);
        float[] gf = ToF32(gateUndamped);
        float[] gfRaw = ToF32(gateF);
        float[] uf = ToF32(upF);

        double dUp = MaxRelDiff(uf, ur);
        double dGate = MaxRelDiff(gf, gr);
        _output.WriteLine($"up-half maxRelDiff  = {dUp:E3}");
        _output.WriteLine($"gate (undamped) maxRelDiff = {dGate:E3}");
        _output.WriteLine($"sample: gateRef[0]={gr[0]:G6} fusedUndamped[0]={gf[0]:G6} fusedRaw[0]={gfRaw[0]:G6}");
        Assert.True(dUp < 0.05, $"up half diverges: {dUp:E3}");
        Assert.True(dGate < 0.05, $"gate half diverges: {dGate:E3}");
        w13.Dispose();
    }
}
