using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>F16-vs-F32 parity for the dit_f16 glue kernels (the Krea2 F16-activation path): RmsNorm,
/// AffineBroadcastLastDim, GatedResidualLastDim, Sigmoid, WanRopeInterleaved, RepeatKvHeads, SliceRows.
/// Each op runs the CUDA F32 kernel (the already-validated reference), then the F16 kernel on the same
/// data cast to F16, comparing at F16 tolerance (inputs bounded [-1,1]; kernels accumulate in F32, so
/// error is I/O rounding only). Skips cleanly when CUDA is unavailable.</summary>
[Collection("CudaSerial")]
public sealed unsafe class DitF16KernelTests
{
    private readonly ITestOutputHelper _output;
    public DitF16KernelTests(ITestOutputHelper output) => _output = output;

    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(dir))
            dir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return dir;
    }

    private static Tensor Random(TensorShape shape, int seed, float lo = -1f, float hi = 1f)
    {
        Tensor t = new Tensor(shape, DType.F32);
        float* p = (float*)t.DataPointer;
        Random rng = new Random(seed);
        long n = shape.ElementCount;
        for (long i = 0; i < n; i++) p[i] = (float)(rng.NextDouble() * (hi - lo) + lo);
        return t;
    }

    /// <summary>Max |a-b| between an F32 reference tensor and an F16 result tensor (host-read).</summary>
    private void AssertClose(Tensor f32Ref, Tensor f16Out, float tol, string name)
    {
        float* a = (float*)f32Ref.DataPointer;
        Half* b = (Half*)f16Out.DataPointer;
        long n = f32Ref.ElementCount;
        float maxDiff = 0f;
        long maxIdx = 0;
        for (long i = 0; i < n; i++)
        {
            float d = MathF.Abs(a[i] - (float)b[i]);
            if (d > maxDiff) { maxDiff = d; maxIdx = i; }
        }
        _output.WriteLine($"{name}: maxDiff={maxDiff:E3} @ {maxIdx} (n={n})");
        Assert.True(maxDiff <= tol, $"{name}: maxDiff {maxDiff:E3} > tol {tol:E1} at index {maxIdx}");
    }

    [Fact]
    public void DitF16Kernels_MatchF32Twins()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }

        const int B = 1, S = 64, H = 8, KvH = 4, D = 32;
        const int hidden = H * D;   // 256
        TensorShape hShape = new TensorShape(B, S, hidden);

        using CudaBackend cuda = new CudaBackend(0, PtxDir());
        IBackend b = cuda;

        // Shared inputs (F32 masters + F16 copies of the activations; params stay F32 in both runs).
        using Tensor x = Random(hShape, 1);
        using Tensor x16 = x.CastTo(DType.F16);
        using Tensor res = Random(hShape, 2);
        using Tensor res16 = res.CastTo(DType.F16);
        using Tensor weight = Random(new TensorShape(hidden), 3, 0.5f, 1.5f);   // rms weight ~1
        using Tensor scaleV = Random(new TensorShape(B, hidden), 4);
        using Tensor shiftV = Random(new TensorShape(B, hidden), 5);
        using Tensor gateV = Random(new TensorShape(B, hidden), 6);

        // ── RmsNorm ──
        using (Tensor o32 = new Tensor(hShape, DType.F32))
        using (Tensor o16 = new Tensor(hShape, DType.F16))
        {
            b.RmsNorm(o32, x, weight, 1e-5f);
            b.RmsNorm(o16, x16, weight, 1e-5f);
            cuda.Sync();
            AssertClose(o32, o16, 6e-3f, "RmsNorm");
        }

        // ── AffineBroadcastLastDim (scale+shift) ──
        using (Tensor o32 = new Tensor(hShape, DType.F32))
        using (Tensor o16 = new Tensor(hShape, DType.F16))
        {
            b.AffineBroadcastLastDim(o32, x, scaleV, shiftV);
            b.AffineBroadcastLastDim(o16, x16, scaleV, shiftV);
            cuda.Sync();
            AssertClose(o32, o16, 6e-3f, "AffineBroadcast");
        }

        // ── GatedResidualLastDim ──
        using (Tensor o32 = new Tensor(hShape, DType.F32))
        using (Tensor o16 = new Tensor(hShape, DType.F16))
        {
            b.GatedResidualLastDim(o32, res, x, gateV);
            b.GatedResidualLastDim(o16, res16, x16, gateV);
            cuda.Sync();
            AssertClose(o32, o16, 8e-3f, "GatedResidual");
        }

        // ── Sigmoid ──
        using (Tensor o32 = new Tensor(hShape, DType.F32))
        using (Tensor o16 = new Tensor(hShape, DType.F16))
        {
            b.Sigmoid(o32, x);
            b.Sigmoid(o16, x16);
            cuda.Sync();
            AssertClose(o32, o16, 2e-3f, "Sigmoid");
        }

        // ── SliceRows (row block copy on rank-3 [B,S,hidden] rows) ──
        using (Tensor o32 = new Tensor(new TensorShape(B, S / 2, hidden), DType.F32))
        using (Tensor o16 = new Tensor(new TensorShape(B, S / 2, hidden), DType.F16))
        {
            b.SliceRows(o32, x, S / 4);
            b.SliceRows(o16, x16, S / 4);
            cuda.Sync();
            AssertClose(o32, o16, 6e-4f, "SliceRows");   // pure copy: only the input's F16 rounding
        }

        // ── WanRopeInterleaved (in-place; [B,S,H,D] layout, cos/sin [S,D] F32) ──
        using (Tensor r32 = x.CastTo(DType.F32))          // fresh copies (op is in-place)
        using (Tensor r16 = x.CastTo(DType.F16))
        using (Tensor cos = Random(new TensorShape(S, D), 7, -1f, 1f))
        using (Tensor sin = Random(new TensorShape(S, D), 8, -1f, 1f))
        {
            b.WanRopeInterleaved(r32, cos, sin, S, H, D);
            b.WanRopeInterleaved(r16, cos, sin, S, H, D);
            cuda.Sync();
            AssertClose(r32, r16, 8e-3f, "RopeInterleaved");
        }

        // ── RepeatKvHeads ([B,KvH,S,D] → [B,H,S,D]) ──
        using (Tensor kv = Random(new TensorShape(B, KvH, S, D), 9))
        using (Tensor kv16 = kv.CastTo(DType.F16))
        using (Tensor o32 = new Tensor(new TensorShape(B, H, S, D), DType.F32))
        using (Tensor o16 = new Tensor(new TensorShape(B, H, S, D), DType.F16))
        {
            b.RepeatKvHeads(o32, kv, KvH, H / KvH);
            b.RepeatKvHeads(o16, kv16, KvH, H / KvH);
            cuda.Sync();
            AssertClose(o32, o16, 6e-4f, "RepeatKvHeads");   // pure copy
        }
    }
}
