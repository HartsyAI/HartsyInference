using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>`LtxVideo25NeighborhoodAttention3d.Forward` RMS-norms q and k through a <c>Reshape</c> view held in a
/// <c>using</c> block, then reads the parent tensor. That is the shape of the `ApplyKeyframesAbsPos` defect already
/// found in this model — the result landing in the view's own device cache, which <c>Dispose</c> frees without a
/// write-back, so it works on <see cref="CpuBackend"/> and silently drops on CUDA. This pins whether the parent
/// actually observes an in-place op performed on a view of it.</summary>
[Collection("CudaSerial")]
public sealed unsafe class Ltx25NaRmsNormViewTests
{
    private readonly ITestOutputHelper _output;
    public Ltx25NaRmsNormViewTests(ITestOutputHelper output) => _output = output;

    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(dir))
            dir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return dir;
    }

    private static Tensor Filled(TensorShape shape, int seed)
    {
        Tensor t = new Tensor(shape, DType.F32);
        float* p = (float*)t.DataPointer;
        Random rng = new Random(seed);
        for (long i = 0; i < shape.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }

    private static Tensor NormDirect(IBackend backend, TensorShape headShape, int headDim, int seed)
    {
        Tensor x = Filled(headShape, seed);
        using Tensor w = Filled(new TensorShape(headDim), seed + 100);
        backend.RmsNorm(x, x, w, 1e-6f);
        return x;
    }

    private static Tensor NormThroughView(IBackend backend, TensorShape headShape, long rows, int headDim, int seed)
    {
        Tensor x = Filled(headShape, seed);
        using (Tensor w = Filled(new TensorShape(headDim), seed + 100))
        {
            using (Tensor view = x.Reshape(new TensorShape(rows, headDim)))
                backend.RmsNorm(view, view, w, 1e-6f);
        }
        return x;
    }

    /// <summary>The form the decoder uses: RmsNorm straight on the rank-6 tensor. It rows by the last dim, so this
    /// is the same arithmetic the view was there to express, and CUDA must match CPU exactly.</summary>
    [Fact]
    public void RmsNormDirectlyOnTheRank6TensorMatchesCpu()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        const int t = 2, h = 3, w = 2, heads = 2, headDim = 8;
        TensorShape headShape = new TensorShape([1, t, h, w, heads, headDim]);

        IBackend cpu = new CpuBackend();
        using Tensor expected = NormDirect(cpu, headShape, headDim, seed: 7);
        using CudaBackend cuda = new CudaBackend(0, PtxDir());
        using Tensor actual = NormDirect(cuda, headShape, headDim, seed: 7);
        cuda.Sync();

        float* e = (float*)expected.DataPointer;
        float* a = (float*)actual.DataPointer;
        double maxErr = 0;
        for (long i = 0; i < headShape.ElementCount; i++) maxErr = Math.Max(maxErr, Math.Abs((double)e[i] - a[i]));
        Assert.True(maxErr <= 1e-5, $"direct rank-6 RmsNorm diverges from CPU by {maxErr:E3}");
    }

    /// <summary>Why the decoder must NOT go back to the view form. This asserts the hazard is still real rather than
    /// asserting it is fixed: an in-place CUDA op on a <c>Reshape</c> view does not reach the parent, so the two
    /// backends disagree. If this ever starts failing, the backend gained view write-back and this test — plus the
    /// comment in <c>LtxVideo25NeighborhoodAttention3d.Forward</c> — should be retired.</summary>
    [Fact]
    public void InPlaceOpThroughAReshapeViewStillDoesNotReachTheParentOnCuda()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        const int t = 2, h = 3, w = 2, heads = 2, headDim = 8;
        TensorShape headShape = new TensorShape([1, t, h, w, heads, headDim]);
        long rows = (long)t * h * w * heads;

        IBackend cpu = new CpuBackend();
        using Tensor viaCpu = NormThroughView(cpu, headShape, rows, headDim, seed: 7);
        using CudaBackend cuda = new CudaBackend(0, PtxDir());
        using Tensor viaCuda = NormThroughView(cuda, headShape, rows, headDim, seed: 7);
        cuda.Sync();

        float* e = (float*)viaCpu.DataPointer;
        float* a = (float*)viaCuda.DataPointer;
        double maxErr = 0;
        for (long i = 0; i < headShape.ElementCount; i++) maxErr = Math.Max(maxErr, Math.Abs((double)e[i] - a[i]));
        Assert.True(maxErr > 1e-3,
            "an in-place CUDA op through a Reshape view now reaches the parent — the hazard this guards is gone.");
    }
}
