using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Hardware validation for the (previously untested) <see cref="CudaGraph"/> capture/replay wrapper — the
/// foundation for collapsing a diffusion denoise step's thousands of per-kernel launches into one <c>cuGraphLaunch</c>
/// (the launch-overhead win for the launch-bound small-token DiT). Confirms on real hardware that capture → instantiate
/// → launch produces the same result as direct execution, and that the backend's stream-ordered activation allocations
/// survive capture as graph-memory nodes with addresses stable across replays (so the tensor→dptr cache set at capture
/// time still reads valid data after a replay). If these hold, wiring the graph into a resident denoise loop is viable.</summary>
[Collection("CudaSerial")]
public sealed class CudaGraphTests
{
    private readonly ITestOutputHelper _output;
    public CudaGraphTests(ITestOutputHelper output) => _output = output;

    private static string ResolvePtxDir()
    {
        string ptxDir = System.IO.Path.Combine(System.AppContext.BaseDirectory, "Ptx");
        if (!System.IO.Directory.Exists(ptxDir))
            ptxDir = System.IO.Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return ptxDir;
    }

    /// <summary>Capture a 4-op elementwise chain (x=1 → ·2 → ·2 → ·2 → ·2 = 16) and replay it; the replayed result
    /// must equal both the direct-execution result and the analytic answer.</summary>
    [Fact]
    public unsafe void CudaGraph_CaptureReplay_Scale_MatchesDirect()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        using CudaBackend backend = new CudaBackend(0, ResolvePtxDir());
        const int n = 64;

        // --- Direct reference ---
        Tensor x = new Tensor(new TensorShape(n), DType.F32);
        float* xp = (float*)x.DataPointer;
        for (int i = 0; i < n; i++) xp[i] = 1f;
        Tensor a = new(x.Shape, DType.F32), b = new(x.Shape, DType.F32), c = new(x.Shape, DType.F32), d = new(x.Shape, DType.F32);
        backend.Scale(a, x, 2f); backend.Scale(b, a, 2f); backend.Scale(c, b, 2f); backend.Scale(d, c, 2f);
        backend.Sync();
        float direct = ((float*)d.DataPointer)[0];
        _output.WriteLine($"direct result = {direct} (expect 16)");

        // --- Graph capture + replay of the same chain ---
        Tensor gx = new Tensor(new TensorShape(n), DType.F32);
        float* gxp = (float*)gx.DataPointer;
        for (int i = 0; i < n; i++) gxp[i] = 1f;
        Tensor ga = new(gx.Shape, DType.F32), gb = new(gx.Shape, DType.F32), gc = new(gx.Shape, DType.F32), gd = new(gx.Shape, DType.F32);
        // Pre-run once so gx is resident and the output buffers exist + are cached before capture.
        backend.Scale(ga, gx, 2f); backend.Scale(gb, ga, 2f); backend.Scale(gc, gb, 2f); backend.Scale(gd, gc, 2f);
        backend.Sync();

        using CudaGraph graph = new CudaGraph(backend.Stream.Handle);
        graph.Capture(() =>
        {
            backend.Scale(ga, gx, 2f); backend.Scale(gb, ga, 2f); backend.Scale(gc, gb, 2f); backend.Scale(gd, gc, 2f);
        });
        Assert.True(graph.IsReady, "graph should be instantiated after Capture");
        graph.Launch();
        backend.Sync();
        float graphResult = ((float*)gd.DataPointer)[0];
        _output.WriteLine($"graph replay result = {graphResult} (expect 16)");

        Assert.Equal(16f, direct, 3);
        Assert.Equal(16f, graphResult, 3);
    }

    /// <summary>DOCUMENTS the wiring blocker for the denoise-loop use case. Capturing the backend's high-level ops
    /// records their per-op stream-ordered <c>cuMemAllocAsync</c> as graph allocation nodes with NO matching free
    /// inside the captured region (the activation is cached and freed later, on Tensor dispose). The first
    /// <c>cuGraphLaunch</c> is correct, but a SECOND launch re-runs the alloc node against graph memory still live
    /// from the first launch → <c>CUDA_ERROR_INVALID_VALUE</c>. The N-step denoise loop needs repeated replay, so
    /// wiring the graph in requires an allocation-free captured region: pre-allocate the activation buffers once
    /// (stable addresses) and have the ops write into them, so the captured graph is pure kernel launches. This test
    /// pins the CURRENT behaviour — it should start failing (and be rewritten to assert stable repeats) once that
    /// persistent-buffer capture path exists.</summary>
    [Fact]
    public unsafe void CudaGraph_RepeatedReplay_WithPerOpAlloc_ThrowsOnSecondLaunch()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        using CudaBackend backend = new CudaBackend(0, ResolvePtxDir());
        const int n = 32;
        Tensor gx = new Tensor(new TensorShape(n), DType.F32);
        float* gxp = (float*)gx.DataPointer;
        for (int i = 0; i < n; i++) gxp[i] = 3f;
        Tensor ga = new(gx.Shape, DType.F32), gb = new(gx.Shape, DType.F32);
        backend.Scale(ga, gx, 2f); backend.Scale(gb, ga, 2f);   // 3 -> 6 -> 12
        backend.Sync();

        using CudaGraph graph = new CudaGraph(backend.Stream.Handle);
        graph.Capture(() => { backend.Scale(ga, gx, 2f); backend.Scale(gb, ga, 2f); });

        // First replay is correct.
        graph.Launch();
        backend.Sync();
        float r0 = ((float*)gb.DataPointer)[0];
        _output.WriteLine($"replay 0: {r0} (expect 12)");
        Assert.Equal(12f, r0, 3);

        // Second replay currently fails: the graph's alloc nodes re-allocate memory still live from replay 0.
        CudaException? ex = Assert.Throws<CudaException>(() =>
        {
            graph.Launch();
            backend.Sync();
        });
        _output.WriteLine($"replay 1 threw (as documented): {ex.Message}");
    }

    /// <summary>THE FIX for the loop case: instantiating with <c>AUTO_FREE_ON_LAUNCH</c> makes the graph free its
    /// previous launch's allocations before relaunching, so capturing the backend's per-op-allocating ops replays
    /// correctly N times — no allocator rewrite needed. Each replay of the (3 → ·2 → ·2 = 12) chain must give 12.</summary>
    [Fact]
    public unsafe void CudaGraph_RepeatedReplay_AutoFreeOnLaunch_IsStable()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        using CudaBackend backend = new CudaBackend(0, ResolvePtxDir());
        const int n = 32;
        Tensor gx = new Tensor(new TensorShape(n), DType.F32);
        float* gxp = (float*)gx.DataPointer;
        for (int i = 0; i < n; i++) gxp[i] = 3f;
        Tensor ga = new(gx.Shape, DType.F32), gb = new(gx.Shape, DType.F32);
        backend.Scale(ga, gx, 2f); backend.Scale(gb, ga, 2f);   // 3 -> 6 -> 12
        backend.Sync();

        using CudaGraph graph = new CudaGraph(backend.Stream.Handle, autoFreeAllocationsOnRelaunch: true);
        graph.Capture(() => { backend.Scale(ga, gx, 2f); backend.Scale(gb, ga, 2f); });
        for (int rep = 0; rep < 4; rep++)
        {
            graph.Launch();
            backend.Sync();
            float r = ((float*)gb.DataPointer)[0];
            _output.WriteLine($"auto-free replay {rep}: {r} (expect 12)");
            Assert.Equal(12f, r, 3);
        }
    }
}
