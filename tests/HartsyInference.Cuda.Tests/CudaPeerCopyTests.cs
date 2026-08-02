using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Cross-backend boundary-copy contract (<see cref="IBackend.CopyFromPeer"/>): data lands intact on the
/// destination backend whichever path ran (direct P2P, no-P2P device→host staging, or plain host fallback), and
/// the SOURCE backend's resident copy survives — the whole point over a naive host round-trip. The dev box's
/// mismatched consumer pair typically reports no P2P, which exercises the staging path; boxes with NVLink/P2P
/// exercise the direct path and bump <see cref="IBackend.GetPeerCopyCount"/>.</summary>
[Collection("CudaSerial")]
public sealed unsafe class CudaPeerCopyTests
{
    private readonly ITestOutputHelper _output;
    public CudaPeerCopyTests(ITestOutputHelper output) => _output = output;

    private static string PtxDir()
    {
        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDir))
            ptxDir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return ptxDir;
    }

    private static Tensor RandomF32(TensorShape shape, int seed)
    {
        Tensor t = new(shape, DType.F32);
        Random rng = new(seed);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return t;
    }

    /// <summary>Produces a device-resident activation on <paramref name="backend"/> (never read on the host, so
    /// its only authoritative copy is the device one) and the CPU reference for its first element.</summary>
    private static (Tensor Activation, float Expected) ProduceActivation(CudaBackend backend, int seed)
    {
        const int dim = 64;
        TensorShape ioShape = new(2, dim);
        TensorShape wShape = new(dim, dim);
        using Tensor input = RandomF32(ioShape, seed);
        using Tensor weight = RandomF32(wShape, seed + 1);
        float* ip = (float*)input.DataPointer;
        float* wp = (float*)weight.DataPointer;
        double acc = 0;
        for (int k = 0; k < dim; k++) acc += ip[k] * wp[k];

        backend.HighPrecisionGemm = true;
        Tensor activation = new(ioShape, DType.F32);
        backend.PreloadWeights(new[] { weight });
        backend.Linear(activation, input, weight, bias: null);
        backend.FreeWeights(new[] { weight });
        return (activation, (float)acc);
    }

    [Fact]
    public void CrossDevice_BoundaryCopy_LandsIntact_AndSourceStaysResident()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        if (CudaContext.GetDeviceCount() < 2) { _output.WriteLine("SKIPPED: needs 2 GPUs"); return; }

        CudaBackend src = new(0, PtxDir());
        CudaBackend dst = new(1, PtxDir());
        try
        {
            (Tensor activation, float expected) = ProduceActivation(src, 501);
            using Tensor moved = new(activation.Shape, DType.F32);

            long peerBefore = dst.GetPeerCopyCount();
            dst.CopyFromPeer(moved, activation, src);
            long peerAfter = dst.GetPeerCopyCount();
            _output.WriteLine(peerAfter > peerBefore
                ? "Direct P2P path taken."
                : "No P2P on this pair — device→host staging path taken.");

            // Destination holds the data (device-resident on the P2P path — this read fires dst's lazy sync).
            Assert.Equal(expected, ((float*)moved.DataPointer)[0], 3);

            // The source backend's device copy must SURVIVE the boundary copy — CopyFromPeer exists precisely to
            // avoid the interface default's demote-on-host-read.
            Assert.True(src.TransferState.ActivationCache.ContainsKey(activation),
                "the boundary copy evicted the source backend's resident activation");

            activation.Dispose();
        }
        finally
        {
            src.Dispose();
            dst.Dispose();
        }
    }

    /// <summary>Same-device sibling backends never report P2P (same ordinal), so the staging path must carry the
    /// handoff — runnable on a one-GPU box.</summary>
    [Fact]
    public void SameDevice_BoundaryCopy_StagesThroughDestinationHostBuffer()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }

        CudaBackend src = new(0, PtxDir());
        CudaBackend dst = new(0, PtxDir());
        try
        {
            (Tensor activation, float expected) = ProduceActivation(src, 601);
            using Tensor moved = new(activation.Shape, DType.F32);

            dst.CopyFromPeer(moved, activation, src);
            Assert.Equal(0, dst.GetPeerCopyCount());
            Assert.Equal(expected, ((float*)moved.DataPointer)[0], 3);
            Assert.True(src.TransferState.ActivationCache.ContainsKey(activation),
                "the boundary copy evicted the source backend's resident activation");

            activation.Dispose();
        }
        finally
        {
            src.Dispose();
            dst.Dispose();
        }
    }

    /// <summary>A source with no device shadow (host-only data) rides the plain host fallback.</summary>
    [Fact]
    public void HostOnlySource_FallsBackToPlainCopy()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }

        CudaBackend src = new(0, PtxDir());
        CudaBackend dst = new(0, PtxDir());
        try
        {
            using Tensor hostOnly = RandomF32(new TensorShape(4, 16), 701);
            float expected = ((float*)hostOnly.DataPointer)[0];
            using Tensor moved = new(hostOnly.Shape, DType.F32);
            dst.CopyFromPeer(moved, hostOnly, src);
            Assert.Equal(expected, ((float*)moved.DataPointer)[0], 6);
        }
        finally
        {
            src.Dispose();
            dst.Dispose();
        }
    }
}
