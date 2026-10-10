using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.ModelAssets.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>
/// A weight group is ONE device allocation holding many weights, each registered at its own offset (the routed experts of
/// one MoE projection). These check what that must not change, and what it must fix: the members compute exactly as
/// individually preloaded weights do, the allocation is freed once and only with its last member, and the driver's
/// per-allocation rounding is gone.
/// </summary>
[Trait("Category", "GpuIntegration")]
[Collection("CudaSerial")]
public sealed unsafe class CudaWeightGroupPreloadTests(ITestOutputHelper output)
{
    private const int Experts = 8;
    private const int Rows = 64;
    private const int Hidden = 256;

    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(dir))
            dir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return dir;
    }

    private bool CudaMissing()
    {
        if (CudaContext.IsAvailable()) return false;
        output.WriteLine("SKIPPED: CUDA unavailable");
        return true;
    }

    private static Tensor RandomF32(long rows, long cols, int seed)
    {
        Tensor t = new(new TensorShape(rows, cols), DType.F32);
        float* p = (float*)t.DataPointer;
        Random rng = new(seed);
        for (long i = 0; i < t.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 2.0 - 1.0) * 0.1f;
        return t;
    }

    /// <summary>A stacked <c>[Experts·Rows, Hidden]</c> quantized tensor, as a GGUF stores a MoE projection.</summary>
    private static Tensor Stacked(DType dtype, int seed)
    {
        using Tensor dense = RandomF32(Experts * Rows, Hidden, seed);
        return GgufQuantizer.Quantize(dense, dtype);
    }

    /// <summary>Per-expert views over the stack: contiguous, back to back, exactly what the GGUF loader produces.</summary>
    private static Tensor[] Views(Tensor stacked) =>
        [.. Enumerable.Range(0, Experts).Select(e => stacked.SliceRows(e * Rows, Rows))];

    /// <summary>Owned copies of the views, so they are distinct tensors that preload one by one.</summary>
    private static Tensor[] Copies(Tensor[] views) =>
        [.. views.Select(v =>
        {
            Tensor copy = new(v.Shape, v.DType);
            long bytes = v.DType.ComputeByteCount(v.ElementCount);
            Buffer.MemoryCopy(v.DataPointer, copy.DataPointer, bytes, bytes);
            return copy;
        })];

    private static float[] Run(CudaBackend cuda, Tensor weight, Tensor input, int m)
    {
        using Tensor output = new(new TensorShape(m, Rows), DType.F32);
        cuda.QuantizedMatMul(output, input, weight, null);
        return output.AsReadOnlySpan<float>().ToArray();
    }

    [Theory]
    [InlineData("Q4_K", 1)]
    [InlineData("Q4_K", 16)]
    [InlineData("Q6_K", 1)]
    [InlineData("Q8_0", 4)]
    public void GroupMembers_ComputeExactlyLikeIndividuallyPreloadedWeights(string dtypeName, int m)
    {
        if (CudaMissing()) return;
        DType dtype = dtypeName switch { "Q4_K" => DType.Q4_K, "Q6_K" => DType.Q6_K, _ => DType.Q8_0 };
        using CudaBackend cuda = new(0, PtxDir());
        using Tensor stacked = Stacked(dtype, seed: 3);
        Tensor[] views = Views(stacked);
        Tensor[] copies = Copies(views);
        using Tensor input = RandomF32(m, Hidden, seed: 7);

        cuda.PreloadWeightGroups([views]);
        cuda.PreloadWeights(copies);
        Assert.Equal(1, cuda.TransferState.WeightGroupCount);

        for (int e = 0; e < Experts; e++)
        {
            Assert.True(GpuTransferHelper.TryGetCachedDevice(views[e], out ulong member));
            Assert.True(cuda.TransferState.IsWeightGroupMember(member));
            Assert.Equal(Run(cuda, copies[e], input, m), Run(cuda, views[e], input, m));
        }
        cuda.FreeWeights(views);
        cuda.FreeWeights(copies);
        foreach (Tensor t in copies) t.Dispose();
    }

    [Fact]
    public void GroupAllocation_IsFreedOnceWithItsLastMember()
    {
        if (CudaMissing()) return;
        using CudaBackend cuda = new(0, PtxDir());
        using Tensor stacked = Stacked(DType.Q4_K, seed: 5);
        Tensor[] views = Views(stacked);

        cuda.PreloadWeightGroups([views]);
        Assert.Equal(1, cuda.TransferState.WeightGroupCount);

        // Half the members leave: the allocation must stay, and the remaining members must still be resident.
        cuda.FreeWeights(views[..(Experts / 2)]);
        Assert.Equal(1, cuda.TransferState.WeightGroupCount);
        for (int e = Experts / 2; e < Experts; e++)
            Assert.True(GpuTransferHelper.TryGetCachedDevice(views[e], out _));

        // A member's interior pointer must never be handed to a caller that would free it on its own.
        Assert.False(GpuTransferHelper.TryUnregisterCachedWeight(views[^1], out ulong refused));
        Assert.Equal(0ul, refused);

        cuda.FreeWeights(views[(Experts / 2)..]);
        Assert.Equal(0, cuda.TransferState.WeightGroupCount);
    }

    [Fact]
    public void Teardown_FreesAGroupStillResident()
    {
        if (CudaMissing()) return;
        using CudaBackend cuda = new(0, PtxDir());
        using Tensor stacked = Stacked(DType.Q8_0, seed: 9);
        cuda.PreloadWeightGroups([Views(stacked)]);
        Assert.Equal(1, cuda.TransferState.WeightGroupCount);
        cuda.FreeAllDeviceMemory();
        Assert.Equal(0, cuda.TransferState.WeightGroupCount);
    }

    /// <summary>The reason this exists: the driver rounds every allocation up, so many expert-sized weights preloaded one by
    /// one take far more memory than their bytes. One group allocation of the same bytes must cost about its size.</summary>
    [Fact]
    public void GroupAllocation_RemovesThePerAllocationRounding()
    {
        if (CudaMissing()) return;
        const int members = 512;
        const long memberBytes = 884_736;   // one Qwen3-30B-A3B Q4_K expert projection
        using CudaBackend cuda = new(0, PtxDir());
        Tensor host = new(new TensorShape(members * memberBytes), DType.U8);
        Tensor[] views = [.. Enumerable.Range(0, members).Select(i => host.SliceRows(i * memberBytes, memberBytes))];
        Tensor[] singles = [.. Enumerable.Range(0, members).Select(_ => new Tensor(new TensorShape(memberBytes), DType.U8))];
        try
        {
            long requested = members * memberBytes;
            (long free0, _) = cuda.GetVramInfo();
            cuda.PreloadWeightGroups([views]);
            (long free1, _) = cuda.GetVramInfo();
            cuda.PreloadWeights(singles);
            (long free2, _) = cuda.GetVramInfo();
            long grouped = free0 - free1;
            long individual = free1 - free2;
            output.WriteLine($"{members} x {memberBytes} B: grouped {grouped / 1e6:F1} MB, one by one {individual / 1e6:F1} MB, requested {requested / 1e6:F1} MB");
            Assert.True(grouped <= requested + (4L << 20), $"Grouped preload took {grouped} bytes for {requested}.");
            Assert.True(individual > grouped, "One-by-one preload was expected to cost more than the group on this driver.");
        }
        finally
        {
            cuda.FreeWeights(views);
            cuda.FreeWeights(singles);
            foreach (Tensor t in singles) t.Dispose();
            host.Dispose();
        }
    }
}
