using HartsyInference.Core.Rope;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Tests.Common;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>FluxRope's device tables now live in a RopeTableCache. These tests pin preload counts, the retirement order and
/// the duplicated-pair content through a recording backend.</summary>
public sealed unsafe class FluxRopeDeviceTableTests
{
    private static readonly float[] Positions = [0.5f, -1f, 2f, 3f];

    [Fact]
    public void ApplyGpu_PreloadsOncePerSignature_AndReusesTheTables()
    {
        using FluxRope rope = new([4, 4], 10000);
        RopeRecordingBackend backend = new();
        rope.Precompute(PositionTensor(Positions, 2, 2));
        Tensor q = Zeros(2, 8);
        Tensor k = Zeros(2, 8);

        rope.ApplyGpu(backend, q, k, 1);
        rope.ApplyGpu(backend, q, k, 1);

        Assert.Equal(new[] { "preload:2" }, backend.Calls);
        Dispose(q, k);
    }

    [Fact]
    public void DeviceTables_HoldTheDuplicatedPairsOfTheHostTables()
    {
        using FluxRope rope = new([4, 4], 10000);
        RopeRecordingBackend backend = new();
        rope.Precompute(PositionTensor(Positions, 2, 2));
        Tensor q = Zeros(2, 8);
        Tensor k = Zeros(2, 8);

        rope.ApplyGpu(backend, q, k, 1);

        float[] expectedCos = new float[16];
        float[] expectedSin = new float[16];
        MultiAxisRopeTables.BuildRows(RopeFrequencyPlan.Create(new RopeSpec { AxisDims = [4, 4], Theta = 10000 }), Positions, 2,
            RopeAngleOrder.MultiplyByInverse, RopeTableLayout.InterleavedPair, 0, RopeTablePadding.Identity, expectedCos, expectedSin);
        Tensor device = backend.PreloadedBatches[0][0];
        Tensor deviceSin = backend.PreloadedBatches[0][1];
        ReadOnlySpan<float> actualCos = new((void*)device.DataPointer, 16);
        ReadOnlySpan<float> actualSin = new((void*)deviceSin.DataPointer, 16);
        for (int i = 0; i < 16; i++)
        {
            Assert.Equal(BitConverter.SingleToInt32Bits(expectedCos[i]), BitConverter.SingleToInt32Bits(actualCos[i]));
            Assert.Equal(BitConverter.SingleToInt32Bits(expectedSin[i]), BitConverter.SingleToInt32Bits(actualSin[i]));
        }
        Dispose(q, k);
    }

    [Fact]
    public void Precompute_NewSignature_RetiresOldTablesThroughBeforeRetireFirst()
    {
        RopeRecordingBackend backend = new();
        using FluxRope rope = new([4, 4], 10000, b => ((RopeRecordingBackend)b).Calls.Add("beforeRetire"));
        Tensor q = Zeros(2, 8);
        Tensor k = Zeros(2, 8);
        rope.Precompute(PositionTensor(Positions, 2, 2));
        rope.ApplyGpu(backend, q, k, 1);

        rope.Precompute(PositionTensor([1f, 1f, 1f, 1f], 2, 2));
        rope.ApplyGpu(backend, q, k, 1);

        Assert.Equal(new[] { "preload:2", "beforeRetire", "free:2", "preload:2" }, backend.Calls);
        Dispose(q, k);
    }

    [Fact]
    public void Dispose_InvalidatesThenFreesDeviceTables()
    {
        RopeRecordingBackend backend = new();
        FluxRope rope = new([4, 4], 10000, b => ((RopeRecordingBackend)b).Calls.Add("beforeRetire"));
        rope.Precompute(PositionTensor(Positions, 2, 2));
        Tensor q = Zeros(2, 8);
        Tensor k = Zeros(2, 8);
        rope.ApplyGpu(backend, q, k, 1);

        rope.Dispose();

        Assert.Equal(new[] { "preload:2", "beforeRetire", "free:2" }, backend.Calls);
        Dispose(q, k);
    }

    [Fact]
    public void Precompute_FailedRetirement_LeavesThePreviousSignaturePublished()
    {
        RopeRecordingBackend backend = new();
        bool failRetire = false;
        using FluxRope rope = new([4, 4], 10000, _ =>
        {
            if (failRetire)
                throw new InvalidOperationException("retirement failed");
        });
        Tensor q = Zeros(2, 8);
        Tensor k = Zeros(2, 8);
        rope.Precompute(PositionTensor(Positions, 2, 2));
        rope.ApplyGpu(backend, q, k, 1);

        failRetire = true;
        Assert.Throws<InvalidOperationException>(() => rope.Precompute(PositionTensor([1f, 1f, 1f, 1f, 1f, 1f], 3, 2)));
        failRetire = false;

        // Still the first signature: two tokens, and its own rows are what the next apply preloads.
        rope.ApplyGpu(backend, q, k, 1);
        float[] expectedCos = new float[16];
        float[] expectedSin = new float[16];
        MultiAxisRopeTables.BuildRows(RopeFrequencyPlan.Create(new RopeSpec { AxisDims = [4, 4], Theta = 10000 }), Positions, 2,
            RopeAngleOrder.MultiplyByInverse, RopeTableLayout.InterleavedPair, 0, RopeTablePadding.Identity, expectedCos, expectedSin);
        Tensor rebuilt = backend.PreloadedBatches[^1][0];
        Assert.Equal(expectedCos, new ReadOnlySpan<float>((void*)rebuilt.DataPointer, 16).ToArray());
        Dispose(q, k);
    }

    [Fact]
    public void ApplyGpu_AfterDispose_IsRejected()
    {
        RopeRecordingBackend backend = new();
        FluxRope rope = new([4, 4], 10000);
        rope.Precompute(PositionTensor(Positions, 2, 2));
        rope.Dispose();
        Tensor q = Zeros(2, 8);
        Tensor k = Zeros(2, 8);

        Assert.Throws<ObjectDisposedException>(() => rope.ApplyGpu(backend, q, k, 1));
        Dispose(q, k);
    }

    private static Tensor PositionTensor(float[] values, int tokens, int numAxes)
    {
        Tensor tensor = new(new TensorShape(tokens, numAxes), DType.F32);
        float* ptr = (float*)tensor.DataPointer;
        for (int i = 0; i < values.Length; i++) ptr[i] = values[i];
        return tensor;
    }

    private static Tensor Zeros(int tokens, int headDim) => new(new TensorShape(1, tokens, 1, headDim), DType.F32);

    private static void Dispose(params Tensor[] tensors)
    {
        foreach (Tensor tensor in tensors) tensor.Dispose();
    }
}
