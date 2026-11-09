using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Streaming;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Streaming state held narrower than it arrives: <see cref="StreamingTensors"/> slices and joins 16-bit tensors element
/// for element on the CPU backend (whose <c>Concat</c> writes four bytes an element), and a <see cref="RollingTokenWindow"/> or
/// <see cref="FrameTailCache"/> with a BF16 storage dtype keeps the BF16 rounding of what it was given, cut and ordered as the
/// F32 version would be.</summary>
public sealed class StreamingBf16StorageTests
{
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    public void Concat_OfBf16Tensors_JoinsElementForElement(int axis)
    {
        using CpuBackend cpu = new();
        long[] dimsA = [2, 3, 4], dimsB = [2, 3, 4];
        dimsB[axis] = axis + 1;
        using Tensor a = Bf16Ramp(dimsA, 0), b = Bf16Ramp(dimsB, 1000);
        using Tensor a32 = a.CastTo(DType.F32), b32 = b.CastTo(DType.F32);
        using Tensor joined = StreamingTensors.Concat(cpu, [a, b], axis);
        using Tensor joined32 = StreamingTensors.Concat(cpu, [a32, b32], axis);

        Assert.Equal(DType.BF16, joined.DType);
        Assert.Equal(joined32.Shape, joined.Shape);
        using Tensor widened = joined.CastTo(DType.F32);
        Assert.Equal(joined32.AsReadOnlySpan<float>().ToArray(), widened.AsReadOnlySpan<float>().ToArray());

        using Tensor slice = StreamingTensors.Slice(cpu, joined, axis, dimsA[axis] - 1, 2);
        using Tensor slice32 = StreamingTensors.Slice(cpu, joined32, axis, dimsA[axis] - 1, 2);
        using Tensor sliceWidened = slice.CastTo(DType.F32);
        Assert.Equal(slice32.AsReadOnlySpan<float>().ToArray(), sliceWidened.AsReadOnlySpan<float>().ToArray());
    }

    [Fact]
    public void Bf16Window_KeepsTheRoundedTokensOfTheSameChunks()
    {
        using CpuBackend cpu = new();
        using RollingTokenWindow narrow = new(cpu, TokenWindowPolicy.RecentChunks(2, sink: true), streams: 1, tokenAxis: 0, DType.BF16);
        using RollingTokenWindow wide = new(cpu, TokenWindowPolicy.RecentChunks(2, sink: true), streams: 1, tokenAxis: 0);
        Assert.Equal(DType.BF16, narrow.StorageDType);
        for (int chunk = 0; chunk < 4; chunk++)
        {
            int units = chunk == 0 ? 4 : 3;
            narrow.Stage([F32Ramp([units * 2, 3], chunk)], units);
            wide.Stage([F32Ramp([units * 2, 3], chunk)], units);
            narrow.Commit();
            wide.Commit();
        }

        Tensor[] narrowParts = new Tensor[narrow.PartCount], wideParts = new Tensor[wide.PartCount];
        Assert.Equal(wide.CopyParts(0, wideParts), narrow.CopyParts(0, narrowParts));
        for (int p = 0; p < wideParts.Length; p++)
        {
            Assert.Equal(DType.BF16, narrowParts[p].DType);
            Assert.Equal(wideParts[p].Shape, narrowParts[p].Shape);
            using Tensor widened = narrowParts[p].CastTo(DType.F32);
            Assert.Equal(Rounded(wideParts[p]), widened.AsReadOnlySpan<float>().ToArray());
        }
    }

    [Fact]
    public void Bf16Tail_KeepsTheRoundedNewestFrames()
    {
        using CpuBackend cpu = new();
        using FrameTailCache narrow = new(cpu, tailFrames: 3, axis: 0, positionsPerFrame: 2, storageDType: DType.BF16);
        using FrameTailCache wide = new(cpu, tailFrames: 3, axis: 0, positionsPerFrame: 2);
        foreach (int frames in new[] { 4, 1, 3, 2 })
        {
            using Tensor chunk = F32Ramp([frames * 2, 5], frames * 7);
            narrow.Stage(chunk);
            wide.Stage(chunk);
            narrow.Commit();
            wide.Commit();
            Assert.Equal(wide.HeldFrames, narrow.HeldFrames);
            Assert.Equal(DType.BF16, narrow.Current!.DType);
            using Tensor widened = narrow.Current.CastTo(DType.F32);
            Assert.Equal(Rounded(wide.Current!), widened.AsReadOnlySpan<float>().ToArray());
        }
    }

    // Values that are not exactly representable in BF16, so a missing or wrong rounding shows.
    private static Tensor F32Ramp(long[] dims, int offset)
    {
        Tensor tensor = new(new TensorShape(dims), DType.F32, DeviceKind.Cpu);
        Span<float> values = tensor.AsSpan<float>();
        for (int i = 0; i < values.Length; i++)
            values[i] = (offset * 100 + i) * 0.0123457f + 0.5f;
        return tensor;
    }

    private static Tensor Bf16Ramp(long[] dims, int offset)
    {
        using Tensor wide = F32Ramp(dims, offset);
        return wide.CastTo(DType.BF16);
    }

    private static float[] Rounded(Tensor f32)
    {
        ReadOnlySpan<float> values = f32.AsReadOnlySpan<float>();
        float[] rounded = new float[values.Length];
        for (int i = 0; i < values.Length; i++)
            rounded[i] = BFloat16Bits.ToFloat(BFloat16Bits.FromFloat(values[i]));
        return rounded;
    }
}
