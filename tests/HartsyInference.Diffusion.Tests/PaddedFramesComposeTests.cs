using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Backends.Layout;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The strided composition of a CausalConv3d's padded input writes exactly the host loop's elements. <c>SyntheticSmoke</c>: the composition
/// runs on the CPU backend, so each case compares it bit for bit with <see cref="IBackend.BuildPaddedFramesReference"/>; the device runs the same
/// composition through its own copy kernel.</summary>
[Trait("Category", "SyntheticSmoke")]
public sealed class PaddedFramesComposeTests
{
    [Theory]
    [InlineData(0, 0, true, 0, 0, 3)]   // no cache, replicate, no spatial pad: the first frame repeats, then the input, then the last frame
    [InlineData(2, 0, true, 1, 2, 2)]   // replicate, cache of 2 frames, spatial pad 1 by 2
    [InlineData(2, 1, false, 1, 1, 3)]  // zero temporal pad, cache of 1 frame, spatial pad 1
    [InlineData(1, 2, true, 0, 1, 4)]   // replicate with a cache, pad only in width
    [InlineData(0, 3, false, 2, 0, 2)]  // zero pad, no cache, pad only in height, clamped tail beyond the input
    public void ComposedPaddingMatchesTheHostLoop(int cacheLen, int zeroPad, bool replicate, int padH, int padW, int extraFrames)
    {
        const int cIn = 2, tin = 3, h = 3, w = 4;
        int paddedT = zeroPad + cacheLen + tin + extraFrames;
        int hp = h + 2 * padH, wp = w + 2 * padW;
        using CpuBackend backend = new();
        using Tensor input = Fill(new TensorShape([1L, cIn, tin, h, w]), 1);
        using Tensor cache = cacheLen == 0 ? null! : Fill(new TensorShape([1L, cIn, cacheLen, h, w]), 2);
        using Tensor expected = Fill(new TensorShape([paddedT, cIn, hp, wp]), 9);
        using Tensor actual = Fill(new TensorShape([paddedT, cIn, hp, wp]), 9);

        IBackend.BuildPaddedFramesReference(expected, input, cacheLen == 0 ? null : cache, zeroPad, replicate, padH, padW, false);
        PaddedFrames.Compose(backend, actual, input, cacheLen == 0 ? null : cache, zeroPad, replicate, padH, padW);

        ReadOnlySpan<float> e = expected.AsReadOnlySpan<float>();
        ReadOnlySpan<float> a = actual.AsReadOnlySpan<float>();
        Assert.Equal(e.Length, a.Length);
        for (int i = 0; i < e.Length; i++) Assert.True(e[i] == a[i], $"element {i}: host {e[i]} composed {a[i]}");
    }

    /// <summary>A tensor of distinct values, so a misplaced element shows up in the comparison.</summary>
    private static Tensor Fill(TensorShape shape, int seed)
    {
        Tensor tensor = new(shape, DType.F32);
        Span<float> values = tensor.AsSpan<float>();
        for (int i = 0; i < values.Length; i++) values[i] = seed * 1000f + i;
        return tensor;
    }
}
