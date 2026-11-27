using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The carried state of <see cref="IBackend.FrameDeltaScan"/> on the host reference: a scan split into two calls with
/// the forward state carried between them ends in the same state as one call, and its last frame reads out the same (the
/// reverse pass contributes nothing to a sequence's last frame), for a full value and for the unit value of a denominator.</summary>
public sealed class FrameDeltaScanStateTests
{
    private const int Frames = 5, Split = 2, Spatial = 6, Heads = 2, KeyDim = 8;

    [Theory]
    [InlineData(8)]
    [InlineData(1)]
    public void TwoCallsWithCarriedState_EqualOneCall(int valueDim)
    {
        using CpuBackend cpu = new();
        IBackend backend = cpu;
        Inputs whole = Inputs.Random(Frames, valueDim, seed: 11);
        using Tensor wholeOut = new(new TensorShape((long)Frames * Spatial, (long)Heads * valueDim), DType.F32);
        using Tensor wholeState = State(valueDim);
        backend.FrameDeltaScan(wholeOut, whole.Query, whole.Key, whole.Value, whole.Beta, whole.Decay, Frames, Spatial, Heads, KeyDim,
            valueDim, initialState: null, finalState: wholeState);

        Inputs first = whole.Frames(0, Split), second = whole.Frames(Split, Frames - Split);
        using Tensor firstOut = new(new TensorShape((long)Split * Spatial, (long)Heads * valueDim), DType.F32);
        using Tensor secondOut = new(new TensorShape((long)(Frames - Split) * Spatial, (long)Heads * valueDim), DType.F32);
        using Tensor carried = State(valueDim), finalState = State(valueDim);
        backend.FrameDeltaScan(firstOut, first.Query, first.Key, first.Value, first.Beta, first.Decay, Split, Spatial, Heads, KeyDim,
            valueDim, initialState: null, finalState: carried);
        backend.FrameDeltaScan(secondOut, second.Query, second.Key, second.Value, second.Beta, second.Decay, Frames - Split, Spatial,
            Heads, KeyDim, valueDim, initialState: carried, finalState: finalState);

        AssertClose("final state", wholeState.AsReadOnlySpan<float>(), finalState.AsReadOnlySpan<float>());
        int row = Heads * valueDim;
        ReadOnlySpan<float> wholeLast = wholeOut.AsReadOnlySpan<float>()[((Frames - 1) * Spatial * row)..];
        ReadOnlySpan<float> splitLast = secondOut.AsReadOnlySpan<float>()[((Frames - Split - 1) * Spatial * row)..];
        AssertClose("last frame readout", wholeLast, splitLast);
        whole.Dispose();
        first.Dispose();
        second.Dispose();
    }

    [Fact]
    public void NullInitialState_EqualsZeroState()
    {
        using CpuBackend cpu = new();
        IBackend backend = cpu;
        Inputs inputs = Inputs.Random(3, KeyDim, seed: 5);
        using Tensor a = new(new TensorShape(3L * Spatial, (long)Heads * KeyDim), DType.F32);
        using Tensor b = new(new TensorShape(3L * Spatial, (long)Heads * KeyDim), DType.F32);
        using Tensor zeros = State(KeyDim);
        backend.FrameDeltaScan(a, inputs.Query, inputs.Key, inputs.Value, inputs.Beta, inputs.Decay, 3, Spatial, Heads, KeyDim, KeyDim);
        backend.FrameDeltaScan(b, inputs.Query, inputs.Key, inputs.Value, inputs.Beta, inputs.Decay, 3, Spatial, Heads, KeyDim, KeyDim,
            initialState: zeros);
        Assert.True(a.AsReadOnlySpan<float>().SequenceEqual(b.AsReadOnlySpan<float>()));
        inputs.Dispose();
    }

    [Fact]
    public void FinalStateAliasingInitialState_IsRefused()
    {
        using CpuBackend cpu = new();
        IBackend backend = cpu;
        Inputs inputs = Inputs.Random(2, KeyDim, seed: 3);
        using Tensor output = new(new TensorShape(2L * Spatial, (long)Heads * KeyDim), DType.F32);
        using Tensor state = State(KeyDim);
        Assert.ThrowsAny<ArgumentException>(() => backend.FrameDeltaScan(output, inputs.Query, inputs.Key, inputs.Value, inputs.Beta,
            inputs.Decay, 2, Spatial, Heads, KeyDim, KeyDim, initialState: state, finalState: state));
        inputs.Dispose();
    }

    private static Tensor State(int valueDim) => new(new TensorShape(Heads, KeyDim, valueDim), DType.F32);

    private static void AssertClose(string label, ReadOnlySpan<float> expected, ReadOnlySpan<float> actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        double scale = 0, worst = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            scale = Math.Max(scale, Math.Abs(expected[i]));
            worst = Math.Max(worst, Math.Abs(expected[i] - actual[i]));
        }

        Assert.True(worst <= 1e-5 * Math.Max(scale, 1), $"{label}: max abs difference {worst:E3} against scale {scale:E3}");
    }

    /// <summary>Token-major scan operands of a clip, sliceable by frames.</summary>
    private sealed class Inputs : IDisposable
    {
        public required Tensor Query { get; init; }
        public required Tensor Key { get; init; }
        public required Tensor? Value { get; init; }
        public required Tensor Beta { get; init; }
        public required Tensor Decay { get; init; }
        public required int FrameCount { get; init; }
        public required int ValueDim { get; init; }

        public static Inputs Random(int frames, int valueDim, int seed)
        {
            Random random = new(seed);
            long tokens = (long)frames * Spatial;
            return new Inputs
            {
                Query = Fill(random, [tokens, (long)Heads * KeyDim], 0f, 1f),
                Key = Fill(random, [tokens, (long)Heads * KeyDim], 0f, 0.3f),
                Value = valueDim == 1 ? null : Fill(random, [tokens, (long)Heads * valueDim], -1f, 1f),
                Beta = Fill(random, [Heads, frames, Spatial], 0f, 1f),
                Decay = Fill(random, [Heads, frames], 0.5f, 1f),
                FrameCount = frames,
                ValueDim = valueDim,
            };
        }

        /// <summary>Frames <c>[start, start + count)</c> as an independent copy.</summary>
        public Inputs Frames(int start, int count)
        {
            long row = (long)Heads * KeyDim, valueRow = (long)Heads * ValueDim;
            return new Inputs
            {
                Query = Rows(Query, start * Spatial * row, count * Spatial * row, [count * (long)Spatial, row]),
                Key = Rows(Key, start * Spatial * row, count * Spatial * row, [count * (long)Spatial, row]),
                Value = Value is null ? null : Rows(Value, start * Spatial * valueRow, count * Spatial * valueRow, [count * (long)Spatial, valueRow]),
                Beta = PerHead(Beta, start * Spatial, count * Spatial, FrameCount * Spatial, [Heads, count, Spatial]),
                Decay = PerHead(Decay, start, count, FrameCount, [Heads, count]),
                FrameCount = count,
                ValueDim = ValueDim,
            };
        }

        public void Dispose()
        {
            Query.Dispose();
            Key.Dispose();
            Value?.Dispose();
            Beta.Dispose();
            Decay.Dispose();
        }

        private static Tensor Fill(Random random, long[] dims, float low, float high)
        {
            Tensor tensor = new(new TensorShape(dims), DType.F32);
            Span<float> values = tensor.AsSpan<float>();
            for (int i = 0; i < values.Length; i++)
                values[i] = low + (high - low) * (float)random.NextDouble();
            return tensor;
        }

        private static Tensor Rows(Tensor source, long offset, long length, long[] dims)
        {
            Tensor slice = new(new TensorShape(dims), DType.F32);
            source.AsReadOnlySpan<float>().Slice((int)offset, (int)length).CopyTo(slice.AsSpan<float>());
            return slice;
        }

        // [heads, …] tensors keep their per-head runs: copy elements [start, start + count) of each head's run of `perHead`.
        private static Tensor PerHead(Tensor source, int start, int count, int perHead, long[] dims)
        {
            Tensor slice = new(new TensorShape(dims), DType.F32);
            ReadOnlySpan<float> from = source.AsReadOnlySpan<float>();
            Span<float> to = slice.AsSpan<float>();
            for (int h = 0; h < Heads; h++)
                from.Slice(h * perHead + start, count).CopyTo(to.Slice(h * count, count));
            return slice;
        }
    }
}
