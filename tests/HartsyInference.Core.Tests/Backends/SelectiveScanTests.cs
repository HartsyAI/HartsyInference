using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Backends.Host;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Core.Tests.Backends;

/// <summary>The Mamba-1 selective scan on the host: the reference against a double-precision recurrence, and the state carried
/// across calls. The GPU backends are checked against this reference in the GpuIntegration tests.</summary>
public sealed class SelectiveScanTests
{
    private const float Tol = 1e-4f;

    [Fact]
    public void SelectiveScan_MatchesNaiveRecurrence_WithGateAndCarriedState()
    {
        const int rows = 2, frames = 5, channels = 3, states = 4;
        Random rng = new(11);
        Operands op = Operands.Random(rng, rows, frames, channels, states);
        float[] state0 = Floats(rng, rows * channels * states, -0.5f, 0.5f);

        using Tensor state = Make(state0, rows, channels, states);
        using Tensor output = new(new TensorShape(rows, frames, channels), DType.F32);
        op.Run(output, state, rows, frames, channels, states);

        double[] expected = Naive(op, state0, rows, frames, channels, states, out double[] expectedState);
        AssertClose(expected, output.AsReadOnlySpan<float>().ToArray(), "output");
        AssertClose(expectedState, state.AsReadOnlySpan<float>().ToArray(), "state");
    }

    [Fact]
    public void SelectiveScan_OneFrame_MatchesNaiveRecurrence()
    {
        const int rows = 1, frames = 1, channels = 2, states = 3;
        Random rng = new(5);
        Operands op = Operands.Random(rng, rows, frames, channels, states);
        float[] state0 = new float[rows * channels * states];

        using Tensor state = Make(state0, rows, channels, states);
        using Tensor output = new(new TensorShape(rows, frames, channels), DType.F32);
        op.Run(output, state, rows, frames, channels, states);

        double[] expected = Naive(op, state0, rows, frames, channels, states, out _);
        AssertClose(expected, output.AsReadOnlySpan<float>().ToArray(), "output");
    }

    [Fact]
    public void SelectiveScan_MaxStates_MatchesNaiveRecurrence()
    {
        const int rows = 1, frames = 3, channels = 2, states = SelectiveScanReference.MaxStates;
        Random rng = new(23);
        Operands op = Operands.Random(rng, rows, frames, channels, states, rateLo: -0.05f);
        float[] state0 = Floats(rng, rows * channels * states, -0.1f, 0.1f);

        using Tensor state = Make(state0, rows, channels, states);
        using Tensor output = new(new TensorShape(rows, frames, channels), DType.F32);
        op.Run(output, state, rows, frames, channels, states);

        double[] expected = Naive(op, state0, rows, frames, channels, states, out double[] expectedState);
        AssertClose(expected, output.AsReadOnlySpan<float>().ToArray(), "output");
        AssertClose(expectedState, state.AsReadOnlySpan<float>().ToArray(), "state");
    }

    [Fact]
    public void SelectiveScan_CarriedState_SplitRunEqualsWholeRun()
    {
        // One scan over all frames must equal two scans over the halves, the second starting from the first's carried state.
        const int rows = 2, frames = 6, channels = 3, states = 4, half = 3;
        Random rng = new(31);
        Operands whole = Operands.Random(rng, rows, frames, channels, states);
        float[] state0 = Floats(rng, rows * channels * states, -0.5f, 0.5f);

        using Tensor wholeState = Make(state0, rows, channels, states);
        using Tensor wholeOut = new(new TensorShape(rows, frames, channels), DType.F32);
        whole.Run(wholeOut, wholeState, rows, frames, channels, states);

        Operands first = whole.Slice(rows, frames, channels, states, 0, half);
        Operands second = whole.Slice(rows, frames, channels, states, half, frames - half);
        using Tensor carried = Make(state0, rows, channels, states);
        using Tensor firstOut = new(new TensorShape(rows, half, channels), DType.F32);
        using Tensor secondOut = new(new TensorShape(rows, frames - half, channels), DType.F32);
        first.Run(firstOut, carried, rows, half, channels, states);
        second.Run(secondOut, carried, rows, frames - half, channels, states);

        float[] whole32 = wholeOut.AsReadOnlySpan<float>().ToArray();
        float[] split = Interleave(firstOut.AsReadOnlySpan<float>().ToArray(), secondOut.AsReadOnlySpan<float>().ToArray(), rows, half, frames - half, channels);
        AssertClose(whole32, split, "output");
        AssertClose(wholeState.AsReadOnlySpan<float>().ToArray(), carried.AsReadOnlySpan<float>().ToArray(), "state");
    }

    [Fact]
    public void SelectiveScan_NullStateAndGate_StartFromZeroAndKeepNothing()
    {
        const int rows = 1, frames = 4, channels = 2, states = 2;
        Random rng = new(41);
        Operands op = Operands.Random(rng, rows, frames, channels, states, withGate: false);
        float[] zeros = new float[rows * channels * states];

        using Tensor output = new(new TensorShape(rows, frames, channels), DType.F32);
        ((IBackend)new CpuBackend()).SelectiveScan(output, op.U, op.Delta, op.A, op.B, op.C, op.D, null, null,
            rows, frames, channels, states);

        double[] expected = Naive(op, zeros, rows, frames, channels, states, out _);
        AssertClose(expected, output.AsReadOnlySpan<float>().ToArray(), "output");
    }

    [Fact]
    public void SelectiveScan_RejectsStatesAboveTheReferenceLimit()
    {
        const int rows = 1, frames = 1, channels = 1, states = SelectiveScanReference.MaxStates + 1;
        using Tensor u = new(new TensorShape(rows, frames, channels), DType.F32);
        using Tensor delta = new(new TensorShape(rows, frames, channels), DType.F32);
        using Tensor a = new(new TensorShape(channels, states), DType.F32);
        using Tensor b = new(new TensorShape(rows, frames, states), DType.F32);
        using Tensor c = new(new TensorShape(rows, frames, states), DType.F32);
        using Tensor d = new(new TensorShape(channels), DType.F32);
        using Tensor output = new(new TensorShape(rows, frames, channels), DType.F32);

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            ((IBackend)new CpuBackend()).SelectiveScan(output, u, delta, a, b, c, d, null, null, rows, frames, channels, states));
    }

    /// <summary>The operands of one scan, in the layout <see cref="IBackend.SelectiveScan"/> documents.</summary>
    private sealed class Operands : IDisposable
    {
        public Tensor U = null!, Delta = null!, A = null!, B = null!, C = null!, D = null!, Z = null!;
        public float[] UV = null!, DeltaV = null!, AV = null!, BV = null!, CV = null!, DV = null!, ZV = null!;

        public static Operands Random(Random rng, int rows, int frames, int channels, int states, float rateLo = -2f, bool withGate = true)
        {
            Operands op = new()
            {
                UV = Floats(rng, rows * frames * channels, -1f, 1f),
                DeltaV = Floats(rng, rows * frames * channels, 0.05f, 1f),
                AV = Floats(rng, channels * states, rateLo, -0.1f),
                BV = Floats(rng, rows * frames * states, -1f, 1f),
                CV = Floats(rng, rows * frames * states, -1f, 1f),
                DV = Floats(rng, channels, -0.5f, 0.5f),
                ZV = withGate ? Floats(rng, rows * frames * channels, -2f, 2f) : null!,
            };
            op.Materialise(rows, frames, channels, states);
            return op;
        }

        /// <summary>The operands restricted to frames [start, start + count) of each row.</summary>
        public Operands Slice(int rows, int frames, int channels, int states, int start, int count)
        {
            Operands op = new()
            {
                UV = Pick(UV, rows, frames, channels, start, count),
                DeltaV = Pick(DeltaV, rows, frames, channels, start, count),
                AV = AV,
                BV = Pick(BV, rows, frames, states, start, count),
                CV = Pick(CV, rows, frames, states, start, count),
                DV = DV,
                ZV = ZV is null ? null! : Pick(ZV, rows, frames, channels, start, count),
            };
            op.Materialise(rows, count, channels, states);
            return op;
        }

        public void Run(Tensor output, Tensor state, int rows, int frames, int channels, int states)
            => ((IBackend)new CpuBackend()).SelectiveScan(output, U, Delta, A, B, C, D, ZV is null ? null : Z, state,
                rows, frames, channels, states);

        private void Materialise(int rows, int frames, int channels, int states)
        {
            U = Make(UV, rows, frames, channels);
            Delta = Make(DeltaV, rows, frames, channels);
            A = Make(AV, channels, states);
            B = Make(BV, rows, frames, states);
            C = Make(CV, rows, frames, states);
            D = Make(DV, channels);
            Z = ZV is null ? new Tensor(new TensorShape(1), DType.F32) : Make(ZV, rows, frames, channels);
        }

        private static float[] Pick(float[] source, int rows, int frames, int width, int start, int count)
        {
            float[] result = new float[rows * count * width];
            for (int r = 0; r < rows; r++)
                Array.Copy(source, ((long)r * frames + start) * width, result, (long)r * count * width, (long)count * width);
            return result;
        }

        public void Dispose()
        {
            U?.Dispose(); Delta?.Dispose(); A?.Dispose(); B?.Dispose(); C?.Dispose(); D?.Dispose(); Z?.Dispose();
        }
    }

    /// <summary>The recurrence in double precision, one channel at a time, straight from the definition.</summary>
    private static double[] Naive(Operands op, float[] state0, int rows, int frames, int channels, int states, out double[] finalState)
    {
        double[] output = new double[rows * frames * channels];
        finalState = new double[rows * channels * states];
        for (int r = 0; r < rows; r++)
        for (int ch = 0; ch < channels; ch++)
        {
            double[] h = new double[states];
            for (int n = 0; n < states; n++) h[n] = state0[((long)r * channels + ch) * states + n];
            for (int t = 0; t < frames; t++)
            {
                long at = ((long)r * frames + t) * channels + ch;
                double step = op.DeltaV[at];
                double x = op.UV[at];
                double acc = 0;
                for (int n = 0; n < states; n++)
                {
                    double bn = op.BV[((long)r * frames + t) * states + n];
                    double cn = op.CV[((long)r * frames + t) * states + n];
                    h[n] = Math.Exp(step * op.AV[ch * states + n]) * h[n] + step * x * bn;
                    acc += cn * h[n];
                }
                acc += x * op.DV[ch];
                if (op.ZV is not null)
                {
                    double g = op.ZV[at];
                    acc *= g / (1 + Math.Exp(-g));
                }
                output[at] = acc;
            }
            for (int n = 0; n < states; n++) finalState[((long)r * channels + ch) * states + n] = h[n];
        }
        return output;
    }

    private static float[] Interleave(float[] first, float[] second, int rows, int firstFrames, int secondFrames, int channels)
    {
        float[] result = new float[rows * (firstFrames + secondFrames) * channels];
        for (int r = 0; r < rows; r++)
        {
            Array.Copy(first, (long)r * firstFrames * channels, result, (long)r * (firstFrames + secondFrames) * channels, (long)firstFrames * channels);
            Array.Copy(second, (long)r * secondFrames * channels, result, ((long)r * (firstFrames + secondFrames) + firstFrames) * channels, (long)secondFrames * channels);
        }
        return result;
    }

    private static void AssertClose(double[] want, float[] got, string what)
    {
        Assert.Equal(want.Length, got.Length);
        for (int i = 0; i < want.Length; i++)
            Assert.True(Math.Abs(want[i] - got[i]) <= Tol, $"{what}[{i}] want {want[i]} got {got[i]}");
    }

    private static void AssertClose(float[] want, float[] got, string what)
    {
        Assert.Equal(want.Length, got.Length);
        for (int i = 0; i < want.Length; i++)
            Assert.True(Math.Abs(want[i] - got[i]) <= Tol, $"{what}[{i}] want {want[i]} got {got[i]}");
    }

    private static Tensor Make(float[] values, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        values.CopyTo(t.AsSpan<float>());
        return t;
    }

    private static float[] Floats(Random rng, int n, float lo, float hi)
    {
        float[] values = new float[n];
        for (int i = 0; i < n; i++) values[i] = lo + (float)rng.NextDouble() * (hi - lo);
        return values;
    }
}
