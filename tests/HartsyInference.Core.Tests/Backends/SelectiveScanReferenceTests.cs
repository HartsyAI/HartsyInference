using HartsyInference.Core.Backends.Host;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Core.Tests.Backends;

public sealed unsafe class SelectiveScanReferenceTests
{
    private const int Rows = 3;
    private const int Frames = 11;
    private const int Channels = 5;
    private const int States = 4;

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void MatchesADoublePrecisionRecurrence(bool gated)
    {
        Operands o = Operands.Random(7, gated);
        SelectiveScanReference.Apply(o.Output, o.U, o.Delta, o.A, o.B, o.C, o.D, o.Z, null, Rows, Frames, Channels, States);
        float[] expected = Naive(o, initial: null, out _);
        Assert.True(MaxAbsDiff(o.Output, expected) < 1e-5f, $"max abs diff {MaxAbsDiff(o.Output, expected)}");
    }

    [Fact]
    public void ACarriedStateContinuesWhereTheLastCallStopped()
    {
        Operands o = Operands.Random(11, gated: true);
        SelectiveScanReference.Apply(o.Output, o.U, o.Delta, o.A, o.B, o.C, o.D, o.Z, null, Rows, Frames, Channels, States);
        float[] whole = ToArray(o.Output);

        int split = 4;
        Tensor state = new(new TensorShape(Rows, Channels, States), DType.F32);
        new Span<float>((float*)state.DataPointer, (int)state.ElementCount).Clear();
        float[] stitched = new float[whole.Length];
        foreach ((int from, int count) in new[] { (0, split), (split, Frames - split) })
        {
            Operands part = o.Slice(from, count);
            SelectiveScanReference.Apply(part.Output, part.U, part.Delta, part.A, part.B, part.C, part.D, part.Z, state, Rows, count, Channels, States);
            float* p = (float*)part.Output.DataPointer;
            for (int r = 0; r < Rows; r++)
            {
                for (int t = 0; t < count; t++)
                {
                    for (int c = 0; c < Channels; c++)
                    {
                        stitched[((r * Frames) + from + t) * Channels + c] = p[((r * count) + t) * Channels + c];
                    }
                }
            }
        }
        for (int i = 0; i < whole.Length; i++)
        {
            Assert.Equal(whole[i], stitched[i], 6);
        }
    }

    [Fact]
    public void RefusesMismatchedOperands()
    {
        Operands o = Operands.Random(3, gated: false);
        Tensor wrong = new(new TensorShape(Rows, Frames, States + 1), DType.F32);
        Assert.Throws<ArgumentException>(() =>
            SelectiveScanReference.Apply(o.Output, o.U, o.Delta, o.A, wrong, o.C, o.D, null, null, Rows, Frames, Channels, States));
    }

    [Fact]
    public void RefusesMoreStatesThanTheStackHolds()
    {
        const int wide = SelectiveScanReference.MaxStates + 1;
        Tensor one = new(new TensorShape(1), DType.F32);
        Tensor wideTensor = new(new TensorShape(wide), DType.F32);
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            SelectiveScanReference.Apply(one, one, one, wideTensor, wideTensor, wideTensor, one, null, null, 1, 1, 1, wide));
    }

    private static float[] Naive(Operands o, float[]? initial, out float[] last)
    {
        float* u = (float*)o.U.DataPointer;
        float* delta = (float*)o.Delta.DataPointer;
        float* a = (float*)o.A.DataPointer;
        float* b = (float*)o.B.DataPointer;
        float* c = (float*)o.C.DataPointer;
        float* d = (float*)o.D.DataPointer;
        float* z = o.Z is null ? null : (float*)o.Z.DataPointer;
        float[] result = new float[Rows * Frames * Channels];
        last = new float[Rows * Channels * States];
        for (int r = 0; r < Rows; r++)
        {
            for (int ch = 0; ch < Channels; ch++)
            {
                double[] h = new double[States];
                for (int n = 0; n < States && initial is not null; n++)
                {
                    h[n] = initial[((r * Channels) + ch) * States + n];
                }
                for (int t = 0; t < Frames; t++)
                {
                    int at = ((r * Frames) + t) * Channels + ch;
                    double y = 0;
                    for (int n = 0; n < States; n++)
                    {
                        h[n] = Math.Exp((double)delta[at] * a[ch * States + n]) * h[n] + (double)delta[at] * b[((r * Frames) + t) * States + n] * u[at];
                        y += c[((r * Frames) + t) * States + n] * h[n];
                    }
                    y += (double)d[ch] * u[at];
                    if (z is not null)
                    {
                        y *= z[at] / (1.0 + Math.Exp(-z[at]));
                    }
                    result[at] = (float)y;
                }
                for (int n = 0; n < States; n++)
                {
                    last[((r * Channels) + ch) * States + n] = (float)h[n];
                }
            }
        }
        return result;
    }

    private static float[] ToArray(Tensor t) => new ReadOnlySpan<float>((float*)t.DataPointer, (int)t.ElementCount).ToArray();

    private static float MaxAbsDiff(Tensor t, float[] expected)
    {
        float* p = (float*)t.DataPointer;
        float max = 0f;
        for (int i = 0; i < expected.Length; i++)
        {
            max = Math.Max(max, Math.Abs(p[i] - expected[i]));
        }
        return max;
    }

    private sealed class Operands
    {
        public required Tensor Output, U, Delta, A, B, C, D;
        public Tensor? Z;

        public static Operands Random(int seed, bool gated)
        {
            Random rng = new(seed);
            Operands o = new()
            {
                Output = New(Rows * Frames * Channels, rng, 0f, 0f),
                U = New(Rows * Frames * Channels, rng, -1f, 1f),
                Delta = New(Rows * Frames * Channels, rng, 0.01f, 0.6f),
                A = New(Channels * States, rng, -2f, -0.1f),
                B = New(Rows * Frames * States, rng, -1f, 1f),
                C = New(Rows * Frames * States, rng, -1f, 1f),
                D = New(Channels, rng, 0.5f, 1.5f),
                Z = gated ? New(Rows * Frames * Channels, rng, -2f, 2f) : null,
            };
            return o;
        }

        public Operands Slice(int from, int count)
        {
            return new Operands
            {
                Output = new Tensor(new TensorShape(Rows * count * Channels), DType.F32),
                U = Cut(U, Channels, from, count),
                Delta = Cut(Delta, Channels, from, count),
                A = A,
                B = Cut(B, States, from, count),
                C = Cut(C, States, from, count),
                D = D,
                Z = Z is null ? null : Cut(Z, Channels, from, count),
            };
        }

        private static Tensor Cut(Tensor source, int width, int from, int count)
        {
            Tensor cut = new(new TensorShape(Rows * count * width), DType.F32);
            float* src = (float*)source.DataPointer;
            float* dst = (float*)cut.DataPointer;
            for (int r = 0; r < Rows; r++)
            {
                for (int t = 0; t < count; t++)
                {
                    for (int i = 0; i < width; i++)
                    {
                        dst[((r * count) + t) * width + i] = src[((r * Frames) + from + t) * width + i];
                    }
                }
            }
            return cut;
        }

        private static Tensor New(int count, Random rng, float low, float high)
        {
            Tensor t = new(new TensorShape(count), DType.F32);
            float* p = (float*)t.DataPointer;
            for (int i = 0; i < count; i++)
            {
                p[i] = low + (float)rng.NextDouble() * (high - low);
            }
            return t;
        }
    }
}
