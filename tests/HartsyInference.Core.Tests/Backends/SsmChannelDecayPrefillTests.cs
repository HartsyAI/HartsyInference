using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Core.Tests.Backends;

public sealed class SsmChannelDecayPrefillTests
{
    private const float Tol = 1e-4f;

    [Fact]
    public void ChannelDecayPrefill_MatchesNaiveDoubleRecurrence_WithGroupedKeyHeads()
    {
        // 3 tokens, 4 value heads over 2 key heads (repeat 2), sv 3, sk 4, with a non-trivial starting state.
        const int seq = 3, hv = 4, sv = 3, sk = 4, repeat = 2, keyHeads = hv / repeat;
        Random rng = new(7);
        float[] q = Floats(rng, seq * keyHeads * sk, -1f, 1f);
        float[] k = Floats(rng, seq * keyHeads * sk, -1f, 1f);
        float[] v = Floats(rng, seq * hv * sv, -1f, 1f);
        float[] logDecay = Floats(rng, seq * hv * sk, -2f, 0f);
        float[] beta = Floats(rng, seq * hv, -2f, 2f);
        float[] state0 = Floats(rng, hv * sv * sk, -0.5f, 0.5f);

        using Tensor qT = Make(q, seq, keyHeads * sk);
        using Tensor kT = Make(k, seq, keyHeads * sk);
        using Tensor vT = Make(v, seq, hv * sv);
        using Tensor gT = Make(logDecay, seq, hv * sk);
        using Tensor bT = Make(beta, seq, hv);
        using Tensor stateT = Make(state0, hv, sv, sk);
        using Tensor outT = new(new TensorShape(seq, hv * sv), DType.F32);

        ((IBackend)new CpuBackend()).SsmChannelDecayPrefill(outT, stateT, qT, kT, vT, gT, bT, hv, sv, sk, repeat);

        double[] expected = Naive(q, k, v, logDecay, beta, state0, seq, hv, sv, sk, repeat, out double[] expectedState);
        float[] actual = outT.AsReadOnlySpan<float>().ToArray();
        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], actual[i], Tol);
        float[] actualState = stateT.AsReadOnlySpan<float>().ToArray();
        for (int i = 0; i < expectedState.Length; i++) Assert.Equal(expectedState[i], actualState[i], Tol);
    }

    [Fact]
    public void ChannelDecayPrefill_ZeroLogDecay_LeavesStateUndecayed()
    {
        // logDecay = 0 is exp(0) = 1 on every channel, so the recurrence reduces to the plain delta rule with no forgetting.
        const int seq = 2, hv = 2, sv = 2, sk = 2;
        float[] q = { 1f, 0f, 0f, 1f, 1f, 1f, 0f, 0f };
        float[] k = { 1f, 0f, 0f, 1f, 1f, 0f, 0f, 1f };
        float[] v = { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f };
        float[] logDecay = new float[seq * hv * sk];
        float[] beta = new float[seq * hv];
        float[] state0 = new float[hv * sv * sk];

        using Tensor qT = Make(q, seq, hv * sk);
        using Tensor kT = Make(k, seq, hv * sk);
        using Tensor vT = Make(v, seq, hv * sv);
        using Tensor gT = Make(logDecay, seq, hv * sk);
        using Tensor bT = Make(beta, seq, hv);
        using Tensor stateT = Make(state0, hv, sv, sk);
        using Tensor outT = new(new TensorShape(seq, hv * sv), DType.F32);

        ((IBackend)new CpuBackend()).SsmChannelDecayPrefill(outT, stateT, qT, kT, vT, gT, bT, hv, sv, sk, 1);

        double[] expected = Naive(q, k, v, logDecay, beta, state0, seq, hv, sv, sk, 1, out _);
        float[] actual = outT.AsReadOnlySpan<float>().ToArray();
        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], actual[i], Tol);
    }

    [Fact]
    public void ChannelDecayPrefill_RejectsMismatchedDecayWidth()
    {
        using Tensor qT = new(new TensorShape(1, 4), DType.F32);
        using Tensor kT = new(new TensorShape(1, 4), DType.F32);
        using Tensor vT = new(new TensorShape(1, 4), DType.F32);
        using Tensor gT = new(new TensorShape(1, 3), DType.F32);
        using Tensor bT = new(new TensorShape(1, 2), DType.F32);
        using Tensor stateT = new(new TensorShape(2, 2, 2), DType.F32);
        using Tensor outT = new(new TensorShape(1, 4), DType.F32);

        Assert.Throws<ArgumentException>(() =>
            ((IBackend)new CpuBackend()).SsmChannelDecayPrefill(outT, stateT, qT, kT, vT, gT, bT, 2, 2, 2, 1));
    }

    /// <summary>Plain double-precision recurrence, one token and one value head at a time, in the order the op documents.</summary>
    private static double[] Naive(float[] q, float[] k, float[] v, float[] logDecay, float[] beta, float[] state0,
        int seq, int hv, int sv, int sk, int repeat, out double[] finalState)
    {
        int keyHeads = hv / repeat;
        double[] s = state0.Select(x => (double)x).ToArray();
        double[] o = new double[seq * hv * sv];
        for (int h = 0; h < hv; h++)
        {
            int kh = h / repeat;
            int sBase = h * sv * sk;
            for (int t = 0; t < seq; t++)
            {
                double b = 1.0 / (1.0 + Math.Exp(-beta[t * hv + h]));
                for (int r = 0; r < sv; r++)
                    for (int c = 0; c < sk; c++)
                        s[sBase + r * sk + c] *= Math.Exp(logDecay[t * hv * sk + h * sk + c]);
                double[] rowDot = new double[sv];
                for (int r = 0; r < sv; r++)
                    for (int c = 0; c < sk; c++)
                        rowDot[r] += s[sBase + r * sk + c] * k[t * keyHeads * sk + kh * sk + c];
                for (int r = 0; r < sv; r++)
                {
                    double delta = (v[t * hv * sv + h * sv + r] - rowDot[r]) * b;
                    for (int c = 0; c < sk; c++)
                        s[sBase + r * sk + c] += k[t * keyHeads * sk + kh * sk + c] * delta;
                }
                for (int r = 0; r < sv; r++)
                {
                    double acc = 0;
                    for (int c = 0; c < sk; c++)
                        acc += s[sBase + r * sk + c] * q[t * keyHeads * sk + kh * sk + c];
                    o[t * hv * sv + h * sv + r] = acc;
                }
            }
        }
        finalState = s;
        return o;
    }

    private static float[] Floats(Random rng, int n, float lo, float hi)
    {
        float[] values = new float[n];
        for (int i = 0; i < n; i++) values[i] = lo + (float)rng.NextDouble() * (hi - lo);
        return values;
    }

    private static Tensor Make(float[] values, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        values.CopyTo(t.AsSpan<float>());
        return t;
    }
}
