using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Kimi;
using HartsyInference.LLM.Transformer.Kda;
using Xunit;

namespace HartsyInference.LLM.Tests.Kda;

/// <summary>The KDA layer: the folded forget gate against the scalar Kimi formula, and the layer on the CPU backend, where one call
/// over a sequence must equal the same tokens split across two calls that share the carried state.</summary>
public sealed class KdaLayerTests
{
    private static readonly KdaSettings Settings = new(Hidden: 16, Heads: 2, HeadDim: 8, ConvKernel: 4, ForgetRank: 4, GateLowerBound: -5f);

    [Fact]
    public void FoldedForgetGate_MatchesScalarKimiLogDecay()
    {
        Random rng = new(3);
        int keyDim = Settings.KeyDim, rank = Settings.ForgetRank;
        float[] up = Floats(rng, keyDim * rank, -0.5f, 0.5f);
        float[] aLog = Floats(rng, keyDim, -1f, 1f);
        float[] dtBias = Floats(rng, keyDim, -1f, 1f);
        float[] low = Floats(rng, rank, -2f, 2f);

        (float[] foldedUp, float[] foldedBias) = KdaLayer.FoldForgetGate(up, aLog, dtBias, keyDim, rank);
        for (int c = 0; c < keyDim; c++)
        {
            float raw = 0f;
            for (int r = 0; r < rank; r++) raw += up[c * rank + r] * low[r];
            float expected = KimiK31Reference.KdaLogDecay(raw, dtBias[c], aLog[c], Settings.GateLowerBound);
            float folded = 0f;
            for (int r = 0; r < rank; r++) folded += foldedUp[c * rank + r] * low[r];
            folded += foldedBias[c];
            float actual = Settings.GateLowerBound * KimiK31Reference.Sigmoid(folded);
            Assert.Equal(expected, actual, 5);
        }
    }

    [Fact]
    public void Forward_OneCallEqualsTwoCallsWithCarriedState()
    {
        const int tokens = 6;
        Random rng = new(17);
        KdaWeights w = RandomWeights(rng);
        IBackend backend = new CpuBackend();
        float[] x = Floats(rng, tokens * Settings.Hidden, -1f, 1f);

        float[] whole = Run(backend, w, x, tokens, split: null);
        float[] halves = Run(backend, w, x, tokens, split: 2);
        for (int i = 0; i < whole.Length; i++)
        {
            Assert.True(float.IsFinite(whole[i]));
            Assert.Equal(whole[i], halves[i], 4);
        }
    }

    [Fact]
    public void LowRankOutputGate_EqualsTheFactorProductAsOneMatrix()
    {
        const int tokens = 5, rank = 3;
        Random rng = new(23);
        KdaSettings s = Settings;
        KdaWeights full = RandomWeights(rng);
        float[] down = Floats(rng, rank * s.Hidden, -0.4f, 0.4f), up = Floats(rng, s.KeyDim * rank, -0.4f, 0.4f);
        float[] product = new float[s.KeyDim * s.Hidden];
        for (int o = 0; o < s.KeyDim; o++)
            for (int i = 0; i < s.Hidden; i++)
                for (int r = 0; r < rank; r++) product[o * s.Hidden + i] += up[o * rank + r] * down[r * s.Hidden + i];
        KdaWeights viaProduct = full with { OutGateProj = Make(product, s.KeyDim, s.Hidden) };
        KdaWeights lowRank = full with { OutGateProj = null, OutGateDown = Make(down, rank, s.Hidden), OutGateUp = Make(up, s.KeyDim, rank) };
        IBackend backend = new CpuBackend();
        float[] x = Floats(rng, tokens * s.Hidden, -1f, 1f);

        float[] a = Run(backend, viaProduct, x, tokens, split: null), b = Run(backend, lowRank, x, tokens, split: null);
        for (int i = 0; i < a.Length; i++) Assert.Equal(a[i], b[i], 5);
    }

    [Fact]
    public void L2Epsilon_IsSeparateFromTheNormEpsilon()
    {
        const int tokens = 4;
        Random rng = new(29);
        KdaWeights w = RandomWeights(rng);
        IBackend backend = new CpuBackend();
        float[] x = Floats(rng, tokens * Settings.Hidden, -1f, 1f);
        using Tensor input = Make(x, tokens, Settings.Hidden);

        float[] Forward(float eps, float? l2Eps)
        {
            using KdaState state = new(Settings);
            using Tensor output = new(new TensorShape(tokens, Settings.Hidden), DType.F32);
            KdaLayer.Forward(backend, Settings, w, input, output, state, eps, l2Eps);
            return output.AsReadOnlySpan<float>().ToArray();
        }

        float[] together = Forward(1e-5f, null), same = Forward(1e-5f, 1e-5f), split = Forward(1e-5f, 5f);
        Assert.Equal(together, same);
        Assert.NotEqual(together, split);
    }

    [Fact]
    public void ExactlyOneOutputGateForm_IsRequired()
    {
        Random rng = new(31);
        KdaWeights w = RandomWeights(rng);
        using Tensor input = Make(Floats(rng, Settings.Hidden, -1f, 1f), 1, Settings.Hidden);
        using Tensor output = new(new TensorShape(1, Settings.Hidden), DType.F32);
        using KdaState state = new(Settings);
        Assert.Throws<ArgumentException>(() => KdaLayer.Forward(new CpuBackend(), Settings, w with { OutGateProj = null }, input, output, state));
        // a full gate with only one factor of a low-rank gate is refused, not run with the factor ignored
        Assert.Throws<ArgumentException>(() => KdaLayer.Forward(new CpuBackend(), Settings, w with { OutGateDown = Make(new float[Settings.Hidden], 1, Settings.Hidden) }, input, output, state));
        // a low-rank gate whose factors disagree in rank is refused
        Assert.Throws<ArgumentException>(() => KdaLayer.Forward(new CpuBackend(), Settings, w with { OutGateProj = null,
            OutGateDown = Make(new float[2 * Settings.Hidden], 2, Settings.Hidden), OutGateUp = Make(new float[Settings.KeyDim * 3], Settings.KeyDim, 3) }, input, output, state));
    }

    /// <summary>Runs the layer over <paramref name="tokens"/> tokens. With <paramref name="split"/> set, the first <c>split</c> tokens
    /// are one call and the rest a second call that carries the state.</summary>
    private static float[] Run(IBackend backend, KdaWeights w, float[] x, int tokens, int? split)
    {
        using KdaState state = new(Settings);
        float[] result = new float[tokens * Settings.Hidden];
        int first = split ?? tokens;
        Chunk(backend, w, state, x, 0, first, result);
        if (split is int s) Chunk(backend, w, state, x, s, tokens - s, result);
        return result;
    }

    private static void Chunk(IBackend backend, KdaWeights w, KdaState state, float[] x, int start, int count, float[] result)
    {
        using Tensor input = Make(x.AsSpan(start * Settings.Hidden, count * Settings.Hidden).ToArray(), count, Settings.Hidden);
        using Tensor output = new(new TensorShape(count, Settings.Hidden), DType.F32);
        KdaLayer.Forward(backend, Settings, w, input, output, state);
        output.AsReadOnlySpan<float>().CopyTo(result.AsSpan(start * Settings.Hidden, count * Settings.Hidden));
    }

    private static KdaWeights RandomWeights(Random rng)
    {
        KdaSettings s = Settings;
        return new KdaWeights(
            QkvProj: Make(Floats(rng, s.ConvDim * s.Hidden, -0.3f, 0.3f), s.ConvDim, s.Hidden),
            ConvTaps: Make(Floats(rng, s.ConvDim * s.ConvKernel, -0.5f, 0.5f), s.ConvDim, s.ConvKernel),
            ForgetDown: Make(Floats(rng, s.ForgetRank * s.Hidden, -0.3f, 0.3f), s.ForgetRank, s.Hidden),
            ForgetUp: Make(Floats(rng, s.KeyDim * s.ForgetRank, -0.3f, 0.3f), s.KeyDim, s.ForgetRank),
            ForgetBias: Make(Floats(rng, s.KeyDim, -0.2f, 0.2f), s.KeyDim),
            BetaProj: Make(Floats(rng, s.Heads * s.Hidden, -0.3f, 0.3f), s.Heads, s.Hidden),
            OutGateProj: Make(Floats(rng, s.KeyDim * s.Hidden, -0.3f, 0.3f), s.KeyDim, s.Hidden),
            OutNorm: Make(Floats(rng, s.HeadDim, 0.8f, 1.2f), s.HeadDim),
            OutProj: Make(Floats(rng, s.Hidden * s.KeyDim, -0.3f, 0.3f), s.Hidden, s.KeyDim));
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
