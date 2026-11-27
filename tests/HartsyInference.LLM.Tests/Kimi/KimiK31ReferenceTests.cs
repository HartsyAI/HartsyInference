using HartsyInference.LLM.Transformer.AttnRes;
using HartsyInference.LLM.Kimi;
using Xunit;

namespace HartsyInference.LLM.Tests.Kimi;

/// <summary>Hand-computed cases for the K3.1 CPU references. Each expected value is derived in the test comment.</summary>
public class KimiK31ReferenceTests
{
    private const float Tol = 1e-5f;

    [Fact]
    public void KdaLogDecay_IsBoundedAndMatchesFormula()
    {
        // z = exp(0) * (0 + 0) = 0 -> sigmoid(0) = 0.5 -> g = -5 * 0.5 = -2.5
        Assert.Equal(-2.5f, KimiK31Reference.KdaLogDecay(0f, 0f, 0f, -5f), Tol);
        // Large positive z saturates to the bound; large negative saturates to 0 (never positive).
        Assert.Equal(-5f, KimiK31Reference.KdaLogDecay(100f, 0f, 0f, -5f), Tol);
        Assert.Equal(0f, KimiK31Reference.KdaLogDecay(-100f, 0f, 0f, -5f), Tol);
        // A_log = log 2 -> exp = 2; raw 1, dt 0 -> z = 2 -> g = -5 * sigmoid(2)
        float expected = -5f * (1f / (1f + MathF.Exp(-2f)));
        Assert.Equal(expected, KimiK31Reference.KdaLogDecay(1f, 0f, MathF.Log(2f), -5f), Tol);
    }

    [Fact]
    public void KdaRecurrence_ZeroBeta_OnlyDecaysState()
    {
        // dk = 2, dv = 1, one token. S0 = [[2],[4]]. g = 0 (decay 1), beta = 0 -> S unchanged; o = Sᵀq.
        float[] state = { 2f, 4f };
        float[] q = { 1f, 1f };
        float[] k = { 1f, 0f };
        float[] v = { 9f };
        float[] g = { 0f, 0f };
        float[] beta = { 0f };
        float[] output = new float[1];
        KimiK31Reference.KdaRecurrenceHead(q, k, v, g, beta, 1, 2, 1, state, output);
        Assert.Equal(6f, output[0], Tol); // 2*1 + 4*1
        Assert.Equal(2f, state[0], Tol);
        Assert.Equal(4f, state[1], Tol);
    }

    [Fact]
    public void KdaRecurrence_FullBeta_DeltaRuleReplacesValueAlongKey()
    {
        // dk = 2, dv = 1, S0 = 0, g = 0, beta = 1, k = (1,0), v = 3, q = k.
        // kv = 0 -> delta = 3 -> S = k·3ᵀ = [[3],[0]] -> o = Sᵀq = 3.
        float[] state = new float[2];
        float[] output = new float[1];
        KimiK31Reference.KdaRecurrenceHead(
            new[] { 1f, 0f }, new[] { 1f, 0f }, new[] { 3f }, new[] { 0f, 0f }, new[] { 1f }, 1, 2, 1, state, output);
        Assert.Equal(3f, output[0], Tol);
        Assert.Equal(3f, state[0], Tol);
        Assert.Equal(0f, state[1], Tol);
    }

    [Fact]
    public void KdaRecurrence_DecayIsPerKeyChannel()
    {
        // dk = 2, dv = 1, S0 = [[1],[1]], g = (ln 0.5, 0) -> decay (0.5, 1), beta 0.
        // S = [[0.5],[1]] -> o with q = (1,1) is 1.5.
        float[] state = { 1f, 1f };
        float[] output = new float[1];
        KimiK31Reference.KdaRecurrenceHead(
            new[] { 1f, 1f }, new[] { 0f, 0f }, new[] { 0f }, new[] { MathF.Log(0.5f), 0f }, new[] { 0f }, 1, 2, 1, state, output);
        Assert.Equal(0.5f, state[0], Tol);
        Assert.Equal(1f, state[1], Tol);
        Assert.Equal(1.5f, output[0], Tol);
    }

    [Fact]
    public void AttnResMix_EqualLogits_IsTheMean()
    {
        // Zero query -> every logit is 0 -> uniform weights over the sources.
        List<float[]> sources = new() { new[] { 1f, 2f }, new[] { 3f, 6f }, new[] { 5f, 10f } };
        float[] output = new float[2];
        AttentionResidual.MixReference(sources, new[] { 1f, 1f }, new[] { 0f, 0f }, output);
        Assert.Equal(3f, output[0], Tol);
        Assert.Equal(6f, output[1], Tol);
    }

    [Fact]
    public void AttnResMix_DominantLogit_SelectsThatSource()
    {
        // Keys are RMS-normalized, so only direction matters. Source 1 = (1,0) is aligned with the query (50,0):
        // RMS key (√2, 0) -> logit 50√2 ≈ 70.7. Source 0 = (0,1) is orthogonal -> logit 0. Softmax puts ~all weight on
        // source 1, so the output is ~ (1, 0).
        List<float[]> sources = new() { new[] { 0f, 1f }, new[] { 1f, 0f } };
        float[] output = new float[2];
        AttentionResidual.MixReference(sources, new[] { 1f, 1f }, new[] { 50f, 0f }, output);
        Assert.InRange(output[0], 0.999f, 1.001f);
        Assert.InRange(output[1], -0.001f, 0.001f);
    }

    [Fact]
    public void DsaIndexerScore_ReluThenHeadWeightedSum()
    {
        // heads = 2, dim = 2. q0 = (1,0), q1 = (-1,0), k = (1,1), w = (1, 2).
        // relu(q0·k) = relu(1) = 1 -> *1 = 1; relu(q1·k) = relu(-1) = 0 -> *2 = 0. Score = 1.
        float score = KimiK31Reference.DsaIndexerScore(new[] { 1f, 0f, -1f, 0f }, new[] { 1f, 1f }, new[] { 1f, 2f }, 2, 2);
        Assert.Equal(1f, score, Tol);
    }

    [Fact]
    public void SelectCausalTopK_ShortContext_KeepsAllCausalKeys()
    {
        float[] scores = { 0.1f, 0.9f, 0.5f, 0.2f, 0.8f };
        int[] kept = KimiK31Reference.SelectCausalTopK(scores, queryPos: 2, topK: 2048);
        Assert.Equal(new[] { 0, 1, 2 }, kept);
    }

    [Fact]
    public void SelectCausalTopK_LongContext_KeepsTopKAscendingAndCausal()
    {
        // Keys 0..3 are causal for queryPos 3; key 4 is in the future and must not appear even though it scores highest.
        float[] scores = { 0.1f, 0.9f, 0.5f, 0.2f, 10f };
        int[] kept = KimiK31Reference.SelectCausalTopK(scores, queryPos: 3, topK: 2);
        Assert.Equal(new[] { 1, 2 }, kept);
    }

    [Fact]
    public void SelectCausalTopK_DropsNonFiniteScores()
    {
        float[] scores = { float.NaN, 0.3f, float.NegativeInfinity, 0.7f };
        int[] kept = KimiK31Reference.SelectCausalTopK(scores, queryPos: 3, topK: 2);
        Assert.Equal(new[] { 1, 3 }, kept);
    }

    [Fact]
    public void SigmoidRoute_BiasChangesSelectionButNotWeights()
    {
        // logits 0 -> sigmoid 0.5 for all. Bias picks experts 2 and 0 for selection; weights are raw sigmoid 0.5 each.
        // renormalize over two: 0.5/(1.0+1e-20) = 0.5, times routedScale 1.0.
        Span<int> idx = stackalloc int[2];
        Span<float> w = stackalloc float[2];
        KimiK31Reference.SigmoidRoute(
            new[] { 0f, 0f, 0f, 0f }, new[] { 0.1f, -0.1f, 0.2f, -0.2f }, 2, true, 1f, idx, w);
        Assert.Equal(new[] { 2, 0 }, new[] { idx[0], idx[1] });
        Assert.Equal(0.5f, w[0], Tol);
        Assert.Equal(0.5f, w[1], Tol);
    }

    [Fact]
    public void SigmoidRoute_RenormalizesAndScales()
    {
        // logits (large, 0): sigmoid ~ (1.0, 0.5). Top-2 with zero bias: weights (1.0, 0.5),
        // renormalized to sum 1 -> (2/3, 1/3), scaled by 2.5 -> (5/3, 5/6).
        Span<int> idx = stackalloc int[2];
        Span<float> w = stackalloc float[2];
        KimiK31Reference.SigmoidRoute(new[] { 50f, 0f }, new[] { 0f, 0f }, 2, true, 2.5f, idx, w);
        Assert.Equal(0, idx[0]);
        Assert.Equal(1, idx[1]);
        Assert.Equal(5f / 3f, w[0], 1e-4f);
        Assert.Equal(5f / 6f, w[1], 1e-4f);
    }

    [Fact]
    public void Situ_GateIsFirstArgumentPerNemo()
    {
        // situ(gate=0, up=x) = 4·tanh(0)·sigmoid(0)·... = 0; situ(gate=g, up=0) = 0 (tanh(0)=0).
        Assert.Equal(0f, KimiK31Reference.Situ(0f, 3f), Tol);
        Assert.Equal(0f, KimiK31Reference.Situ(2f, 0f), Tol);
        // Gate large positive: 4·tanh(g/4)·sigmoid(g) -> ~4·1·1 = 4; up = 25·tanh(u/25) -> ~u for small u.
        Assert.Equal(4f * 1f * 1f * 25f * MathF.Tanh(0.01f / 25f), KimiK31Reference.Situ(50f, 0.01f), 1e-3f);
    }
}
