using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>Validates <see cref="MoeFeedForward"/> (router top-k → grouped expert SwiGLU → weighted combine +
/// shared expert) against an independent pure-C# reference that mirrors the HF Qwen-MoE math.</summary>
public sealed unsafe class MoeTests
{
    private static uint _rng = 0x6C0FFEEu;
    private static float Rand() { _rng ^= _rng << 13; _rng ^= _rng >> 17; _rng ^= _rng << 5; return ((_rng & 0xFFFF) / 65535f - 0.5f) * 0.5f; }
    private static Tensor F2(int a, int b) { Tensor t = new(new TensorShape(a, b), DType.F32); float* p = (float*)t.DataPointer; for (long i = 0; i < t.ElementCount; i++) p[i] = Rand(); return t; }
    private static float[] Host(Tensor t) { float[] r = new float[t.ElementCount]; float* p = (float*)t.DataPointer; for (long i = 0; i < r.Length; i++) r[i] = p[i]; return r; }

    [Fact]
    public void MoeFeedForward_MatchesReference()
    {
        const int hidden = 16, inter = 24, shInter = 20, e = 4, topK = 2, n = 5;
        MoeConfig moe = new()
        {
            NumExperts = e, NumExpertsPerTok = topK, MoeIntermediateSize = inter,
            NormTopKProb = true, Scoring = MoeScoring.Softmax, SharedExpertIntermediateSize = shInter,
        };

        const string p = "model.layers.0";
        Dictionary<string, Tensor> w = new()
        {
            [$"{p}.mlp.gate.weight"] = F2(e, hidden),
            [$"{p}.mlp.shared_expert.gate_proj.weight"] = F2(shInter, hidden),
            [$"{p}.mlp.shared_expert.up_proj.weight"] = F2(shInter, hidden),
            [$"{p}.mlp.shared_expert.down_proj.weight"] = F2(hidden, shInter),
            [$"{p}.mlp.shared_expert_gate.weight"] = F2(1, hidden),
        };
        for (int i = 0; i < e; i++)
        {
            w[$"{p}.mlp.experts.{i}.gate_proj.weight"] = F2(inter, hidden);
            w[$"{p}.mlp.experts.{i}.up_proj.weight"] = F2(inter, hidden);
            w[$"{p}.mlp.experts.{i}.down_proj.weight"] = F2(hidden, inter);
        }

        using CpuBackend backend = new();
        MoeFeedForward moeFf = new(moe, hidden, lowVram: false);
        moeFf.LoadWeights(w, p);

        using Tensor x = F2(n, hidden);   // [N, hidden] == [1, N, hidden] memory
        using Tensor outActual = moeFf.Forward(backend, Reshape(x, n, hidden), n);

        float[] expected = Reference(Host(x), w, p, moe, hidden, inter, shInter, e, topK, n);
        float* a = (float*)outActual.DataPointer;
        float max = 0f;
        for (int i = 0; i < expected.Length; i++) max = MathF.Max(max, MathF.Abs(expected[i] - a[i]));
        Assert.True(max <= 1e-4f, $"MoE diverges from reference by {max:E3}");

        foreach (Tensor t in w.Values) t.Dispose();
    }

    private static Tensor Reshape(Tensor x, int n, int hidden)
    {
        Tensor r = new(new TensorShape(1, n, hidden), DType.F32);
        float* s = (float*)x.DataPointer;
        float* d = (float*)r.DataPointer;
        for (long i = 0; i < x.ElementCount; i++) d[i] = s[i];
        return r;
    }

    // output[t,h] = sigmoid(x·shGate) * sharedSwiGLU(x)[h] + Σ_{selected e} w_e * expertSwiGLU_e(x)[h]
    private static float[] Reference(float[] x, Dictionary<string, Tensor> w, string p, MoeConfig moe,
        int hidden, int inter, int shInter, int e, int topK, int n)
    {
        float[] router = Host(w[$"{p}.mlp.gate.weight"]);
        float[] shGate = Host(w[$"{p}.mlp.shared_expert_gate.weight"]);
        float[] shG = Host(w[$"{p}.mlp.shared_expert.gate_proj.weight"]);
        float[] shU = Host(w[$"{p}.mlp.shared_expert.up_proj.weight"]);
        float[] shD = Host(w[$"{p}.mlp.shared_expert.down_proj.weight"]);
        float[][] eg = new float[e][], eu = new float[e][], ed = new float[e][];
        for (int i = 0; i < e; i++)
        {
            eg[i] = Host(w[$"{p}.mlp.experts.{i}.gate_proj.weight"]);
            eu[i] = Host(w[$"{p}.mlp.experts.{i}.up_proj.weight"]);
            ed[i] = Host(w[$"{p}.mlp.experts.{i}.down_proj.weight"]);
        }

        float[] outp = new float[n * hidden];
        for (int t = 0; t < n; t++)
        {
            ReadOnlySpan<float> xt = x.AsSpan(t * hidden, hidden);

            // shared expert, scaled by sigmoid gate
            float gate = 1f / (1f + MathF.Exp(-Dot(xt, shGate, 0, hidden)));
            float[] sh = SwiGlu(xt, shG, shU, shD, hidden, shInter);
            for (int h = 0; h < hidden; h++) outp[t * hidden + h] += gate * sh[h];

            // router softmax + top-k
            float[] score = new float[e];
            float mx = float.NegativeInfinity;
            for (int i = 0; i < e; i++) { score[i] = Dot(xt, router, i * hidden, hidden); mx = MathF.Max(mx, score[i]); }
            float sum = 0f;
            for (int i = 0; i < e; i++) { score[i] = MathF.Exp(score[i] - mx); sum += score[i]; }
            for (int i = 0; i < e; i++) score[i] /= sum;

            int[] pick = new int[topK];
            float wsum = 0f;
            for (int kk = 0; kk < topK; kk++)
            {
                int best = -1; float bv = float.NegativeInfinity;
                for (int i = 0; i < e; i++)
                {
                    bool taken = false;
                    for (int j = 0; j < kk; j++) if (pick[j] == i) { taken = true; break; }
                    if (!taken && score[i] > bv) { bv = score[i]; best = i; }
                }
                pick[kk] = best; wsum += bv;
            }
            for (int kk = 0; kk < topK; kk++)
            {
                int ex = pick[kk];
                float wt = moe.NormTopKProb ? score[ex] / wsum : score[ex];
                float[] h = SwiGlu(xt, eg[ex], eu[ex], ed[ex], hidden, inter);
                for (int j = 0; j < hidden; j++) outp[t * hidden + j] += wt * h[j];
            }
        }
        return outp;
    }

    private static float[] SwiGlu(ReadOnlySpan<float> x, float[] gateW, float[] upW, float[] downW, int hidden, int inter)
    {
        float[] comb = new float[inter];
        for (int i = 0; i < inter; i++)
        {
            float g = Dot(x, gateW, i * hidden, hidden);
            float u = Dot(x, upW, i * hidden, hidden);
            float silu = g / (1f + MathF.Exp(-g));
            comb[i] = silu * u;
        }
        float[] outp = new float[hidden];
        for (int h = 0; h < hidden; h++)
        {
            float acc = 0f;
            for (int i = 0; i < inter; i++) acc += comb[i] * downW[h * inter + i];
            outp[h] = acc;
        }
        return outp;
    }

    private static float Dot(ReadOnlySpan<float> x, float[] w, int wOffset, int len)
    {
        float acc = 0f;
        for (int i = 0; i < len; i++) acc += x[i] * w[wOffset + i];
        return acc;
    }
}
