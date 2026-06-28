using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.LLM.Multimodal;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>Numerically validates the Llama-3.2-Vision (mllama) gated cross-attention decoder layer — the only new
/// core primitive the model adds over the verified Llama text decoder — against an independent host reference, and
/// checks the tanh-gate-at-zero property (a zero gate makes the layer an exact no-op, as it is at init). The full
/// 11B model exceeds local VRAM, so this slice-parity test is the build-defer correctness gate for the primitive.</summary>
public sealed unsafe class MllamaCrossAttentionTests
{
    private static uint _rng = 0x3A1Du;
    private static float Rand() { _rng ^= _rng << 13; _rng ^= _rng >> 17; _rng ^= _rng << 5; return ((_rng & 0xFFFF) / 65535f - 0.5f) * 0.4f; }
    private static Tensor Fill(Tensor t) { float* p = (float*)t.DataPointer; for (long i = 0; i < t.ElementCount; i++) p[i] = Rand(); return t; }
    private static Tensor F2(int a, int b) => Fill(new Tensor(new TensorShape(a, b), DType.F32));
    private static Tensor F1(int a) => Fill(new Tensor(new TensorShape(a), DType.F32));
    private static Tensor Pos1(int a) { Tensor t = new(new TensorShape(a), DType.F32); float* p = (float*)t.DataPointer; for (int i = 0; i < a; i++) p[i] = 0.5f + 0.5f * MathF.Abs(Rand()); return t; }
    private static Tensor ScalarT(float v) { Tensor t = new(new TensorShape(1), DType.F32); *(float*)t.DataPointer = v; return t; }
    private static float[] Host(Tensor t) { float[] r = new float[t.ElementCount]; float* p = (float*)t.DataPointer; for (long i = 0; i < r.Length; i++) r[i] = p[i]; return r; }

    private const int Hidden = 16, Hq = 4, Hkv = 2, D = 8, Inter = 24, T = 3, L = 5;   // L = vision tokens
    private const float Eps = 1e-6f;

    private static Dictionary<string, Tensor> Weights(string p, float attnGate, float mlpGate) => new()
    {
        [$"{p}.input_layernorm.weight"] = Pos1(Hidden),
        [$"{p}.post_attention_layernorm.weight"] = Pos1(Hidden),
        [$"{p}.cross_attn.q_proj.weight"] = F2(Hq * D, Hidden),
        [$"{p}.cross_attn.k_proj.weight"] = F2(Hkv * D, Hidden),
        [$"{p}.cross_attn.v_proj.weight"] = F2(Hkv * D, Hidden),
        [$"{p}.cross_attn.o_proj.weight"] = F2(Hidden, Hq * D),
        [$"{p}.cross_attn.q_norm.weight"] = Pos1(D),
        [$"{p}.cross_attn.k_norm.weight"] = Pos1(D),
        [$"{p}.mlp.gate_proj.weight"] = F2(Inter, Hidden),
        [$"{p}.mlp.up_proj.weight"] = F2(Inter, Hidden),
        [$"{p}.mlp.down_proj.weight"] = F2(Hidden, Inter),
        [$"{p}.cross_attn_attn_gate"] = ScalarT(attnGate),
        [$"{p}.cross_attn_mlp_gate"] = ScalarT(mlpGate),
    };

    [Fact]
    public void CrossAttention_MatchesReference()
    {
        const string p = "blk.0";
        Dictionary<string, Tensor> w = Weights(p, attnGate: 0.7f, mlpGate: -0.4f);
        using CpuBackend backend = new();
        MllamaCrossAttentionLayer layer = new(Hidden, Hq, Hkv, D, Inter, Eps);
        layer.LoadWeights(w, p);

        using Tensor text = Fill(new Tensor(new TensorShape(1, T, Hidden), DType.F32));
        using Tensor vision = Fill(new Tensor(new TensorShape(1, L, Hidden), DType.F32));
        using Tensor actual = layer.Forward(backend, text, T, vision, L);

        float[] expected = Reference(Host(text), Host(vision), w, p);
        float* a = (float*)actual.DataPointer;
        float max = 0f;
        for (int i = 0; i < expected.Length; i++) max = MathF.Max(max, MathF.Abs(expected[i] - a[i]));
        Assert.True(max <= 1e-4f, $"mllama cross-attention diverges from reference by {max:E3}");
        foreach (Tensor t in w.Values) t.Dispose();
    }

    [Fact]
    public void CrossAttention_ZeroGates_IsIdentity()
    {
        const string p = "blk.0";
        Dictionary<string, Tensor> w = Weights(p, attnGate: 0f, mlpGate: 0f);
        using CpuBackend backend = new();
        MllamaCrossAttentionLayer layer = new(Hidden, Hq, Hkv, D, Inter, Eps);
        layer.LoadWeights(w, p);

        using Tensor text = Fill(new Tensor(new TensorShape(1, T, Hidden), DType.F32));
        using Tensor vision = Fill(new Tensor(new TensorShape(1, L, Hidden), DType.F32));
        using Tensor actual = layer.Forward(backend, text, T, vision, L);

        float* a = (float*)actual.DataPointer, ti = (float*)text.DataPointer;
        float max = 0f;
        for (long i = 0; i < actual.ElementCount; i++) max = MathF.Max(max, MathF.Abs(a[i] - ti[i]));
        Assert.True(max <= 1e-5f, $"tanh(0)=0 gates must make the layer a no-op, diff {max:E3}");
        foreach (Tensor t in w.Values) t.Dispose();
    }

    // Independent host reference of the gated cross-attention layer.
    private static float[] Reference(float[] text, float[] vision, Dictionary<string, Tensor> w, string p)
    {
        float[] inN = Host(w[$"{p}.input_layernorm.weight"]), postN = Host(w[$"{p}.post_attention_layernorm.weight"]);
        float[] qW = Host(w[$"{p}.cross_attn.q_proj.weight"]), kW = Host(w[$"{p}.cross_attn.k_proj.weight"]);
        float[] vW = Host(w[$"{p}.cross_attn.v_proj.weight"]), oW = Host(w[$"{p}.cross_attn.o_proj.weight"]);
        float[] qNorm = Host(w[$"{p}.cross_attn.q_norm.weight"]), kNorm = Host(w[$"{p}.cross_attn.k_norm.weight"]);
        float[] gW = Host(w[$"{p}.mlp.gate_proj.weight"]), uW = Host(w[$"{p}.mlp.up_proj.weight"]), dW = Host(w[$"{p}.mlp.down_proj.weight"]);
        float attnGate = MathF.Tanh(*(float*)w[$"{p}.cross_attn_attn_gate"].DataPointer);
        float mlpGate = MathF.Tanh(*(float*)w[$"{p}.cross_attn_mlp_gate"].DataPointer);
        int group = Hq / Hkv;

        // K/V from vision (rms-normed per head for K).
        float[,][] kHead = new float[L, Hkv][], vHead = new float[L, Hkv][];
        for (int s = 0; s < L; s++)
        {
            ReadOnlySpan<float> vs = vision.AsSpan(s * Hidden, Hidden);
            for (int hh = 0; hh < Hkv; hh++)
            {
                float[] kk = new float[D], vv = new float[D];
                for (int x = 0; x < D; x++) { kk[x] = Dot(vs, kW, (hh * D + x) * Hidden, Hidden); vv[x] = Dot(vs, vW, (hh * D + x) * Hidden, Hidden); }
                kHead[s, hh] = RmsVec(kk, kNorm); vHead[s, hh] = vv;
            }
        }

        float[] outp = new float[T * Hidden];
        float scale = 1f / MathF.Sqrt(D);
        for (int r = 0; r < T; r++)
        {
            float[] xr = RmsVec(text.AsSpan(r * Hidden, Hidden).ToArray(), inN);
            float[] attnConcat = new float[Hq * D];
            for (int hh = 0; hh < Hq; hh++)
            {
                int kvh = hh / group;
                float[] qh = new float[D];
                for (int x = 0; x < D; x++) qh[x] = Dot(xr, qW, (hh * D + x) * Hidden, Hidden);
                qh = RmsVec(qh, qNorm);
                // softmax over vision tokens, weighted V.
                float[] sc = new float[L]; float mx = float.NegativeInfinity;
                for (int s = 0; s < L; s++) { float dp = 0f; for (int x = 0; x < D; x++) dp += qh[x] * kHead[s, kvh][x]; sc[s] = dp * scale; mx = MathF.Max(mx, sc[s]); }
                float z = 0f; for (int s = 0; s < L; s++) { sc[s] = MathF.Exp(sc[s] - mx); z += sc[s]; }
                for (int x = 0; x < D; x++) { float acc = 0f; for (int s = 0; s < L; s++) acc += sc[s] * vHead[s, kvh][x]; attnConcat[hh * D + x] = acc / z; }
            }
            // o_proj + gated residual.
            float[] hvec = new float[Hidden];
            for (int o = 0; o < Hidden; o++) { float acc = 0f; for (int i = 0; i < Hq * D; i++) acc += attnConcat[i] * oW[o * Hq * D + i]; hvec[o] = text[r * Hidden + o] + attnGate * acc; }
            // gated FFN.
            float[] yn = RmsVec(hvec, postN);
            float[] comb = new float[Inter];
            for (int i = 0; i < Inter; i++) { float g = Dot(yn, gW, i * Hidden, Hidden); float u = Dot(yn, uW, i * Hidden, Hidden); comb[i] = (g / (1f + MathF.Exp(-g))) * u; }
            for (int o = 0; o < Hidden; o++) { float acc = 0f; for (int i = 0; i < Inter; i++) acc += comb[i] * dW[o * Inter + i]; outp[r * Hidden + o] = hvec[o] + mlpGate * acc; }
        }
        return outp;
    }

    private static float[] RmsVec(float[] x, float[] weight)
    {
        float ms = 0f; for (int i = 0; i < x.Length; i++) ms += x[i] * x[i]; ms /= x.Length;
        float inv = 1f / MathF.Sqrt(ms + Eps);
        float[] r = new float[x.Length];
        for (int i = 0; i < x.Length; i++) r[i] = x[i] * inv * weight[i];
        return r;
    }

    private static float Dot(ReadOnlySpan<float> x, float[] w, int off, int len) { float a = 0f; for (int i = 0; i < len; i++) a += x[i] * w[off + i]; return a; }
}
