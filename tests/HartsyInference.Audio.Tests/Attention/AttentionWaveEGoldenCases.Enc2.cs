using HartsyInference.Audio.Models.CosyVoice;
using HartsyInference.Audio.Models.GptSoVits;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using static HartsyInference.Tests.Common.Attention.AttentionGolden;
using Owner = HartsyInference.Tests.Common.Attention.GoldenOwner;
using Weights = HartsyInference.Tests.Common.Attention.GoldenWeights;

namespace HartsyInference.Audio.Tests.Attention;

internal static unsafe partial class AttentionWaveEGoldenCases
{
    private static void RegisterEnc2(Dictionary<string, Func<CpuBackend, string>> d)
    {
        d["s3_block"] = S3BlockCase;
        d["text2semantic_block"] = Text2SemanticBlockCase;
    }

    // ── S3 tokenizer block (fixed 1280 wide x 20 heads; FSMN side branch plus rotate-half RoPE) ────────────────────

    private static string S3BlockCase(CpuBackend cpu)
    {
        const int n = 1280, hd = 64, half = hd / 2, ff = 2560, t = 9;
        const float s = 0.02f;
        Owner o = new();
        Weights w = new(160);
        w.AddNorm("b.attn_ln.weight", n); w.AddBias("b.attn_ln.bias", n);
        foreach (string x in new[] { "query", "value", "out" }) { w.AddTensor($"b.attn.{x}.weight", s, n, n); w.AddBias($"b.attn.{x}.bias", n); }
        w.AddTensor("b.attn.key.weight", s, n, n);
        w.AddTensor("b.attn.fsmn_block.weight", 0.3f, n, 1, 31);
        w.AddNorm("b.mlp_ln.weight", n); w.AddBias("b.mlp_ln.bias", n);
        w.AddTensor("b.mlp.0.weight", s, ff, n); w.AddBias("b.mlp.0.bias", ff);
        w.AddTensor("b.mlp.2.weight", s, n, ff); w.AddBias("b.mlp.2.bias", n);
        float[] cos = new float[t * half], sin = new float[t * half];
        for (int p = 0; p < t; p++)
            for (int i = 0; i < half; i++)
            {
                double ang = p * Math.Pow(10_000.0, -2.0 * i / hd);
                cos[p * half + i] = (float)Math.Cos(ang); sin[p * half + i] = (float)Math.Sin(ang);
            }
        S3Block block = new();
        block.LoadWeights(w, "b");
        Tensor seq = Seq(o, 1601, 1f, 1, t, n);
        Tensor y = o.Own(block.Forward(cpu, seq, t, cos, sin));
        return Finish(o, ("out", y));
    }

    // ── GPT-SoVITS text-to-semantic block (private Block, prefix-LM mask, ReLU FFN, post-norm) ─────────────────────

    private static string Text2SemanticBlockCase(CpuBackend cpu)
    {
        const int h = 32, heads = 2, ffn = 48, t = 9, prefix = 4;
        Owner o = new();
        Text2SemanticConfig cfg = new() { Hidden = h, NumHeads = heads, FfnDim = ffn };
        Weights w = new(161);
        w.Linear("b.self_attn.out_proj", h, h, true);
        w.Add("b.self_attn.in_proj_weight", 3 * h, h); w.AddBias("b.self_attn.in_proj_bias", 3 * h);
        w.Linear("b.linear1", ffn, h, true); w.Linear("b.linear2", h, ffn, true);
        foreach (string n in new[] { "norm1", "norm2" }) { w.AddNorm($"b.{n}.weight", h); w.AddBias($"b.{n}.bias", h); }
        object block = NewPrivate(typeof(Text2Semantic), "Block", cfg);
        Invoke(block, "LoadWeights", w, "b");
        Tensor x = Seq(o, 1611, 1f, 1, t, h);
        // Prefix rows see the whole prefix; later rows are causal.
        Tensor mask = o.Own(AdditiveMask(t, t, (i, j) => j < prefix || j <= i));
        Tensor y = o.Own((Tensor)Invoke(block, "Forward", cpu, x, t, mask));
        return Finish(o, ("out", y));
    }
}
