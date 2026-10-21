using System.Reflection;
using HartsyInference.Audio.Models.PocketTts;
using HartsyInference.Audio.Models.QwenTts;
using HartsyInference.Audio.Models.StyleTts2;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using static HartsyInference.Tests.Common.Attention.AttentionGolden;
using Owner = HartsyInference.Tests.Common.Attention.GoldenOwner;
using Weights = HartsyInference.Tests.Common.Attention.GoldenWeights;

namespace HartsyInference.Audio.Tests.Attention;

internal static unsafe partial class AttentionWaveEGoldenCases
{
    private static void RegisterTts2(Dictionary<string, Func<CpuBackend, string>> d)
    {
        d["pockettts_backbone_b1"] = cpu => PocketBackbone(cpu, 1);
        d["pockettts_backbone_b2"] = cpu => PocketBackbone(cpu, 2);
        d["pockettts_flowlm_primed"] = PocketFlowLmPrimed;
        d["pockettts_mimi"] = PocketMimiDecoder;
        d["qwen3tts_pre_transformer"] = Qwen3TtsLayer;
        d["style_transformer1d_self_cross"] = StyleTransformer;
    }

    // ── PocketTTS ───────────────────────────────────────────────────────────────────────────────────────────────

    private const int PkD = 32, PkHeads = 4, PkFfn = 48, PkLayers = 2;

    private static Weights PocketWeights(uint salt, bool layerScale)
    {
        Weights w = new(salt);
        for (int i = 0; i < PkLayers; i++)
        {
            string p = $"p.layers.{i}";
            w.AddNorm($"{p}.norm1.weight", PkD); w.AddBias($"{p}.norm1.bias", PkD);
            w.AddNorm($"{p}.norm2.weight", PkD); w.AddBias($"{p}.norm2.bias", PkD);
            w.Add($"{p}.self_attn.in_proj.weight", 3 * PkD, PkD); w.Add($"{p}.self_attn.out_proj.weight", PkD, PkD);
            w.Add($"{p}.linear1.weight", PkFfn, PkD); w.Add($"{p}.linear2.weight", PkD, PkFfn);
            if (layerScale) { w.AddTensor($"{p}.layer_scale_1.scale", 0.5f, PkD); w.AddTensor($"{p}.layer_scale_2.scale", 0.5f, PkD); }
        }
        return w;
    }

    private static string PocketBackbone(CpuBackend cpu, int batch)
    {
        // The Mimi-style 6-key context bites at t = 10; layer scale on.
        Owner o = new();
        PocketTtsStreamingTransformer tr = new("p", PkD, PkHeads, PkLayers, PkFfn, context: 6, layerScale: true);
        tr.LoadWeights(PocketWeights(70u + (uint)batch, layerScale: true));
        Tensor x = Seq(o, 710 + (uint)batch, 1f, batch, 10, PkD);
        Tensor y = o.Own(tr.Forward(cpu, x, batch, 10));
        return Finish(o, ("out", y));
    }

    private static string PocketFlowLmPrimed(CpuBackend cpu)
    {
        // FlowLm's backbone config (no context window, no layer scale) primed with a 4-row prefix; FlowLm itself is fixed 6x1024.
        Owner o = new();
        PocketTtsStreamingTransformer tr = new("p", PkD, PkHeads, PkLayers, PkFfn, context: null, layerScale: false);
        tr.LoadWeights(PocketWeights(73, layerScale: false));
        const int prefixLen = 4, t = 3, hd = PkD / PkHeads;
        Tensor[] pk = new Tensor[PkLayers], pv = new Tensor[PkLayers];
        for (int i = 0; i < PkLayers; i++)
        {
            pk[i] = Seq(o, 730 + (uint)i, 1f, 1, PkHeads, prefixLen, hd);
            pv[i] = Seq(o, 740 + (uint)i, 1f, 1, PkHeads, prefixLen, hd);
        }
        Tensor x = Seq(o, 750, 1f, 1, t, PkD);
        Tensor y = o.Own(tr.ForwardPrimed(cpu, x, t, pk, pv, prefixLen));
        PocketTtsStreamingTransformer windowed = new("p", PkD, PkHeads, PkLayers, PkFfn, context: 5, layerScale: false);
        windowed.LoadWeights(PocketWeights(73, layerScale: false));
        Tensor yw = o.Own(windowed.ForwardPrimed(cpu, x, t, pk, pv, prefixLen));
        return Finish(o, ("nowindow", y), ("window5", yw));
    }

    private static string PocketMimiDecoder(CpuBackend cpu)
    {
        // 16 latent frames x16 upsample = 256 steps against the 250-key window, so the window masks real keys.
        const float s = 0.05f;
        Owner o = new();
        Weights w = new(74);
        w.AddTensor("mimi.quantizer.output_proj.weight", s, 512, 32, 1);
        w.AddTensor("mimi.upsample.convtr.convtr.weight", s, 512, 1, 32);
        for (int i = 0; i < 2; i++)
        {
            string p = $"mimi.decoder_transformer.transformer.layers.{i}";
            w.AddNorm($"{p}.norm1.weight", 512); w.AddBias($"{p}.norm1.bias", 512);
            w.AddNorm($"{p}.norm2.weight", 512); w.AddBias($"{p}.norm2.bias", 512);
            w.AddTensor($"{p}.self_attn.in_proj.weight", s, 1536, 512); w.AddTensor($"{p}.self_attn.out_proj.weight", s, 512, 512);
            w.AddTensor($"{p}.linear1.weight", s, 2048, 512); w.AddTensor($"{p}.linear2.weight", s, 512, 2048);
            w.AddTensor($"{p}.layer_scale_1.scale", 0.5f, 512); w.AddTensor($"{p}.layer_scale_2.scale", 0.5f, 512);
        }
        string d = "mimi.decoder.model";
        w.AddTensor($"{d}.0.conv.weight", s, 512, 512, 7); w.AddBias($"{d}.0.conv.bias", 512);
        int[] upIdx = [2, 5, 8], rbIdx = [3, 6, 9], ratios = [6, 5, 4];
        for (int i = 0; i < 3; i++)
        {
            int inCh = (8 >> i) * 64, outCh = inCh / 2;
            w.AddTensor($"{d}.{upIdx[i]}.convtr.weight", s, inCh, outCh, 2 * ratios[i]); w.AddBias($"{d}.{upIdx[i]}.convtr.bias", outCh);
            w.AddTensor($"{d}.{rbIdx[i]}.block.1.conv.weight", s, outCh / 2, outCh, 3); w.AddBias($"{d}.{rbIdx[i]}.block.1.conv.bias", outCh / 2);
            w.AddTensor($"{d}.{rbIdx[i]}.block.3.conv.weight", s, outCh, outCh / 2, 1); w.AddBias($"{d}.{rbIdx[i]}.block.3.conv.bias", outCh);
        }
        w.AddTensor($"{d}.11.conv.weight", s, 1, 64, 3); w.AddBias($"{d}.11.conv.bias", 1);
        PocketTtsMimiDecoder dec = new();
        dec.LoadWeights(w);
        Tensor latent = Seq(o, 760, 1f, 1, 32, 16);
        Tensor pcm = o.Own(dec.Forward(cpu, latent, 1, 16));
        return Finish(o, ("pcm", pcm));
    }

    // ── Qwen3-TTS vocoder pre-transformer layer (a private nested class, built by reflection) ───────────────────

    private static string Qwen3TtsLayer(CpuBackend cpu)
    {
        // Window 4 (-> 5 inclusive keys) against t = 12; then the t == 1 null-mask step.
        const int dim = 16, heads = 2, hd = 8, ffn = 24;
        Owner o = new();
        Qwen3TtsVocoderConfig cfg = new() { TransformerDim = dim, TransformerHeads = heads, TransformerHeadDim = hd, TransformerFfnDim = ffn, SlidingWindow = 4 };
        Weights w = new(75);
        w.AddNorm("l.input_layernorm.weight", dim); w.AddNorm("l.post_attention_layernorm.weight", dim);
        w.Linear("l.self_attn.q_proj", heads * hd, dim, true); w.Linear("l.self_attn.k_proj", heads * hd, dim, true);
        w.Linear("l.self_attn.v_proj", heads * hd, dim, true); w.Linear("l.self_attn.o_proj", dim, heads * hd, true);
        w.Add("l.mlp.gate_proj.weight", ffn, dim); w.Add("l.mlp.up_proj.weight", ffn, dim); w.Add("l.mlp.down_proj.weight", dim, ffn);
        w.AddTensor("l.self_attn_layer_scale.scale", 0.5f, dim); w.AddTensor("l.mlp_layer_scale.scale", 0.5f, dim);
        Type layerType = typeof(Qwen3TtsVocoder).GetNestedType("TransformerLayer", BindingFlags.NonPublic)!;
        object layer = Activator.CreateInstance(layerType, cfg)!;
        layerType.GetMethod("LoadWeights")!.Invoke(layer, [w, "l"]);
        MethodInfo forward = layerType.GetMethod("Forward")!;
        (float[] cos, float[] sin) = HartsyInference.Audio.Models.Moonshine.RotaryEmbedding.GetTables(hd, cfg.TransformerRopeTheta, 64);
        Tensor x = Seq(o, 750, 1f, 1, 12, dim), x1 = Seq(o, 751, 1f, 1, 1, dim);
        Tensor mask = o.Own(Qwen3TtsVocoder.BuildSlidingCausalMask(12, cfg.SlidingWindow));
        Tensor y = o.Own((Tensor)forward.Invoke(layer, [cpu, x, 12, mask, cos, sin])!);
        Tensor y1 = o.Own((Tensor)forward.Invoke(layer, [cpu, x1, 1, null, cos, sin])!);
        return Finish(o, ("windowed", y), ("t1", y1));
    }

    // ── StyleTTS2 diffusion transformer (self-attention over one token, cross-attention over the text context) ────

    private static string StyleTransformer(CpuBackend cpu)
    {
        const int c = 16, heads = 2, hf = 8, inner = heads * hf, mult = 2, ctxLen = 5;
        Owner o = new();
        Weights w = new(76);
        Tensor T(string name, long rows, long cols) { w.Add(name, rows, cols); return w[name]; }
        Tensor B(string name, long n) { w.AddBias(name, n); return w[name]; }
        Tensor N(string name) { w.AddNorm(name, c); return w[name]; }
        StyleTransformerLayer[] layers = new StyleTransformerLayer[2];
        for (int i = 0; i < layers.Length; i++)
        {
            string p = $"b{i}.";
            layers[i] = new StyleTransformerLayer
            {
                AdaLnW = T(p + "ada", 2 * c, c), AdaLnB = B(p + "ada.b", 2 * c),
                SelfNormW = N(p + "n1"), SelfNormB = B(p + "n1.b", c),
                SelfQW = T(p + "q", inner, c), SelfQB = B(p + "q.b", inner), SelfKW = T(p + "k", inner, c), SelfKB = B(p + "k.b", inner),
                SelfVW = T(p + "v", inner, c), SelfVB = B(p + "v.b", inner), SelfOW = T(p + "o", c, inner), SelfOB = B(p + "o.b", c),
                CrossNormW = N(p + "n2"), CrossNormB = B(p + "n2.b", c),
                CrossQW = T(p + "cq", inner, c), CrossQB = B(p + "cq.b", inner), CrossKW = T(p + "ck", inner, c), CrossKB = B(p + "ck.b", inner),
                CrossVW = T(p + "cv", inner, c), CrossVB = B(p + "cv.b", inner), CrossOW = T(p + "co", c, inner), CrossOB = B(p + "co.b", c),
                FfNormW = N(p + "n3"), FfNormB = B(p + "n3.b", c),
                FfW1 = T(p + "f1", c * mult, c), FfB1 = B(p + "f1.b", c * mult), FfW2 = T(p + "f2", c, c * mult), FfB2 = B(p + "f2.b", c),
            };
        }
        StyleTransformer1d net = new(c, heads, hf, c, mult, layers, T("tw1", c, c), B("tb1", c), T("tw2", c, c), B("tb2", c),
            T("inw", c, c), B("inb", c), T("outw", c, c), B("outb", c));
        Tensor refStyle = Seq(o, 761, 1f, 1, 1, c);
        net.SetReferenceStyle(refStyle);
        Tensor style = Seq(o, 762, 1f, 1, c), cond = Seq(o, 763, 1f, 1, ctxLen, c);
        Tensor y = o.Own(net.Forward(cpu, style, 0.7f, cond));
        Tensor yNoCond = o.Own(net.Forward(cpu, style, 0.7f, null));
        return Finish(o, ("cross", y), ("selfonly", yNoCond));
    }
}
