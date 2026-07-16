using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using HartsyInference.Audio.Models.PocketTts;
using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.Tokenizers;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Voiced end-to-end generation parity: the production AR path (LUT text conditioner + voice-KV-primed
/// backbone via <see cref="PocketTtsStreamingTransformer.ForwardPrimed"/> + out_eos + flow_net) with the SAME
/// noise the reference drew must reproduce the reference latents. Validates the new voice-priming infra +
/// tokenizer + conditioner that the low-level <see cref="PocketTtsGenParityTests"/> (random text_emb) can't.
/// Gated on <c>POCKET_W</c> (tts_*.safetensors) + <c>POCKET_VOICE</c> (alba.safetensors) + <c>POCKET_SPM</c>
/// (tokenizer.model) + <c>POCKET_PARITY</c> (pocket_parity_dump.py output).</summary>
public sealed unsafe class PocketTtsVoicedParityTests
{
    private readonly ITestOutputHelper _out;
    public PocketTtsVoicedParityTests(ITestOutputHelper o) => _out = o;

    [Fact]
    [Trait("Category", "Integration")]
    public void VoicedGeneration_MatchesReference()
    {
        string? wp = Env("POCKET_W"), vp = Env("POCKET_VOICE"), sp = Env("POCKET_SPM"), pp = Env("POCKET_PARITY");
        if (wp is null || vp is null || sp is null || pp is null) return;

        using SafeTensorsLoader wl = new(); wl.Load(wp);
        PocketTtsFlowLm flm = new("flow_lm"); flm.LoadWeights(wl.GetAllTensors());
        using SafeTensorsLoader vl = new(); vl.Load(vp);
        using PocketTtsVoice voice = PocketTtsVoice.Load(vl.GetAllTensors());
        PocketTtsTokenizer tok = new(sp);

        using SafeTensorsLoader pl = new(); pl.Load(pp);
        IReadOnlyDictionary<string, Tensor> r = pl.GetAllTensors();
        Tensor idsT = r["ids"];
        int nIds = (int)idsT.Shape[idsT.Shape.Rank - 1];
        long* idp = (long*)idsT.DataPointer;
        int[] wantIds = new int[nIds];
        for (int i = 0; i < nIds; i++) wantIds[i] = (int)idp[i];

        // Tokenizer parity: C# SentencePiece must reproduce the reference ids.
        int[] gotIds = tok.Encode("Hello there, this is a test of the Pocket T T S model.").ToArray();
        _out.WriteLine($"want ids: [{string.Join(",", wantIds)}]");
        _out.WriteLine($"got  ids: [{string.Join(",", gotIds)}]");
        Assert.Equal(wantIds, gotIds);

        Tensor noisesT = r["noises"];              // [N,32]
        int n = (int)noisesT.Shape[0], ld = (int)noisesT.Shape[1];
        float* np = (float*)noisesT.DataPointer;
        float[][] noises = new float[n][];
        for (int f = 0; f < n; f++) { noises[f] = new float[ld]; for (int i = 0; i < ld; i++) noises[f][i] = np[(long)f * ld + i]; }

        using CpuBackend backend = new();
        using Tensor textEmb = flm.EmbedText(gotIds);
        using Tensor lat = flm.GenerateVoiced(backend, textEmb, voice.PrefixK, voice.PrefixV, voice.PrefixLen,
            temp: 0.7f, noiseClamp: null, eosThreshold: -4.0f, framesAfterEos: 1_000_000, maxFrames: n, seed: 0,
            fixedNoises: noises);

        (double corr, double maxAbs) = CorrMax(lat, r["latents"]);
        _out.WriteLine($"latents corr={corr:F6}  maxAbs={maxAbs:E4}  frames={(int)lat.Shape[2]} (ref {n})");
        Assert.Equal(n, (int)lat.Shape[2]);
        Assert.True(corr > 0.999, $"voiced latent corr too low ({corr:F6}).");
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void PrimedHidden_MatchesReference()
    {
        string? wp = Env("POCKET_W"), vp = Env("POCKET_VOICE"), hp = Env("POCKET_HIDDEN");
        if (wp is null || vp is null || hp is null) return;

        using SafeTensorsLoader wl = new(); wl.Load(wp);
        PocketTtsFlowLm flm = new("flow_lm"); flm.LoadWeights(wl.GetAllTensors());
        using SafeTensorsLoader vl = new(); vl.Load(vp);
        using PocketTtsVoice voice = PocketTtsVoice.Load(vl.GetAllTensors());

        using SafeTensorsLoader hl = new(); hl.Load(hp);
        IReadOnlyDictionary<string, Tensor> r = hl.GetAllTensors();
        Tensor teRaw = r["text_emb"];                       // [...,17,1024]
        int tText = (int)teRaw.Shape[teRaw.Shape.Rank - 2];
        Tensor textEmb = new(new TensorShape(1, tText, 1024), DType.F32);
        Buffer.MemoryCopy((void*)teRaw.DataPointer, (void*)textEmb.DataPointer, (long)tText * 1024 * 4, (long)tText * 1024 * 4);
        float[] ilBos = new float[1024];
        float* ib = (float*)r["il_bos"].DataPointer;
        for (int i = 0; i < 1024; i++) ilBos[i] = ib[i];

        using CpuBackend backend = new();
        float[][] layerBos = new float[6][];
        float[] got = flm.DebugPrimedHidden(backend, textEmb, ilBos, voice.PrefixK, voice.PrefixV, voice.PrefixLen, layerBos);
        textEmb.Dispose();

        string? lp = Env("POCKET_LAYERS");
        if (lp is not null)
        {
            using SafeTensorsLoader ll = new(); ll.Load(lp);
            IReadOnlyDictionary<string, Tensor> lr = ll.GetAllTensors();
            for (int i = 0; i < 6; i++)
            {
                float* pl = (float*)lr[$"l{i}"].DataPointer;
                double c = Corr1024(layerBos[i], pl);
                _out.WriteLine($"layer {i} bos corr={c:F6}");
            }
        }

        float* refh = (float*)r["hidden0"].DataPointer;
        double sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0, mx = 0;
        for (int i = 0; i < 1024; i++)
        {
            double x = got[i], y = refh[i];
            sa += x; sb += y; saa += x * x; sbb += y * y; sab += x * y; mx = Math.Max(mx, Math.Abs(x - y));
        }
        double cov = sab - sa * sb / 1024, va = saa - sa * sa / 1024, vb = sbb - sb * sb / 1024;
        double corr = cov / (Math.Sqrt(va * vb) + 1e-12);
        _out.WriteLine($"hidden0 corr={corr:F6} maxAbs={mx:E4}");
        Assert.True(corr > 0.999, $"primed hidden corr too low ({corr:F6}).");
    }

    private static double Corr1024(float[] a, float* b)
    {
        double sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
        for (int i = 0; i < 1024; i++) { double x = a[i], y = b[i]; sa += x; sb += y; saa += x * x; sbb += y * y; sab += x * y; }
        double cov = sab - sa * sb / 1024, va = saa - sa * sa / 1024, vb = sbb - sb * sb / 1024;
        return cov / (Math.Sqrt(va * vb) + 1e-12);
    }

    private static string? Env(string k) { string? v = Environment.GetEnvironmentVariable(k); return string.IsNullOrEmpty(v) || !File.Exists(v) ? null : v; }

    private static (double, double) CorrMax(Tensor a, Tensor b)
    {
        float* pa = (float*)a.DataPointer; float* pb = (float*)b.DataPointer;
        long n = Math.Min(a.ElementCount, b.ElementCount);
        double sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0, mx = 0;
        for (long i = 0; i < n; i++)
        {
            double x = pa[i], y = pb[i];
            sa += x; sb += y; saa += x * x; sbb += y * y; sab += x * y;
            mx = Math.Max(mx, Math.Abs(x - y));
        }
        double cov = sab - sa * sb / n, va = saa - sa * sa / n, vb = sbb - sb * sb / n;
        return (cov / (Math.Sqrt(va * vb) + 1e-12), mx);
    }
}
