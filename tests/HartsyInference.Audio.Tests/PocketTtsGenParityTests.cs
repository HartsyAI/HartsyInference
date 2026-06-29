using System;
using System.Collections.Generic;
using System.IO;
using HartsyInference.Audio.Models.PocketTts;
using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelHandler.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Phase D: deterministic end-to-end Pocket-TTS generation parity. With injected fixed noise, the AR
/// loop (input_linear + bos + backbone + flow_net lsd_decode) + Mimi decode must match the reference exactly.
/// Gated on <c>POCKETTTS_WEIGHTS</c> + <c>POCKETTTS_GEN_REF</c> (<c>dump_pockettts_gen.py</c>).</summary>
public sealed unsafe class PocketTtsGenParityTests
{
    private readonly ITestOutputHelper _out;
    public PocketTtsGenParityTests(ITestOutputHelper o) => _out = o;

    [Fact]
    [Trait("Category", "Integration")]
    public void FullGeneration_MatchesReference()
    {
        string? wPath = Environment.GetEnvironmentVariable("POCKETTTS_WEIGHTS");
        string? refP = Environment.GetEnvironmentVariable("POCKETTTS_GEN_REF");
        if (string.IsNullOrEmpty(wPath) || !File.Exists(wPath) || string.IsNullOrEmpty(refP) || !File.Exists(refP))
            return;

        SafeTensorsLoader wl = new(); wl.Load(wPath);
        IReadOnlyDictionary<string, Tensor> w = wl.GetAllTensors();
        SafeTensorsLoader rl = new(); rl.Load(refP);
        IReadOnlyDictionary<string, Tensor> r = rl.GetAllTensors();

        PocketTtsFlowLm flm = new("flow_lm");
        flm.LoadWeights(w);
        PocketTtsMimiDecoder dec = new("mimi");
        dec.LoadWeights(w);

        Tensor textEmb = r["text_emb"];        // [1,T,1024]
        Tensor noisesT = r["noises"];          // [N,32]
        int n = (int)noisesT.Shape[0];
        int ld = (int)noisesT.Shape[1];
        float[][] noises = new float[n][];
        float* npp = (float*)noisesT.DataPointer;
        for (int f = 0; f < n; f++)
        {
            noises[f] = new float[ld];
            for (int i = 0; i < ld; i++) noises[f][i] = npp[(long)f * ld + i];
        }

        using CpuBackend backend = new();
        Tensor lat = flm.GenerateLatents(backend, textEmb, noises);
        (double latCorr, double latMax) = CorrMax(lat, r["latents"]);
        _out.WriteLine($"latents corr={latCorr:F6}  maxAbs={latMax:E4}");
        Assert.True(latCorr > 0.9999, $"latent corr too low ({latCorr:F6}).");

        Tensor audio = dec.Forward(backend, lat, 1, n);
        (double aCorr, double aMax) = CorrMax(audio, r["audio_out"]);
        _out.WriteLine($"audio corr={aCorr:F6}  maxAbs={aMax:E4}");
        Assert.True(aCorr > 0.999, $"audio corr too low ({aCorr:F6}).");
        lat.Dispose(); audio.Dispose();
    }

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
