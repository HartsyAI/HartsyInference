using System;
using System.Collections.Generic;
using System.IO;
using HartsyInference.Audio.Models.PocketTts;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelHandler.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Parity for the Pocket-TTS FlowLM backbone (Phase B) and flow_net head (Phase C) against the
/// kyutai-labs/pocket-tts reference. Gated on <c>POCKETTTS_WEIGHTS</c> + <c>POCKETTTS_FLOWLM_REF</c>
/// (<c>dump_pockettts_flowlm.py</c>).</summary>
public sealed unsafe class PocketTtsFlowLmParityTests
{
    private readonly ITestOutputHelper _out;
    public PocketTtsFlowLmParityTests(ITestOutputHelper o) => _out = o;

    private static (IReadOnlyDictionary<string, Tensor> w, IReadOnlyDictionary<string, Tensor> r)? Load()
    {
        string? wPath = Environment.GetEnvironmentVariable("POCKETTTS_WEIGHTS");
        string? refP = Environment.GetEnvironmentVariable("POCKETTTS_FLOWLM_REF");
        if (string.IsNullOrEmpty(wPath) || !File.Exists(wPath) || string.IsNullOrEmpty(refP) || !File.Exists(refP))
            return null;
        SafeTensorsLoader wl = new(); wl.Load(wPath);
        SafeTensorsLoader rl = new(); rl.Load(refP);
        return (wl.GetAllTensors(), rl.GetAllTensors());
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void FlowLmBackbone_MatchesReference()
    {
        (IReadOnlyDictionary<string, Tensor> w, IReadOnlyDictionary<string, Tensor> r)? loaded = Load();
        if (loaded is null) return;
        (IReadOnlyDictionary<string, Tensor> w, IReadOnlyDictionary<string, Tensor> r) = loaded.Value;

        PocketTtsStreamingTransformer tr = new("flow_lm.transformer", dim: 1024, heads: 16, layers: 6, ffn: 4096, context: null, layerScale: false);
        tr.LoadWeights(w);
        Tensor onW = WhisperOps.EnsureF32(w["flow_lm.out_norm.weight"]);
        Tensor onB = WhisperOps.EnsureF32(w["flow_lm.out_norm.bias"]);

        Tensor input = r["b_input"];            // [1,S,1024]
        int s = (int)input.Shape[1];
        using CpuBackend backend = new();
        Tensor t1 = tr.Forward(backend, input, 1, s);
        Tensor normed = new(t1.Shape, DType.F32);
        backend.LayerNorm(normed, t1, onW, onB, 1e-5f);
        t1.Dispose();

        (double corr, double maxAbs) = CorrMax(normed, r["b_out"]);
        _out.WriteLine($"FlowLM backbone corr={corr:F6}  maxAbs={maxAbs:E4}");
        Assert.True(corr > 0.9999, $"backbone corr too low ({corr:F6}).");
        normed.Dispose();
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void FlowNet_MatchesReference()
    {
        (IReadOnlyDictionary<string, Tensor> w, IReadOnlyDictionary<string, Tensor> r)? loaded = Load();
        if (loaded is null) return;
        (IReadOnlyDictionary<string, Tensor> w, IReadOnlyDictionary<string, Tensor> r) = loaded.Value;

        PocketTtsFlowNet flow = new("flow_lm.flow_net");
        flow.LoadWeights(w);

        Tensor condT = r["c_cond"], xtT = r["c_xt"], sT = r["c_s"], tT = r["c_t"], velT = r["c_vel"];
        float[] cond = ToArray(condT);
        float[] xt = ToArray(xtT);
        float s = ((float*)sT.DataPointer)[0];
        float t = ((float*)tT.DataPointer)[0];

        float[] vel = flow.Forward(cond, s, t, xt);

        float* vr = (float*)velT.DataPointer;
        double sab = 0, saa = 0, sbb = 0, mx = 0, sa = 0, sb = 0;
        int n = vel.Length;
        for (int i = 0; i < n; i++)
        {
            double a = vel[i], b = vr[i];
            sa += a; sb += b; saa += a * a; sbb += b * b; sab += a * b;
            mx = Math.Max(mx, Math.Abs(a - b));
        }
        double cov = sab - sa * sb / n, va = saa - sa * sa / n, vb = sbb - sb * sb / n;
        double corr = cov / (Math.Sqrt(va * vb) + 1e-12);
        _out.WriteLine($"flow_net velocity corr={corr:F6}  maxAbs={mx:E4}");
        Assert.True(corr > 0.9999, $"flow_net corr too low ({corr:F6}).");
        Assert.True(mx < 1e-3, $"flow_net maxAbs too high ({mx:E4}).");
    }

    private static float[] ToArray(Tensor t)
    {
        long n = t.ElementCount;
        float[] a = new float[n];
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < n; i++) a[i] = p[i];
        return a;
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
