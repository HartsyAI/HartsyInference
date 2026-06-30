using System;
using System.Collections.Generic;
using System.IO;
using HartsyInference.Audio.Models.Csm;
using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelHandler.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Real-weight parity for the Sesame CSM-1B backbone + codebook-0 head vs `transformers`
/// CsmForConditionalGeneration (unsloth/csm-1b, converted to the sesame key layout). Builds an audio-frame
/// context from fixed codes, runs the 1B Llama backbone, and compares the c0 logits (greedy argmax + corr).
/// The Mimi codec is verified separately. Gated on <c>CSM_WEIGHTS</c> + <c>CSM_REF</c> (dump_csm_reference.py).</summary>
public sealed unsafe class CsmParityTests
{
    private readonly ITestOutputHelper _out;
    public CsmParityTests(ITestOutputHelper o) => _out = o;

    [Fact]
    [Trait("Category", "Integration")]
    public void CsmBackbone_C0Logits_MatchReference()
    {
        string? wPath = Environment.GetEnvironmentVariable("CSM_WEIGHTS");
        string? refP = Environment.GetEnvironmentVariable("CSM_REF");
        if (string.IsNullOrEmpty(wPath) || !File.Exists(wPath) || string.IsNullOrEmpty(refP) || !File.Exists(refP))
            return;

        SafeTensorsLoader wl = new(); wl.Load(wPath);
        IReadOnlyDictionary<string, Tensor> w = wl.GetAllTensors();
        SafeTensorsLoader rl = new(); rl.Load(refP);
        IReadOnlyDictionary<string, Tensor> r = rl.GetAllTensors();

        Tensor codesT = r["ctx_codes"];        // [N, 32]
        Tensor c0Ref = r["c0_logits"];         // [1, 2051]
        int refArgmax = ((int*)r["c0_argmax"].DataPointer)[0];
        int n = (int)codesT.Shape[0];
        int nc = (int)codesT.Shape[1];
        int* cp = (int*)codesT.DataPointer;

        CsmConfig cfg = CsmConfig.V1B;
        CsmModel model = new(cfg);
        model.LoadWeights(w);

        using CpuBackend backend = new();
        int bh = cfg.Backbone.HiddenSize;
        Tensor ctx = new(new TensorShape(1, n, bh), DType.F32);
        float* xp = (float*)ctx.DataPointer;
        int[] frame = new int[nc];
        for (int t = 0; t < n; t++)
        {
            for (int c = 0; c < nc; c++) frame[c] = cp[(long)t * nc + c];
            Tensor fe = model.EmbedAudioFrame(frame);
            Buffer.MemoryCopy((void*)fe.DataPointer, xp + (long)t * bh, bh * 4, bh * 4);
            fe.Dispose();
        }

        // forced = c0..c30 (31 codes); DebugFrameLogits wants length NumCodebooks (the last slot is unused).
        Tensor forcedT = r["forced"];
        int nf = (int)forcedT.Shape[0];
        int* ffp = (int*)forcedT.DataPointer;
        int[] forced = new int[cfg.NumCodebooks];
        for (int i = 0; i < nf && i < forced.Length; i++) forced[i] = ffp[i];

        (float[] c0, float[][] dec) = model.DebugFrameLogits(backend, ctx, forced);
        ctx.Dispose();
        model.Dispose();

        int vocab = cfg.AudioVocab;
        int myArgmax = 0; float best = c0[0];
        for (int k = 1; k < vocab; k++) if (c0[k] > best) { best = c0[k]; myArgmax = k; }
        (double corr, double mx) = Compare(c0, (float*)c0Ref.DataPointer, vocab);
        _out.WriteLine($"CSM c0: corr={corr:F6} maxAbs={mx:E3}  argmax mine={myArgmax} ref={refArgmax}");
        Assert.Equal(refArgmax, myArgmax);
        Assert.True(corr > 0.999, $"CSM c0 logits corr too low ({corr:F6}).");

        // Depth decoder: dec[j] predicts codebook j+1; compare to dec_logits[j] + argmax.
        Tensor decRef = r["dec_logits"];        // [1, 31, vocab]
        int* decAm = (int*)r["dec_argmax"].DataPointer;
        float* drp = (float*)decRef.DataPointer;
        int nCb = (int)decRef.Shape[1];
        double worstCorr = 1.0; int amHits = 0;
        for (int j = 0; j < nCb; j++)
        {
            int a = 0; float b2 = dec[j][0];
            for (int k = 1; k < vocab; k++) if (dec[j][k] > b2) { b2 = dec[j][k]; a = k; }
            if (a == decAm[j]) amHits++;
            (double dc, _) = Compare(dec[j], drp + (long)j * vocab, vocab);
            worstCorr = Math.Min(worstCorr, dc);
        }
        _out.WriteLine($"CSM depth decoder: worstCorr={worstCorr:F6}  argmax {amHits}/{nCb}");
        Assert.Equal(nCb, amHits);
        Assert.True(worstCorr > 0.999, $"CSM depth decoder corr too low ({worstCorr:F6}).");
    }

    private static (double corr, double mx) Compare(float[] a, float* b, int n)
    {
        double sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0, mx = 0;
        for (int i = 0; i < n; i++)
        {
            double x = a[i], y = b[i];
            sa += x; sb += y; saa += x * x; sbb += y * y; sab += x * y;
            mx = Math.Max(mx, Math.Abs(x - y));
        }
        double cov = sab - sa * sb / n, va = saa - sa * sa / n, vb = sbb - sb * sb / n;
        return (cov / (Math.Sqrt(va * vb) + 1e-12), mx);
    }
}
