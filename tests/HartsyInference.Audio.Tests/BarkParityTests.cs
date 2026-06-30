using System;
using System.Collections.Generic;
using System.IO;
using HartsyInference.Audio.Models.Bark;
using HartsyInference.Audio.Models.LanguageModels.Gpt;
using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelHandler.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Real-weight parity for the Bark GPT cascade stages (semantic + coarse) vs `transformers` BarkModel
/// (suno/bark). Teacher-forced logits over a fixed token sequence; checks per-position greedy argmax (exact) and
/// logit correlation. Bark's EnCodec-24k codec is verified separately. Gated on <c>BARK_WEIGHTS</c> +
/// <c>BARK_REF</c> (dump_bark_reference.py).</summary>
public sealed unsafe class BarkParityTests
{
    private readonly ITestOutputHelper _out;
    public BarkParityTests(ITestOutputHelper o) => _out = o;

    [Fact]
    [Trait("Category", "Integration")]
    public void BarkStages_Logits_MatchReference()
    {
        string? wPath = Environment.GetEnvironmentVariable("BARK_WEIGHTS");
        string? refP = Environment.GetEnvironmentVariable("BARK_REF");
        if (string.IsNullOrEmpty(wPath) || !File.Exists(wPath) || string.IsNullOrEmpty(refP) || !File.Exists(refP))
            return;

        SafeTensorsLoader wl = new(); wl.Load(wPath);
        IReadOnlyDictionary<string, Tensor> w = wl.GetAllTensors();
        SafeTensorsLoader rl = new(); rl.Load(refP);
        IReadOnlyDictionary<string, Tensor> r = rl.GetAllTensors();

        using CpuBackend backend = new();
        CheckStage(backend, w, r, "semantic", 129_600, 10_048);
        CheckStage(backend, w, r, "coarse", 12_096, 12_096);
        CheckFine(backend, w, r);
    }

    private void CheckFine(CpuBackend backend, IReadOnlyDictionary<string, Tensor> w,
        IReadOnlyDictionary<string, Tensor> r)
    {
        using HartsyInference.Audio.Models.Bark.BarkFineModel fine = new(BarkConfig.Full);
        fine.LoadWeights(w, "fine_acoustics");

        Tensor gridT = r["fine_grid"];          // [1, T, 8]
        Tensor logitsRef = r["fine_logits"];    // [1, T, fineVocab]
        Tensor argmaxRef = r["fine_argmax"];    // [T]
        int cbIdx = ((int*)r["fine_cb_idx"].DataPointer)[0];
        int t = (int)gridT.Shape[1];
        int nc = (int)gridT.Shape[2];
        int outVocab = (int)logitsRef.Shape[2];
        int[,] codes = new int[nc, t];
        int* gp = (int*)gridT.DataPointer;
        for (int j = 0; j < t; j++)
            for (int c = 0; c < nc; c++) codes[c, j] = gp[(long)j * nc + c];

        Tensor logits = fine.DebugLogits(backend, codes, cbIdx, t);
        int* am = (int*)argmaxRef.DataPointer;
        float* lp = (float*)logits.DataPointer;
        int hits = 0;
        for (int pos = 0; pos < t; pos++)
        {
            long off = (long)pos * outVocab;
            int a = 0; float best = lp[off];
            for (int k = 1; k < outVocab; k++) { float v = lp[off + k]; if (v > best) { best = v; a = k; } }
            if (a == am[pos]) hits++;
        }
        (double corr, double mx) = Compare(logits, logitsRef);
        _out.WriteLine($"Bark fine(cb={cbIdx}): corr={corr:F6} maxAbs={mx:E3}  argmax {hits}/{t}");
        logits.Dispose();
        Assert.Equal(t, hits);
        Assert.True(corr > 0.999, $"Bark fine logits corr too low ({corr:F6}).");
    }

    private void CheckStage(CpuBackend backend, IReadOnlyDictionary<string, Tensor> w,
        IReadOnlyDictionary<string, Tensor> r, string name, int inVocab, int outVocab)
    {
        string prefix = name == "coarse" ? "coarse_acoustics" : name;
        using BarkCausalStage stage = new(GptConfig.BarkFull, inVocab, outVocab);
        stage.LoadWeights(w, prefix);

        Tensor idsT = r[$"{name}_ids"];
        Tensor logitsRef = r[$"{name}_logits"];
        Tensor argmaxRef = r[$"{name}_argmax"];
        int t = (int)idsT.Shape[1];
        int* ip = (int*)idsT.DataPointer;
        List<int> ids = new(t);
        for (int i = 0; i < t; i++) ids.Add(ip[i]);

        Tensor logits = stage.ForwardLogits(backend, ids);
        int* am = (int*)argmaxRef.DataPointer;
        float* lp = (float*)logits.DataPointer;
        int hits = 0;
        for (int pos = 0; pos < t; pos++)
        {
            long off = (long)pos * outVocab;
            int a = 0; float best = lp[off];
            for (int k = 1; k < outVocab; k++) { float v = lp[off + k]; if (v > best) { best = v; a = k; } }
            if (a == am[pos]) hits++;
        }
        (double corr, double mx) = Compare(logits, logitsRef);
        _out.WriteLine($"Bark {name}: corr={corr:F6} maxAbs={mx:E3}  argmax {hits}/{t}");
        logits.Dispose();
        Assert.Equal(t, hits);
        Assert.True(corr > 0.999, $"Bark {name} logits corr too low ({corr:F6}).");
    }

    private static (double corr, double mx) Compare(Tensor mine, Tensor refT)
    {
        long n = Math.Min(mine.ElementCount, refT.ElementCount);
        float* a = (float*)mine.DataPointer; float* b = (float*)refT.DataPointer;
        double sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0, mx = 0;
        for (long i = 0; i < n; i++)
        {
            double x = a[i], y = b[i];
            sa += x; sb += y; saa += x * x; sbb += y * y; sab += x * y;
            mx = Math.Max(mx, Math.Abs(x - y));
        }
        double cov = sab - sa * sb / n, va = saa - sa * sa / n, vb = sbb - sb * sb / n;
        return (cov / (Math.Sqrt(va * vb) + 1e-12), mx);
    }
}
