using System;
using System.Collections.Generic;
using System.IO;
using HartsyInference.Audio.Models.Demucs;
using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Per-stage parity debug for HTDemucs. Captures each HtDemucs DebugHook stage and diffs it against
/// demucs_stages.safetensors (dump_demucs_stages.py). Gated on <c>DEMUCS_WEIGHTS</c> + <c>DEMUCS_STAGES</c>
/// + <c>DEMUCS_REF</c> (for the input wav). Writes diffs to <c>DEMUCS_DEBUG_OUT</c>.</summary>
public sealed unsafe class DemucsStageDebugTests
{
    private readonly ITestOutputHelper _out;
    public DemucsStageDebugTests(ITestOutputHelper o) => _out = o;

    [Fact]
    [Trait("Category", "Integration")]
    public void DemucsStages()
    {
        string? wPath = Environment.GetEnvironmentVariable("DEMUCS_WEIGHTS");
        string? stagesP = Environment.GetEnvironmentVariable("DEMUCS_STAGES");
        string? refP = Environment.GetEnvironmentVariable("DEMUCS_REF");
        if (string.IsNullOrEmpty(wPath) || !File.Exists(wPath) || string.IsNullOrEmpty(stagesP) || !File.Exists(stagesP)
            || string.IsNullOrEmpty(refP) || !File.Exists(refP)) return;

        SafeTensorsLoader wl = new(); wl.Load(wPath);
        IReadOnlyDictionary<string, Tensor> w = wl.GetAllTensors();
        SafeTensorsLoader sl = new(); sl.Load(stagesP);
        IReadOnlyDictionary<string, Tensor> stages = sl.GetAllTensors();
        SafeTensorsLoader rl = new(); rl.Load(refP);
        Tensor wav = rl.GetAllTensors()["wav"];
        int length = (int)wav.Shape[2];

        HtDemucs model = new(HtDemucsConfig.Htdemucs);
        model.LoadWeights(w);

        List<string> lines = new();
        model.DebugHook = (k, x) =>
        {
            if (!stages.ContainsKey(k)) { lines.Add($"{k}: no ref"); return; }
            Tensor r = stages[k];
            string sh = x.ElementCount == r.ElementCount ? "" : $"  SHAPE mine {x.Shape} ref {r.Shape}";
            (double corr, double mx) = Compare(x, r);
            lines.Add($"{k}: corr={corr:F6} maxAbs={mx:E3}{sh}");
        };
        using CpuBackend backend = new();
        Tensor stems = model.Forward(backend, wav, length);
        (double fc, double fm) = Compare(stems, stages["stems"]);
        lines.Add($"FINAL stems: corr={fc:F6} maxAbs={fm:E3}");
        stems.Dispose();

        string dest = Environment.GetEnvironmentVariable("DEMUCS_DEBUG_OUT") ?? "/tmp/demucs_stages.txt";
        File.WriteAllLines(dest, lines);
        foreach (string l in lines) _out.WriteLine(l);
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
