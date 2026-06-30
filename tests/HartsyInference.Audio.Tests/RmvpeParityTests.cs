using System;
using System.Collections.Generic;
using System.IO;
using HartsyInference.Audio.Models.Rvc;
using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelHandler.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Real-weight parity for RMVPE (RVC v2 pitch extractor) E2E network: a fixed random mel →
/// 360-bin pitch posterior, vs the upstream <c>E2E(4,1,(2,2))</c> (RVC-WebUI rmvpe.py). Isolates the network
/// from the mel front-end by feeding the oracle's exact mel. Gated on <c>RMVPE_WEIGHTS</c> + <c>RMVPE_REF</c>
/// (dump_rmvpe_reference.py); also writes per-stage diffs to <c>RMVPE_DEBUG_OUT</c>.</summary>
public sealed unsafe class RmvpeParityTests
{
    private readonly ITestOutputHelper _out;
    public RmvpeParityTests(ITestOutputHelper o) => _out = o;

    [Fact]
    [Trait("Category", "Integration")]
    public void RmvpeHidden_MatchesReference()
    {
        string? wPath = Environment.GetEnvironmentVariable("RMVPE_WEIGHTS");
        string? refP = Environment.GetEnvironmentVariable("RMVPE_REF");
        if (string.IsNullOrEmpty(wPath) || !File.Exists(wPath) || string.IsNullOrEmpty(refP) || !File.Exists(refP))
            return;

        SafeTensorsLoader wl = new(); wl.Load(wPath);
        IReadOnlyDictionary<string, Tensor> w = wl.GetAllTensors();
        SafeTensorsLoader rl = new(); rl.Load(refP);
        IReadOnlyDictionary<string, Tensor> r = rl.GetAllTensors();

        Tensor melRef = r["mel"];            // [1, 128, T]
        int melBins = (int)melRef.Shape[1];
        int frames = (int)melRef.Shape[2];

        RvcRmvpe model = new(RvcRmvpeConfig.Default);
        model.LoadWeights(w);

        List<string> lines = new();
        model.DebugHook = (k, x) =>
        {
            string key = k switch
            {
                "cnn" => "cnn_out",
                "gru" => "gru_out",
                "hidden" => "hidden",
                _ => k,
            };
            if (!r.ContainsKey(key)) return;
            (double corr, double mx) = Compare(x, r[key]);
            lines.Add($"{k}: corr={corr:F6} maxAbs={mx:E3}");
        };

        using CpuBackend backend = new();
        Tensor mel = new(new TensorShape(1, melBins, frames), DType.F32);
        Buffer.MemoryCopy((void*)melRef.DataPointer, (void*)mel.DataPointer, melRef.ElementCount * 4, melRef.ElementCount * 4);
        Tensor hidden = model.Hidden(backend, mel, melBins, frames);
        mel.Dispose();

        (double hc, double hm) = Compare(hidden, r["hidden"]);
        lines.Add($"FINAL hidden: corr={hc:F6} maxAbs={hm:E3}");
        string dest = Environment.GetEnvironmentVariable("RMVPE_DEBUG_OUT") ?? "/tmp/rmvpe_stages.txt";
        File.WriteAllLines(dest, lines);
        foreach (string l in lines) _out.WriteLine(l);

        hidden.Dispose();
        Assert.True(hc > 0.999, $"RMVPE hidden corr too low ({hc:F6}).");
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
