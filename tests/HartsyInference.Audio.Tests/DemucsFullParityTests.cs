using System;
using System.Collections.Generic;
using System.IO;
using HartsyInference.Audio.Models.Demucs;
using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelHandler.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Full-model parity for HTDemucs v4 (4-stem separation) vs the demucs package on real weights.
/// Gated on <c>DEMUCS_WEIGHTS</c> + <c>DEMUCS_REF</c> (dump_demucs_reference.py, use_train_segment=False).</summary>
public sealed unsafe class DemucsFullParityTests
{
    private readonly ITestOutputHelper _out;
    public DemucsFullParityTests(ITestOutputHelper o) => _out = o;

    [Fact]
    [Trait("Category", "Integration")]
    public void DemucsFull_MatchesReference()
    {
        string? wPath = Environment.GetEnvironmentVariable("DEMUCS_WEIGHTS");
        string? refP = Environment.GetEnvironmentVariable("DEMUCS_REF");
        if (string.IsNullOrEmpty(wPath) || !File.Exists(wPath) || string.IsNullOrEmpty(refP) || !File.Exists(refP))
            return;

        SafeTensorsLoader wl = new(); wl.Load(wPath);
        IReadOnlyDictionary<string, Tensor> w = wl.GetAllTensors();
        SafeTensorsLoader rl = new(); rl.Load(refP);
        IReadOnlyDictionary<string, Tensor> r = rl.GetAllTensors();

        Tensor wav = r["wav"];                  // [1, C, L]
        Tensor stemsRef = r["stems"];           // [1, srcs, C, L]
        int channels = (int)wav.Shape[1];
        int length = (int)wav.Shape[2];

        HtDemucs model = new(HtDemucsConfig.Htdemucs);
        model.LoadWeights(w);
        using CpuBackend backend = new();
        Tensor stems = model.Forward(backend, wav, length);
        _out.WriteLine($"my stems {stems.Shape}  ref {stemsRef.Shape}");
        Assert.Equal((int)stemsRef.ElementCount, (int)stems.ElementCount);

        (double corr, double mx) = Compare(stems, stemsRef);
        _out.WriteLine($"Demucs full corr={corr:F6} maxAbs={mx:E3}");
        stems.Dispose();
        Assert.True(corr > 0.99, $"Demucs full corr too low ({corr:F6}).");
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
