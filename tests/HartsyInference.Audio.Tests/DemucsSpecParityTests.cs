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

/// <summary>Real-reference parity for the HTDemucs STFT + complex-as-channels front-end (<see cref="DemucsSpec"/>),
/// the bit-exact-risky piece, vs demucs's <c>HTDemucs._spec</c> (htdemucs v4). Gated on <c>DEMUCS_REF</c>
/// (dump_demucs_reference.py).</summary>
public sealed unsafe class DemucsSpecParityTests
{
    private readonly ITestOutputHelper _out;
    public DemucsSpecParityTests(ITestOutputHelper o) => _out = o;

    [Fact]
    [Trait("Category", "Integration")]
    public void DemucsSpec_MatchesReference()
    {
        string? refP = Environment.GetEnvironmentVariable("DEMUCS_REF");
        if (string.IsNullOrEmpty(refP) || !File.Exists(refP)) return;

        SafeTensorsLoader rl = new(); rl.Load(refP);
        IReadOnlyDictionary<string, Tensor> r = rl.GetAllTensors();
        Tensor wav = r["wav"];                  // [1, C, L]
        Tensor specRe = r["spec_re"];           // [1, C, Fq, T]
        Tensor specIm = r["spec_im"];
        int channels = (int)wav.Shape[1];
        int length = (int)wav.Shape[2];
        int fq = (int)specRe.Shape[2];
        int tt = (int)specRe.Shape[3];

        using CpuBackend backend = new();
        Tensor cac = DemucsSpec.Spec(backend, wav, channels, length, 4096, 1024, out int freq, out int time);
        _out.WriteLine($"my cac {cac.Shape}  ref [1,{2 * channels},{fq},{tt}]");
        Assert.Equal(fq, freq);
        Assert.Equal(tt, time);

        // Build the reference cac: channel 2c = real, 2c+1 = imag.
        float* cp = (float*)cac.DataPointer;
        float* rp = (float*)specRe.DataPointer;
        float* ip = (float*)specIm.DataPointer;
        double sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0, mx = 0; long n = 0;
        for (int c = 0; c < channels; c++)
            for (int k = 0; k < freq; k++)
                for (int t = 0; t < time; t++)
                {
                    double myRe = cp[(((long)(2 * c) * freq + k) * time) + t];
                    double myIm = cp[(((long)(2 * c + 1) * freq + k) * time) + t];
                    double reRef = rp[(((long)c * fq + k) * tt) + t];
                    double imRef = ip[(((long)c * fq + k) * tt) + t];
                    foreach ((double x, double y) in new[] { (myRe, reRef), (myIm, imRef) })
                    {
                        sa += x; sb += y; saa += x * x; sbb += y * y; sab += x * y;
                        mx = Math.Max(mx, Math.Abs(x - y)); n++;
                    }
                }
        double cov = sab - sa * sb / n, va = saa - sa * sa / n, vb = sbb - sb * sb / n;
        double corr = cov / (Math.Sqrt(va * vb) + 1e-12);
        _out.WriteLine($"Demucs spec corr={corr:F6} maxAbs={mx:E3}");
        cac.Dispose();
        Assert.True(corr > 0.999, $"Demucs spec corr too low ({corr:F6}).");
    }
}
