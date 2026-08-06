using System;
using System.Collections.Generic;
using System.IO;
using HartsyInference.Audio.Models.Codecs.Snac;
using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Numerical parity for the SNAC 24 kHz DECODE path (the codec Orpheus consumes) against a torch
/// oracle built from the real <c>hubertsiuzdak/snac_24khz</c> weights (<c>dump_snac_reference.py</c>). The
/// reference decode stubs <c>torch.randn</c> to zeros so the stochastic NoiseBlock is deterministic; we match
/// it by decoding with <c>NoiseScale = 0</c>. This exercises the depthwise initial split, the NoiseBlock conv
/// load, residual <c>groups</c>, the hierarchical RVQ decode, and the final Tanh.
///
/// <para>Gated on <c>SNAC_WEIGHTS_PATH</c> (the dumped <c>snac_24khz.safetensors</c>) and <c>SNAC_REF_IO</c>
/// (<c>snac_ref_io.safetensors</c>). Skips cleanly when unset. See PARITY_VERIFICATION.md.</para></summary>
public sealed unsafe class SnacParityTests
{
    private readonly ITestOutputHelper _out;
    public SnacParityTests(ITestOutputHelper o) => _out = o;

    [Fact]
    [Trait("Category", "Integration")]
    public void Snac24kHzDecode_MatchesTorchReference()
    {
        string? wPath = Environment.GetEnvironmentVariable("SNAC_WEIGHTS_PATH");
        string? refIo = Environment.GetEnvironmentVariable("SNAC_REF_IO");
        if (string.IsNullOrEmpty(wPath) || !File.Exists(wPath) || string.IsNullOrEmpty(refIo) || !File.Exists(refIo))
            return; // gated

        SafeTensorsLoader wl = new();
        wl.Load(wPath);
        IReadOnlyDictionary<string, Tensor> w = wl.GetAllTensors();
        _out.WriteLine($"weights tensors: {w.Count}");

        Snac codec = new(SnacConfig.Snac24kHz with { NoiseScale = 0f });
        codec.LoadWeights(w);

        SafeTensorsLoader io = new();
        io.Load(refIo);
        IReadOnlyDictionary<string, Tensor> d = io.GetAllTensors();

        int nCb = (int)d["n_codebooks"].Shape[0] == 1 ? ((int*)d["n_codebooks"].DataPointer)[0] : 3;
        List<Tensor> codes = new(nCb);
        for (int i = 0; i < nCb; i++)
        {
            Tensor src = d[$"codes_{i}"];           // [1, len] int32
            int len = (int)src.ElementCount;
            Tensor c = new(new TensorShape(1, len), DType.I32);
            int* sp = (int*)src.DataPointer;
            int* cp = (int*)c.DataPointer;
            for (int j = 0; j < len; j++) cp[j] = sp[j];
            codes.Add(c);
            _out.WriteLine($"codes_{i} len={len}");
        }

        using CpuBackend backend = new();
        Tensor pcm = codec.Decode(backend, codes, batch: 1);
        Tensor audioRef = d["audio_out"];           // [1,1,S]
        _out.WriteLine($"my pcm {pcm.Shape}  ref {audioRef.Shape}");
        Assert.Equal((int)audioRef.ElementCount, (int)pcm.ElementCount);

        (double corr, double rms, double maxAbs) = CorrRmsMax(pcm, audioRef);
        _out.WriteLine($"SNAC decode corr={corr:F6}  rms(mine)={rms:E4}  maxAbs={maxAbs:E4}");
        Assert.True(corr > 0.999, $"SNAC decode corr too low ({corr:F6}).");
        Assert.True(maxAbs < 1e-3, $"SNAC decode maxAbs too high ({maxAbs:E4}).");

        foreach (Tensor c in codes) c.Dispose();
        pcm.Dispose();
    }

    private static (double Corr, double Rms, double MaxAbs) CorrRmsMax(Tensor a, Tensor b)
    {
        float* pa = (float*)a.DataPointer;
        float* pb = (float*)b.DataPointer;
        long n = Math.Min(a.ElementCount, b.ElementCount);
        double sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0, sqMine = 0, maxAbs = 0;
        for (long i = 0; i < n; i++)
        {
            double x = pa[i], y = pb[i];
            sa += x; sb += y; saa += x * x; sbb += y * y; sab += x * y; sqMine += x * x;
            maxAbs = Math.Max(maxAbs, Math.Abs(x - y));
        }
        double cov = sab - sa * sb / n;
        double va = saa - sa * sa / n;
        double vb = sbb - sb * sb / n;
        double corr = cov / (Math.Sqrt(va * vb) + 1e-12);
        return (corr, Math.Sqrt(sqMine / n), maxAbs);
    }
}
