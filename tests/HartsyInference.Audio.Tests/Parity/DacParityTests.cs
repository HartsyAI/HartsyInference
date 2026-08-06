using System;
using System.Collections.Generic;
using System.IO;
using HartsyInference.Audio.Models.Codecs.Dac;
using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;
using Xunit.Abstractions;
using DacModel = HartsyInference.Audio.Models.Codecs.Dac.Dac;

namespace HartsyInference.Audio.Tests;

/// <summary>Numerical parity for the Descript Audio Codec (DAC) 44.1 kHz DECODE path against a torch oracle
/// built from the real <c>descript-audio-codec</c> 44 kHz weights (<c>dump_dac_reference.py</c>). DAC underpins
/// Spark-TTS (BiCodec reuses <c>DacDecoder</c>), Dia, IndexTTS, and Higgs, so a standalone parity check pins the
/// factorized-RVQ decode + snake decoder + final tanh.
///
/// <para>Gated on <c>DAC_WEIGHTS_PATH</c> (<c>dac_44khz.safetensors</c>) + <c>DAC_REF_IO</c>
/// (<c>dac_ref_io.safetensors</c>). Skips cleanly when unset. See PARITY_VERIFICATION.md.</para></summary>
public sealed unsafe class DacParityTests
{
    private readonly ITestOutputHelper _out;
    public DacParityTests(ITestOutputHelper o) => _out = o;

    [Fact]
    [Trait("Category", "Integration")]
    public void Dac44kHzDecode_MatchesTorchReference()
    {
        string? wPath = Environment.GetEnvironmentVariable("DAC_WEIGHTS_PATH");
        string? refIo = Environment.GetEnvironmentVariable("DAC_REF_IO");
        if (string.IsNullOrEmpty(wPath) || !File.Exists(wPath) || string.IsNullOrEmpty(refIo) || !File.Exists(refIo))
            return; // gated

        SafeTensorsLoader wl = new();
        wl.Load(wPath);
        IReadOnlyDictionary<string, Tensor> w = wl.GetAllTensors();
        _out.WriteLine($"weights tensors: {w.Count}");

        DacModel codec = new(DacConfig.Dac44kHz);
        codec.LoadWeights(w);

        SafeTensorsLoader io = new();
        io.Load(refIo);
        IReadOnlyDictionary<string, Tensor> d = io.GetAllTensors();

        Tensor codesRef = d["codes"];        // [nQ, B, T] int32
        int nq = (int)codesRef.Shape[0];
        int batch = (int)codesRef.Shape[1];
        int t = (int)codesRef.Shape[2];
        _out.WriteLine($"codes [{nq},{batch},{t}]");

        Tensor codes = new(new TensorShape(nq, batch, t), DType.I32);
        int* cp = (int*)codes.DataPointer;
        int* cr = (int*)codesRef.DataPointer;
        for (long i = 0; i < codes.ElementCount; i++) cp[i] = cr[i];

        using CpuBackend backend = new();
        Tensor pcm = codec.Decode(backend, codes, batch, t);
        Tensor audioRef = d["audio_out"];
        _out.WriteLine($"my pcm {pcm.Shape}  ref {audioRef.Shape}");
        Assert.Equal((int)audioRef.ElementCount, (int)pcm.ElementCount);

        (double corr, double rms, double maxAbs) = CorrRmsMax(pcm, audioRef);
        _out.WriteLine($"DAC decode corr={corr:F6}  rms(mine)={rms:E4}  maxAbs={maxAbs:E4}");
        Assert.True(corr > 0.999, $"DAC decode corr too low ({corr:F6}).");
        Assert.True(maxAbs < 1e-3, $"DAC decode maxAbs too high ({maxAbs:E4}).");

        codes.Dispose();
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
