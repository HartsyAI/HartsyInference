using System;
using System.Collections.Generic;
using System.IO;
using HartsyInference.Audio.Models.Codecs.XCodec;
using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Numerical parity for the YuE x-codec (SoundStream) DECODE path against a standalone torch
/// reference re-derived from the real <c>ckpt_00360000.pth</c> (<c>codec_model</c> state, exported to
/// <c>xcodec.safetensors</c>). Validates, f32-vs-f32:
/// <list type="bullet">
///   <item>EMA-VQ ResidualVQ decode of a fixed code grid (<c>rvq_out</c>) — exact table gather + sum.</item>
///   <item>full codec decode (<c>full_out</c> = decoder_2(fc_post2(rvq_out))) — corr &gt; 0.99.</item>
/// </list>
/// Gated on <c>YUE_XCODEC_PATH</c> (the converted safetensors) + <c>YUE_XCODEC_REF_IO</c>
/// (<c>dump_yue_xcodec_reference.py</c> output). See PARITY_VERIFICATION.md.</summary>
public sealed unsafe class YueXCodecParityTests
{
    private readonly ITestOutputHelper _out;
    public YueXCodecParityTests(ITestOutputHelper o) => _out = o;

    [Fact]
    [Trait("Category", "Integration")]
    public void XCodecDecode_MatchesTorchReference()
    {
        string? ckpt = Environment.GetEnvironmentVariable("YUE_XCODEC_PATH");
        string? refIo = Environment.GetEnvironmentVariable("YUE_XCODEC_REF_IO");
        if (string.IsNullOrEmpty(ckpt) || !File.Exists(ckpt) || string.IsNullOrEmpty(refIo) || !File.Exists(refIo))
            return; // gated

        (Dictionary<string, Tensor> w, SafeTensorsLoader loader) = YueCheckpointConverter.LoadXCodec(ckpt, castToF32: true);
        using SafeTensorsLoader _ = loader;
        _out.WriteLine($"x-codec tensors after mapping: {w.Count}");

        XCodec codec = new(XCodecConfig.XCodec16kHz);
        codec.LoadWeights(w);

        SafeTensorsLoader io = new();
        io.Load(refIo);
        IReadOnlyDictionary<string, Tensor> d = io.GetAllTensors();
        Tensor codesRef = d["codes_i32"];     // [n_q, B, T] int32
        Tensor rvqRef = d["rvq_out"];        // [B, 1024, T]
        Tensor fullRef = d["full_out"];       // [B, 1, S]

        int nq = (int)codesRef.Shape[0];
        int batch = (int)codesRef.Shape[1];
        int t = (int)codesRef.Shape[2];
        _out.WriteLine($"codes [{nq},{batch},{t}]  rvq {rvqRef.Shape}  full {fullRef.Shape}");

        using CpuBackend backend = new();

        // ── EMA-VQ decode (exact) ──
        Tensor codes = new(new TensorShape(nq, batch, t), DType.I32);
        int* cp = (int*)codes.DataPointer;
        int* cr = (int*)codesRef.DataPointer;
        for (long i = 0; i < codes.ElementCount; i++) cp[i] = cr[i];

        XCodecEmaResidualVectorQuantizer rvq = new(12, 1024, 1024);
        rvq.LoadWeights(w, "quantizer");
        Tensor rvqMine = rvq.Decode(backend, codes, batch, t);
        double rvqMax = MaxAbs(rvqMine, rvqRef);
        _out.WriteLine($"EMA-VQ decode maxAbs = {rvqMax:E4}");
        Assert.True(rvqMax < 1e-4, $"EMA-VQ decode diverges (maxAbs={rvqMax:E4}).");

        // ── full codec decode (corr) ──
        Tensor pcm = codec.Decode(backend, codes, batch, t);
        _out.WriteLine($"my pcm {pcm.Shape}  (ref {fullRef.Shape})");
        Assert.Equal((int)fullRef.ElementCount, (int)pcm.ElementCount);
        (double corr, double rms, double maxAbs) = CorrRmsMax(pcm, fullRef);
        _out.WriteLine($"full decode corr={corr:F6}  rms(mine)={rms:E4}  maxAbs={maxAbs:E4}");
        Assert.True(corr > 0.99, $"full codec decode corr too low ({corr:F6}).");

        codes.Dispose();
        rvqMine.Dispose();
        pcm.Dispose();
    }

    private static double MaxAbs(Tensor a, Tensor b)
    {
        float* pa = (float*)a.DataPointer;
        float* pb = (float*)b.DataPointer;
        long n = Math.Min(a.ElementCount, b.ElementCount);
        double m = 0;
        for (long i = 0; i < n; i++) m = Math.Max(m, Math.Abs(pa[i] - pb[i]));
        return m;
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
