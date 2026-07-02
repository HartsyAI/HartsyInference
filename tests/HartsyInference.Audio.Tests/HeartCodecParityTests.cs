using System;
using System.Collections.Generic;
using System.IO;
using HartsyInference.Audio.Models.HeartMula;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelHandler.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Numerical parity for the HeartCodec decoder (48 kHz flow-matching codec) against the upstream
/// <c>heartlib.heartcodec</c>. The Python oracle (<c>tests/python-reference/dump_heartcodec_reference.py</c>)
/// loads the real f32 checkpoint, casts to f32, and dumps fixed-input → fixed-output tensors to
/// <c>heartcodec_ref.safetensors</c> for each risky piece:
/// <list type="number">
///   <item><b>RVQ decode</b> — <c>rvq_codes [8,T]</c> → <c>rvq_out [T,512]</c> (must be bit-exact).</item>
///   <item><b>cond emb</b> — <c>rvq_out</c> → <c>cond_emb [2T,512]</c> (Linear + nearest 2×).</item>
///   <item><b>estimator velocity</b> — cat(<c>est_x</c>, 0, <c>est_cond</c>) + <c>est_t</c> → <c>est_v [2T,256]</c>.</item>
///   <item><b>ScalarModel decode</b> — <c>scalar_in [128,L]</c> → <c>scalar_out [L·1920]</c> (corr &gt; 0.99).</item>
///   <item><b>full single-segment decode</b> — <c>full_codes</c> + fixed init noise + the same CFG Euler ODE →
///   <c>full_wav [2, samples]</c> (corr &gt; 0.99).</item>
/// </list>
///
/// <para>Gated on <c>HEARTCODEC_WEIGHTS_DIR</c> (a dir with the 2 sharded codec safetensors) +
/// <c>HEARTCODEC_REF</c> (the dumped reference). Runs on CPU (the codec is small, f32). Skips cleanly when
/// either is absent.</para></summary>
public sealed unsafe class HeartCodecParityTests
{
    private readonly ITestOutputHelper _out;
    public HeartCodecParityTests(ITestOutputHelper o) => _out = o;

    private static (double maxAbs, double corr) Diff(float* a, float* b, long n)
    {
        double mx = 0, sa = 0, sb = 0, sab = 0, saa = 0, sbb = 0;
        for (long i = 0; i < n; i++)
        {
            double x = a[i], y = b[i];
            double d = Math.Abs(x - y);
            if (d > mx) mx = d;
            sa += x; sb += y; sab += x * y; saa += x * x; sbb += y * y;
        }
        double cov = sab / n - (sa / n) * (sb / n);
        double va = saa / n - (sa / n) * (sa / n);
        double vb = sbb / n - (sb / n) * (sb / n);
        double corr = cov / (Math.Sqrt(va * vb) + 1e-12);
        return (mx, corr);
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void HeartCodec_MatchesHeartlibReference()
    {
        string? dir = Environment.GetEnvironmentVariable("HEARTCODEC_WEIGHTS_DIR");
        string? refP = Environment.GetEnvironmentVariable("HEARTCODEC_REF");
        if (string.IsNullOrEmpty(dir) || !Directory.Exists(dir) || string.IsNullOrEmpty(refP) || !File.Exists(refP))
            return;

        List<SafeTensorsLoader> loaders = new();
        Dictionary<string, Tensor> w = new();
        foreach (string f in Directory.GetFiles(dir, "*.safetensors"))
        {
            SafeTensorsLoader l = new(); l.Load(f); loaders.Add(l);
            foreach ((string k, Tensor tv) in l.GetAllTensors()) w[k] = tv;
        }

        using SafeTensorsLoader refLoader = new(); refLoader.Load(refP);
        IReadOnlyDictionary<string, Tensor> rd = refLoader.GetAllTensors();

        using CpuBackend backend = new();
        using HeartCodecDecoder codec = new(HeartMulaConfig.Oss3B);
        codec.LoadWeights(w);

        // ── 1. RVQ decode ──
        Tensor rvqCodes = rd["rvq_codes"];   // int64 [8,T]
        int t = (int)rvqCodes.Shape[1];
        int[,] codes = ReadCodes(rvqCodes);
        Tensor rvqOut = codec.RvqDecode(backend, codes, t);   // [1,T,512]
        Tensor rvqRef = rd["rvq_out"];
        (double mx0, double c0) = Diff((float*)rvqOut.DataPointer, (float*)rvqRef.DataPointer, rvqRef.ElementCount);
        _out.WriteLine($"rvq_out: maxAbs={mx0:E3} corr={c0:F6}");
        Assert.True(mx0 < 1e-3, $"RVQ decode maxAbs {mx0:E3} too large.");

        // ── 2. cond emb (Linear + nearest 2x) ──
        Tensor cond = codec.CondEmb(backend, rvqOut, t);     // [1,2T,512]
        rvqOut.Dispose();
        Tensor condRef = rd["cond_emb"];
        (double mx1, double c1) = Diff((float*)cond.DataPointer, (float*)condRef.DataPointer, condRef.ElementCount);
        _out.WriteLine($"cond_emb: maxAbs={mx1:E3} corr={c1:F6}");
        cond.Dispose();
        Assert.True(mx1 < 5e-3, $"cond_emb maxAbs {mx1:E3} too large.");

        // ── 3. estimator velocity ──
        // Build estimator input [1, 2T, 1024] = cat(est_x[2T,256], zeros[2T,256], est_cond[2T,512]).
        Tensor estX = rd["est_x"]; Tensor estCond = rd["est_cond"]; Tensor estT = rd["est_t"];
        int t2 = (int)estX.Shape[0];
        Tensor estIn = new(new TensorShape(1, t2, 1024), DType.F32);
        float* ei = (float*)estIn.DataPointer; float* ex = (float*)estX.DataPointer; float* ec = (float*)estCond.DataPointer;
        for (int i = 0; i < t2; i++)
        {
            long o = (long)i * 1024;
            for (int c = 0; c < 256; c++) ei[o + c] = ex[(long)i * 256 + c];
            for (int c = 0; c < 256; c++) ei[o + 256 + c] = 0f;
            for (int c = 0; c < 512; c++) ei[o + 512 + c] = ec[(long)i * 512 + c];
        }
        float estTime = ((float*)estT.DataPointer)[0];
        Tensor estV = codec.EstimatorForward(backend, estIn, [estTime]);
        estIn.Dispose();
        Tensor estVRef = rd["est_v"];
        (double mx2, double c2) = Diff((float*)estV.DataPointer, (float*)estVRef.DataPointer, estVRef.ElementCount);
        _out.WriteLine($"est_v: maxAbs={mx2:E3} corr={c2:F6}");
        estV.Dispose();
        Assert.True(c2 > 0.999, $"estimator velocity corr {c2:F4} too low.");

        // ── 4. ScalarModel decode ──
        Tensor scalarIn = rd["scalar_in"];   // [128, L]
        int L = (int)scalarIn.Shape[1];
        Tensor sclBCL = new(new TensorShape(1, 128, L), DType.F32);
        Buffer.MemoryCopy((void*)scalarIn.DataPointer, (void*)sclBCL.DataPointer, scalarIn.ElementCount * 4, scalarIn.ElementCount * 4);
        Tensor scalarOut = codec.ScalarDecode(backend, sclBCL);   // [1,1,L*1920]
        sclBCL.Dispose();
        Tensor scalarRef = rd["scalar_out"];
        (double mx3, double c3) = Diff((float*)scalarOut.DataPointer, (float*)scalarRef.DataPointer, scalarRef.ElementCount);
        _out.WriteLine($"scalar_out: maxAbs={mx3:E3} corr={c3:F6}");
        scalarOut.Dispose();
        Assert.True(c3 > 0.99, $"ScalarModel decode corr {c3:F4} too low.");

        // ── 5. full single-segment decode (CFG Euler from a fixed init noise) ──
        Tensor fullCodes = rd["full_codes"]; int tf = (int)fullCodes.Shape[1];
        int[,] fcodes = ReadCodes(fullCodes);
        Tensor initNoise = rd["full_init_noise"];   // [2tf, 256]
        Tensor wavRef = rd["full_wav"];             // [2, samples]
        float[][] stereo = codec.DecodeSegmentStereo(backend, fcodes, initNoise);   // oracle's exact init noise
        int samples = (int)wavRef.Shape[1];
        float* wr = (float*)wavRef.DataPointer;
        for (int ch = 0; ch < 2; ch++)
        {
            fixed (float* sp = stereo[ch])
            {
                (double mxf, double cf) = Diff(sp, wr + (long)ch * samples, samples);
                _out.WriteLine($"full_wav[ch{ch}]: maxAbs={mxf:E3} corr={cf:F6}");
                Assert.True(cf > 0.99, $"full decode ch{ch} corr {cf:F4} too low.");
            }
        }

        foreach (SafeTensorsLoader l in loaders) l.Dispose();
    }

    private static int[,] ReadCodes(Tensor codes)
    {
        int q = (int)codes.Shape[0], t = (int)codes.Shape[1];
        int[,] outp = new int[q, t];
        long* p = (long*)codes.DataPointer;
        for (int i = 0; i < q; i++) for (int j = 0; j < t; j++) outp[i, j] = (int)p[(long)i * t + j];
        return outp;
    }
}
