using System;
using System.Collections.Generic;
using System.IO;
using HartsyInference.Audio.Models.Codecs.EnCodec;
using HartsyInference.Audio.Models.Music;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Numerical parity for MusicGen-small (<c>facebook/musicgen-small</c>) against an upstream HF
/// <c>transformers</c> reference. Validates the three risky pieces in isolation, each against a fixed-input
/// dump (<c>refs.safetensors</c>, produced by <c>tests/python-reference/dump_musicgen_reference.py</c>):
/// <list type="number">
///   <item><b>T5-base text encoder</b> — hidden states for a fixed prompt (<c>t5_ids</c>/<c>t5_mask</c> →
///   <c>t5_hidden</c>). T5 v1.0 needs the non-gated ReLU FFN AND <b>no</b> 1/√d attention scaling.</item>
///   <item><b>Decoder logits</b> under teacher-forcing — a fixed delayed code grid (<c>dec_codes</c>) +
///   pre-projected cross states (<c>dec_cross</c>, already enc_to_dec'd to hidden=1024) → per-codebook
///   logits (<c>dec_logits</c>, last timestep). Validates the AR core without greedy cascade.</item>
///   <item><b>EnCodec 32 kHz decode</b> — a fixed 4-codebook grid (<c>codes</c>) → waveform
///   (<c>encodec_wav</c>). Exercises the reflect-padded non-causal SEANet + RVQ.</item>
/// </list>
///
/// <para>Gated on <c>MUSICGEN_PATH</c> (the combined <c>model.safetensors</c>; defaults to the HF cache copy)
/// + <c>MUSICGEN_REF</c> (the reference dump). Skips cleanly when either is absent. Runs on CPU (F32).</para></summary>
public sealed unsafe class MusicGenParityTests
{
    private readonly ITestOutputHelper _out;
    public MusicGenParityTests(ITestOutputHelper o) => _out = o;

    private static string? FindCheckpoint()
    {
        string? p = Environment.GetEnvironmentVariable("MUSICGEN_PATH");
        if (!string.IsNullOrEmpty(p) && File.Exists(p)) return p;
        string cache = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".cache", "huggingface", "hub", "models--facebook--musicgen-small", "snapshots");
        if (Directory.Exists(cache))
        {
            string[] hits = Directory.GetFiles(cache, "model.safetensors", SearchOption.AllDirectories);
            if (hits.Length > 0) return hits[0];
        }
        return null;
    }

    private static string? FindRef()
    {
        string? r = Environment.GetEnvironmentVariable("MUSICGEN_REF");
        return !string.IsNullOrEmpty(r) && File.Exists(r) ? r : null;
    }

    private static (double maxAbs, double corr, double refPeak) Diff(float* a, float* b, long n)
    {
        double mx = 0, sx = 0, sy = 0, sxy = 0, sxx = 0, syy = 0, peak = 0;
        for (long i = 0; i < n; i++)
        {
            double da = a[i], db = b[i];
            mx = Math.Max(mx, Math.Abs(da - db));
            peak = Math.Max(peak, Math.Abs(db));
            sx += da; sy += db; sxy += da * db; sxx += da * da; syy += db * db;
        }
        double cov = sxy - sx * sy / n, vx = sxx - sx * sx / n, vy = syy - sy * sy / n;
        double corr = (vx > 0 && vy > 0) ? cov / Math.Sqrt(vx * vy) : 0;
        return (mx, corr, peak);
    }

    private static int[] ReadInts(Tensor t)
    {
        int n = (int)t.ElementCount; int[] o = new int[n];
        if (t.DType == DType.I64) { long* p = (long*)t.DataPointer; for (int i = 0; i < n; i++) o[i] = (int)p[i]; }
        else { int* p = (int*)t.DataPointer; for (int i = 0; i < n; i++) o[i] = p[i]; }
        return o;
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void T5_MatchesPythonReference()
    {
        string? ckpt = FindCheckpoint(); string? refp = FindRef();
        if (ckpt is null || refp is null) { _out.WriteLine("SKIPPED: MUSICGEN_PATH/MUSICGEN_REF not set."); return; }

        SafeTensorsLoader refL = new(); refL.Load(refp);
        IReadOnlyDictionary<string, Tensor> refs = refL.GetAllTensors();

        (Dictionary<string, Tensor> w, IDisposable l) = MusicGenCheckpointConverter.LoadTextEncoderAny(ckpt, castToF32: true);
        T5TextEncoder t5 = new(T5TextEncoderConfig.T5Base); t5.LoadWeights(w);
        using CpuBackend backend = new();

        int[] ids = ReadInts(refs["t5_ids"]);
        int[] mask = ReadInts(refs["t5_mask"]);
        Tensor outT = t5.Encode(backend, new[] { ids }, new[] { mask });
        Tensor refH = refs["t5_hidden"];
        (double mx, double corr, double peak) = Diff((float*)outT.DataPointer, (float*)refH.DataPointer, outT.ElementCount);
        _out.WriteLine($"T5 hidden {outT.Shape}: maxAbs={mx:E4} corr={corr:F6} (ref peak {peak:F4})");
        outT.Dispose(); GC.KeepAlive(l);
        Assert.True(mx < 5e-3, $"T5 hidden states diverge (maxAbs={mx:E4}).");
        Assert.True(corr > 0.9999, $"T5 hidden states corr too low ({corr:F6}).");
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void Decoder_TeacherForcedLogits_MatchPythonReference()
    {
        string? ckpt = FindCheckpoint(); string? refp = FindRef();
        if (ckpt is null || refp is null) { _out.WriteLine("SKIPPED: MUSICGEN_PATH/MUSICGEN_REF not set."); return; }

        SafeTensorsLoader refL = new(); refL.Load(refp);
        IReadOnlyDictionary<string, Tensor> refs = refL.GetAllTensors();

        (Dictionary<string, Tensor> w, IDisposable l) = MusicGenCheckpointConverter.LoadDecoderAny(ckpt, castToF32: true);
        MusicGenConfig cfg = MusicGenConfig.Small;
        MusicGenDecoder dec = new(cfg); dec.LoadWeights(w, prefix: "model.decoder");
        using CpuBackend backend = new();

        Tensor codesT = refs["dec_codes"]; // [1,K,T]
        int K = (int)codesT.Shape[1]; int T = (int)codesT.Shape[2];
        int[] codes = ReadInts(codesT);
        int[,] frames = new int[T, K];
        for (int t = 0; t < T; t++) for (int k = 0; k < K; k++) frames[t, k] = codes[(0 * K + k) * T + t];
        Tensor emb = dec.EmbedFrames(frames);

        Tensor refCross = refs["dec_cross"]; // [1,S,1024] pre-projected
        Tensor cross = new(new TensorShape(1, (int)refCross.Shape[1], cfg.Hidden), DType.F32);
        Buffer.MemoryCopy((void*)refCross.DataPointer, (void*)cross.DataPointer, refCross.ElementCount * 4, refCross.ElementCount * 4);

        float[][] logits = dec.Forward(backend, emb, cross); // last-step [K][vocab]
        Tensor refLogits = refs["dec_logits"]; // [K,T,vocab]
        int V = cfg.CodebookSize;
        double worstMax = 0, worstCorr = 1;
        for (int k = 0; k < K; k++)
        {
            float* rk = (float*)refLogits.DataPointer + ((long)k * T + (T - 1)) * V;
            fixed (float* mk = logits[k])
            {
                (double mx, double corr, double peak) = Diff(mk, rk, V);
                _out.WriteLine($"cb{k} last-step: maxAbs={mx:E4} corr={corr:F6} (ref peak {peak:F4})");
                worstMax = Math.Max(worstMax, mx); worstCorr = Math.Min(worstCorr, corr);
            }
        }
        emb.Dispose(); cross.Dispose(); GC.KeepAlive(l);
        // Logits over 24 layers carry large magnitude — a few 1e-2 abs is expected; correlation is the gate.
        Assert.True(worstCorr > 0.9999, $"decoder logits corr too low ({worstCorr:F6}).");
        Assert.True(worstMax < 0.1, $"decoder logits maxAbs too high ({worstMax:E4}).");
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void EnCodecDecode_MatchesPythonReference()
    {
        string? ckpt = FindCheckpoint(); string? refp = FindRef();
        if (ckpt is null || refp is null) { _out.WriteLine("SKIPPED: MUSICGEN_PATH/MUSICGEN_REF not set."); return; }

        SafeTensorsLoader refL = new(); refL.Load(refp);
        IReadOnlyDictionary<string, Tensor> refs = refL.GetAllTensors();

        (Dictionary<string, Tensor> w, IDisposable l) = MusicGenCheckpointConverter.LoadEnCodecAny(ckpt, castToF32: true);
        EnCodec codec = new(EnCodecConfig.EnCodec32kHz); codec.LoadWeights(w);
        using CpuBackend backend = new();

        Tensor codesT = refs["codes"]; // [1,K,T] = [B,K,T]
        int K = (int)codesT.Shape[1]; int T = (int)codesT.Shape[2]; int B = 1;
        int[] codes = ReadInts(codesT);
        Tensor codesKBT = new(new TensorShape(K, B, T), DType.I32);
        int* cp = (int*)codesKBT.DataPointer;
        for (int k = 0; k < K; k++) for (int t = 0; t < T; t++) cp[(k * B + 0) * T + t] = codes[(0 * K + k) * T + t];

        Tensor pcm = codec.Decode(backend, codesKBT, B, T); // [B,1,T*640]
        Tensor refW = refs["encodec_wav"];
        long n = Math.Min(pcm.ElementCount, refW.ElementCount);
        (double mx, double corr, double peak) = Diff((float*)pcm.DataPointer, (float*)refW.DataPointer, n);
        _out.WriteLine($"EnCodec wav {pcm.Shape} (ref {refW.ElementCount}): maxAbs={mx:E4} corr={corr:F6} (ref peak {peak:F4})");
        codesKBT.Dispose(); pcm.Dispose(); GC.KeepAlive(l);
        Assert.Equal(refW.ElementCount, pcm.ElementCount);
        Assert.True(corr > 0.99, $"EnCodec decode corr too low ({corr:F6}).");
        Assert.True(mx < 1e-2, $"EnCodec decode maxAbs too high ({mx:E4}).");
    }
}
