using System;
using System.Collections.Generic;
using System.IO;
using HartsyInference.Audio.Models.Codecs.EnCodec;
using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelHandler.CheckpointConverters;
using HartsyInference.ModelHandler.SafeTensors;
using Xunit;
using Xunit.Abstractions;
using EnCodecModel = HartsyInference.Audio.Models.Codecs.EnCodec.EnCodec;

namespace HartsyInference.Audio.Tests;

/// <summary>Per-stage parity debug for the EnCodec 24 kHz causal decoder. Captures each
/// SeaNetDecoder DebugStageHook activation and diffs it against encodec24_stages.safetensors
/// (dump_encodec24_stages.py). Gated on <c>ENCODEC24_WEIGHTS</c> + <c>ENCODEC24_STAGES</c>.</summary>
public sealed unsafe class EnCodec24StageDebugTests
{
    private readonly ITestOutputHelper _out;
    public EnCodec24StageDebugTests(ITestOutputHelper o) => _out = o;

    [Fact]
    [Trait("Category", "Integration")]
    public void EnCodec24Decode_StageByStage()
    {
        string? wPath = Environment.GetEnvironmentVariable("ENCODEC24_WEIGHTS");
        string? stagesP = Environment.GetEnvironmentVariable("ENCODEC24_STAGES");
        string? refP = Environment.GetEnvironmentVariable("ENCODEC24_REF");
        if (string.IsNullOrEmpty(wPath) || !File.Exists(wPath) || string.IsNullOrEmpty(stagesP) || !File.Exists(stagesP)
            || string.IsNullOrEmpty(refP) || !File.Exists(refP))
            return;

        SafeTensorsLoader wl = new(); wl.Load(wPath);
        Dictionary<string, Tensor> w = MusicGenCheckpointConverter.MapEnCodecWeights(wl.GetAllTensors(), castToF32: true);
        SafeTensorsLoader sl = new(); sl.Load(stagesP);
        IReadOnlyDictionary<string, Tensor> stages = sl.GetAllTensors();
        SafeTensorsLoader rl = new(); rl.Load(refP);
        IReadOnlyDictionary<string, Tensor> r = rl.GetAllTensors();

        Tensor codesRef = r["codes"];
        int nq = (int)codesRef.Shape[0], batch = (int)codesRef.Shape[1], t = (int)codesRef.Shape[2];

        EnCodecModel codec = new(EnCodecConfig.EnCodec24kHz);
        codec.LoadWeights(w);

        Tensor codes = new(new TensorShape(nq, batch, t), DType.I32);
        int* cp = (int*)codes.DataPointer; int* cr = (int*)codesRef.DataPointer;
        for (long i = 0; i < codes.ElementCount; i++) cp[i] = cr[i];

        List<string> lines = new();
        codec.SetDecoderDebugHook((idx, x) =>
        {
            string key = $"stage_{idx:D2}";
            if (!stages.ContainsKey(key)) { lines.Add($"stage {idx}: no ref"); return; }
            Tensor refT = stages[key];
            (double corr, double mx) = Compare(x, refT);
            string shapeOk = x.ElementCount == refT.ElementCount ? "" : $"  SHAPE MINE {x.Shape} REF {refT.Shape}";
            lines.Add($"stage_{idx:D2}: corr={corr:F6} maxAbs={mx:E3}{shapeOk}");
        });

        using CpuBackend backend = new();
        Tensor pcm = codec.Decode(backend, codes, batch, t);
        (double fc, double fm) = Compare(pcm, stages["stage_15"]);
        lines.Add($"FINAL pcm: corr={fc:F6} maxAbs={fm:E3}");
        string dest = Environment.GetEnvironmentVariable("ENCODEC24_DEBUG_OUT") ?? "/tmp/ec24_stages.txt";
        File.WriteAllLines(dest, lines);
        foreach (string l in lines) _out.WriteLine(l);
        codes.Dispose(); pcm.Dispose();
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
