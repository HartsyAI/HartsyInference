using System;
using System.Collections.Generic;
using System.IO;
using HartsyInference.Audio.Models.FishSpeech;
using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Per-stage parity debug for the firefly codec decode. Compares quantizer + generator sub-stages against
/// firefly_stages.safetensors (dump_firefly_stages.py). Gated on <c>FIREFLY_WEIGHTS</c> + <c>FIREFLY_STAGES</c>.</summary>
public sealed unsafe class FireflyStageDebugTests
{
    private readonly ITestOutputHelper _out;
    public FireflyStageDebugTests(ITestOutputHelper o) => _out = o;

    [Fact]
    [Trait("Category", "Integration")]
    public void FireflyDecode_StageByStage()
    {
        string? wPath = Environment.GetEnvironmentVariable("FIREFLY_WEIGHTS");
        string? stagesP = Environment.GetEnvironmentVariable("FIREFLY_STAGES");
        if (string.IsNullOrEmpty(wPath) || !File.Exists(wPath) || string.IsNullOrEmpty(stagesP) || !File.Exists(stagesP))
            return;

        SafeTensorsLoader wl = new(); wl.Load(wPath);
        IReadOnlyDictionary<string, Tensor> w = wl.GetAllTensors();
        SafeTensorsLoader sl = new(); sl.Load(stagesP);
        IReadOnlyDictionary<string, Tensor> stages = sl.GetAllTensors();

        Tensor idxRef = stages["fsq_out"];          // just for T
        int t = (int)stages["fsq_out"].Shape[2];
        // indices come from the stages ref's source — reuse firefly_ref_io via FIREFLY_REF.
        string refP = Environment.GetEnvironmentVariable("FIREFLY_REF")!;
        SafeTensorsLoader rl = new(); rl.Load(refP);
        Tensor indicesT = rl.GetAllTensors()["indices"];   // [1,8,T]
        int groups = (int)indicesT.Shape[1];
        int[,] codes = new int[groups, t];
        int* ip = (int*)indicesT.DataPointer;
        for (int g = 0; g < groups; g++)
            for (int j = 0; j < t; j++) codes[g, j] = ip[(long)g * t + j];

        List<string> lines = new();
        void Cmp(string key, Tensor mine)
        {
            if (!stages.ContainsKey(key)) { lines.Add($"{key}: no ref"); return; }
            Tensor refT = stages[key];
            (double corr, double mx) = Compare(mine, refT);
            string sh = mine.ElementCount == refT.ElementCount ? "" : $"  SHAPE MINE {mine.Shape} REF {refT.Shape}";
            lines.Add($"{key}: corr={corr:F6} maxAbs={mx:E3}{sh}");
        }

        using CpuBackend backend = new();
        FireflyQuantizer q = new(inputDim: 512);
        q.LoadWeights(w, "quantizer");
        q.DebugHook = (k, x) => Cmp(k, x);
        Tensor decIn = q.DequantToDecoderInput(backend, codes, t);

        FireflySiluGenerator gen = new(512, [8, 8, 2, 2, 2], [16, 16, 4, 4, 4], [3, 7, 11],
            [[1, 3, 5], [1, 3, 5], [1, 3, 5]]);
        gen.LoadWeights(w, "head");
        gen.DebugHook = (k, x) => Cmp(k, x);
        float[] audio = gen.Forward(backend, decIn, (int)decIn.Shape[2]);
        decIn.Dispose();

        Tensor audioRef = stages["audio_out"];
        Tensor audioMine = new(new TensorShape(1, 1, audio.Length), DType.F32);
        float* ap = (float*)audioMine.DataPointer;
        for (int i = 0; i < audio.Length; i++) ap[i] = audio[i];
        Cmp("audio_out", audioMine);
        audioMine.Dispose();

        string dest = Environment.GetEnvironmentVariable("FIREFLY_DEBUG_OUT") ?? "/tmp/firefly_stages.txt";
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
