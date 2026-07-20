using System;
using System.Collections.Generic;
using System.IO;
using HartsyInference.Audio.Models.Demucs;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Real-generation check: runs the production <see cref="DemucsPipeline.Separate"/> on a real audio clip
/// (jfk speech → 44.1k stereo) and compares the 4 stems to (a) the demucs raw model forward and (b) demucs
/// apply_model (the full segment/overlap pipeline). Gated on <c>DEMUCS_WEIGHTS</c> + <c>DEMUCS_REALGEN</c>
/// (dump_demucs_realgen.py). Writes the C# vocals stem to <c>DEMUCS_REALGEN_OUT</c>.</summary>
public sealed unsafe class DemucsRealGenTests
{
    private readonly ITestOutputHelper _out;
    public DemucsRealGenTests(ITestOutputHelper o) => _out = o;

    [Fact]
    [Trait("Category", "Integration")]
    public void DemucsRealGen_MatchesDemucs()
    {
        string? wPath = Environment.GetEnvironmentVariable("DEMUCS_WEIGHTS");
        string? refP = Environment.GetEnvironmentVariable("DEMUCS_REALGEN");
        if (string.IsNullOrEmpty(wPath) || !File.Exists(wPath) || string.IsNullOrEmpty(refP) || !File.Exists(refP)) return;

        SafeTensorsLoader wl = new(); wl.Load(wPath);
        IReadOnlyDictionary<string, Tensor> w = wl.GetAllTensors();
        SafeTensorsLoader rl = new(); rl.Load(refP);
        IReadOnlyDictionary<string, Tensor> r = rl.GetAllTensors();

        Tensor wav = r["wav"];               // [1,2,L]
        Tensor stemsModel = r["stems_model"]; // [4,2,L] raw forward
        Tensor stemsApply = r["stems_apply"]; // [4,2,L] production
        int length = (int)wav.Shape[2];
        float* wp = (float*)wav.DataPointer;
        float[] left = new float[length]; float[] right = new float[length];
        for (int i = 0; i < length; i++) { left[i] = wp[i]; right[i] = wp[(long)length + i]; }

        DemucsPipeline pipe = new(HtDemucsConfig.Htdemucs);
        pipe.LoadWeights(w);
        using CpuBackend backend = new();
        (float[] Left, float[] Right)[] stems = pipe.Separate(backend, left, right);
        Assert.Equal(4, stems.Length);

        // Repack C# stems → [4,2,L] and compare to both references.
        float* smp = (float*)stemsModel.DataPointer;
        float* sap = (float*)stemsApply.DataPointer;
        string[] names = { "drums", "bass", "other", "vocals" };
        for (int s = 0; s < 4; s++)
        {
            (double cM, double mM) = Corr(stems[s].Left, stems[s].Right, smp, s, length);
            (double cA, double mA) = Corr(stems[s].Left, stems[s].Right, sap, s, length);
            _out.WriteLine($"{names[s]}: vs raw-forward corr={cM:F6} maxAbs={mM:E3} | vs apply_model corr={cA:F6} maxAbs={mA:E3}");
        }

        // Write the C# vocals stem as an artifact.
        string? outWav = Environment.GetEnvironmentVariable("DEMUCS_REALGEN_OUT");
        if (!string.IsNullOrEmpty(outWav)) WriteWav(outWav, stems[3].Left, stems[3].Right, 44100);

        // The pipeline now implements apply_model (segment + overlap-add) → must match demucs apply_model.
        for (int s = 0; s < 4; s++)
        {
            (double c, _) = Corr(stems[s].Left, stems[s].Right, sap, s, length);
            Assert.True(c > 0.99, $"C# Separate {names[s]} vs demucs apply_model corr too low ({c:F6}).");
        }
    }

    private static (double corr, double mx) Corr(float[] l, float[] rr, float* refBase, int s, int length)
    {
        // ref layout [4,2,L]: source s, channel 0 = L, channel 1 = R.
        double sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0, mx = 0; long n = 0;
        for (int ch = 0; ch < 2; ch++)
        {
            float[] mine = ch == 0 ? l : rr;
            long off = (((long)s * 2) + ch) * length;
            for (int i = 0; i < length; i++)
            {
                double x = mine[i], y = refBase[off + i];
                sa += x; sb += y; saa += x * x; sbb += y * y; sab += x * y;
                mx = Math.Max(mx, Math.Abs(x - y)); n++;
            }
        }
        double cov = sab - sa * sb / n, va = saa - sa * sa / n, vb = sbb - sb * sb / n;
        return (cov / (Math.Sqrt(va * vb) + 1e-12), mx);
    }

    private static void WriteWav(string path, float[] l, float[] r, int sr)
    {
        int n = l.Length;
        using BinaryWriter bw = new(File.Create(path));
        int dataBytes = n * 2 * 2;
        bw.Write("RIFF"u8); bw.Write(36 + dataBytes); bw.Write("WAVE"u8);
        bw.Write("fmt "u8); bw.Write(16); bw.Write((short)1); bw.Write((short)2);
        bw.Write(sr); bw.Write(sr * 2 * 2); bw.Write((short)4); bw.Write((short)16);
        bw.Write("data"u8); bw.Write(dataBytes);
        for (int i = 0; i < n; i++)
        {
            bw.Write((short)Math.Clamp((int)(l[i] * 32767f), -32768, 32767));
            bw.Write((short)Math.Clamp((int)(r[i] * 32767f), -32768, 32767));
        }
    }
}
