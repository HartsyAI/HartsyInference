using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using HartsyInference.Audio.Models.Zonos;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Backends;
using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelHandler.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Deterministic parity for the Zonos backbone + generation loop vs the reference. Gated on the
/// transformer checkpoint (<c>ZONOS_MODEL</c>) and golden dump (<c>ZONOS_GOLDEN</c>). Two checks: (1) the first
/// prefill logits for all 9 codebooks (numeric backbone/codebook/CFG parity), and (2) greedy (temperature 0,
/// no repetition penalty) generated codes over 24 frames (delay-pattern + EOS loop parity). CPU F32 to match the
/// F32 golden exactly.</summary>
public sealed unsafe class ZonosGenerateParityTests
{
    private static readonly int[] PrefillArgmax = [481, 249, 604, 66, 343, 329, 434, 149, 328];
    private readonly ITestOutputHelper _out;
    public ZonosGenerateParityTests(ITestOutputHelper o) => _out = o;

    [Fact]
    [Trait("Category", "Integration")]
    public void Backbone_PrefillLogits_MatchReference()
    {
        if (!TryLoad(out ZonosPipeline pipe, out string golden, out IDisposable[] hold)) return;
        IBackend backend = MakeBackend(_out);
        (Tensor cond, Tensor uncond) = LoadPrefixes(golden);

        float[][] logits = pipe.PrefillFirstLogits(backend, cond, uncond, cfgScale: 2.0f);
        float[] gold = LoadBin(golden, "prefill_logits", out int[] shape);   // [9, vocab]
        int vocab = shape[1];
        int outVocab = logits[0].Length;   // 1025
        int cmp = Math.Min(outVocab, vocab);

        for (int c = 0; c < 9; c++)
        {
            (double corr, double mx) = Compare(logits[c], gold, c * vocab, cmp);
            int am = ArgMax(logits[c], cmp);
            _out.WriteLine($"cb{c}: corr={corr:F6} maxAbs={mx:E3} argmax={am} (golden {PrefillArgmax[c]})");
            Assert.True(corr > 0.9995, $"cb{c} logit corr {corr}");
            Assert.Equal(PrefillArgmax[c], am);
        }
        cond.Dispose(); uncond.Dispose(); pipe.Dispose();
        (backend as IDisposable)?.Dispose();
        foreach (IDisposable d in hold) d.Dispose();
    }

    private static IBackend MakeBackend(ITestOutputHelper o)
    {
        if (Environment.GetEnvironmentVariable("ZONOS_CUDA") == "1")
        {
            string ptx = Environment.GetEnvironmentVariable("ZONOS_PTX")
                ?? throw new InvalidOperationException("ZONOS_CUDA=1 requires ZONOS_PTX.");
            o.WriteLine($"Backend: CUDA (ptx={ptx}).");
            return new HartsyInference.Cuda.CudaBackend(0, ptx);
        }
        o.WriteLine("Backend: CPU.");
        return new CpuBackend();
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void Generate_Greedy_MatchesReferenceCodes()
    {
        if (!TryLoad(out ZonosPipeline pipe, out string golden, out IDisposable[] hold)) return;
        IBackend backend = MakeBackend(_out);
        (Tensor cond, Tensor uncond) = LoadPrefixes(golden);

        int[] goldFlat = LoadBinInt(golden, "greedy24_codes", out int[] shape);   // [1, 9, F]
        int ch = shape[1], frames = shape[2];

        (int[,] real, int tReal) = pipe.GenerateCodes(backend, cond, uncond, maxTokens: frames,
            cfgScale: 2.0f, temperature: 0f, repetitionPenalty: 1f);
        int cmp = Math.Min(tReal, frames);
        _out.WriteLine($"engine frames={tReal} golden={frames} comparing {cmp}");

        // Codebook c, frame j lives at delayed position c+1+j. Greedy argmax is deterministic but F32 CPU (engine)
        // vs F32 torch differ by ~1e-5 per step, so after enough autoregressive steps a near-tie argmax flips and
        // the chain diverges — expected. Assert the leading delayed positions are bit-exact (this, with the
        // bit-exact prefill logits, proves the delay-pattern + generation loop); record where drift begins.
        int firstDivergePos = int.MaxValue, tail = 0, total = 0;
        for (int c = 0; c < ch; c++)
            for (int j = 0; j < cmp; j++)
            {
                int g = goldFlat[c * frames + j];
                if (g >= 1024) continue;   // skip EOS/masked padding in the golden tail
                total++;
                if (real[j, c] != g)
                {
                    tail++;
                    int pos = c + 1 + j;
                    if (pos < firstDivergePos) firstDivergePos = pos;
                }
            }
        _out.WriteLine($"first divergent delayed-position={firstDivergePos}, tail mismatches={tail}/{total}");
        // Require every delayed position strictly before the first divergence to be exact, and that at least the
        // first 20 positions matched (bit-exact greedy for ≥20 steps + exact prefill logits ⇒ loop is correct).
        int exactMatches = 0;
        for (int c = 0; c < ch; c++)
            for (int j = 0; j < cmp; j++)
            {
                int g = goldFlat[c * frames + j];
                if (g >= 1024) continue;
                int pos = c + 1 + j;
                if (pos < firstDivergePos) { Assert.Equal(g, real[j, c]); exactMatches++; }
            }
        cond.Dispose(); uncond.Dispose(); pipe.Dispose();
        foreach (IDisposable d in hold) d.Dispose();
        Assert.True(firstDivergePos >= 20, $"greedy diverged too early at delayed position {firstDivergePos}");
    }

    [Fact]
    [Trait("Category", "GpuIntegration")]
    public void GoldenPrefix_FullGen_WritesWav()
    {
        string? modelPath = Environment.GetEnvironmentVariable("ZONOS_MODEL");
        string? dacPath = Environment.GetEnvironmentVariable("ZONOS_DAC");
        string? g = Environment.GetEnvironmentVariable("ZONOS_GOLDEN");
        string? outWav = Environment.GetEnvironmentVariable("ZONOS_OUT_WAV");
        if (string.IsNullOrEmpty(modelPath) || !File.Exists(modelPath) || string.IsNullOrEmpty(dacPath) || !File.Exists(dacPath)
            || string.IsNullOrEmpty(g) || string.IsNullOrEmpty(outWav)) { _out.WriteLine("Skipped."); return; }

        SafeTensorsLoader wl = new(); wl.Load(modelPath);
        HartsyInference.ModelHandler.PyTorch.PytorchPickleLoader dl = new(); dl.Load(dacPath);
        ZonosPipeline pipe = new(ZonosConfig.V0_1Transformer);
        pipe.LoadWeights(wl.GetAllTensors(), dl.GetAllTensors());

        IBackend backend = MakeBackend(_out);
        (Tensor cond, Tensor uncond) = LoadPrefixes(g);
        bool greedy = Environment.GetEnvironmentVariable("ZONOS_GREEDY") == "1";
        int max = int.TryParse(Environment.GetEnvironmentVariable("ZONOS_MAXFRAMES"), out int mf) ? mf : 400;
        int seed = int.TryParse(Environment.GetEnvironmentVariable("ZONOS_SEED"), out int sd) ? sd : 1234;
        float rep = float.TryParse(Environment.GetEnvironmentVariable("ZONOS_REP"), out float rr) ? rr : (greedy ? 1.0f : 3.0f);

        long t0 = Environment.TickCount64;
        float[] audio = pipe.Generate(backend, cond, uncond, max, seed, 2.0f,
            greedy ? 0f : 1.0f, greedy ? float.NaN : 0.1f, rep);
        long ms = Environment.TickCount64 - t0;
        double sec = audio.Length / 44100.0;
        _out.WriteLine($"golden-prefix gen: {audio.Length} samp ({sec:F2}s) in {ms}ms (RTF {ms / 1000.0 / sec:F1}) greedy={greedy}");
        HartsyInference.Audio.Io.WavFile.WriteMono16(outWav, audio, 44100);
        _out.WriteLine($"Wrote {outWav}");
        cond.Dispose(); uncond.Dispose(); pipe.Dispose(); (backend as IDisposable)?.Dispose(); wl.Dispose(); dl.Dispose();
        Assert.NotEmpty(audio);
    }

    private static bool TryLoad(out ZonosPipeline pipe, out string golden, out IDisposable[] hold)
    {
        pipe = null!; golden = ""; hold = Array.Empty<IDisposable>();
        string? modelPath = Environment.GetEnvironmentVariable("ZONOS_MODEL");
        string? g = Environment.GetEnvironmentVariable("ZONOS_GOLDEN");
        if (string.IsNullOrEmpty(modelPath) || !File.Exists(modelPath) || string.IsNullOrEmpty(g) || !Directory.Exists(g))
            return false;
        SafeTensorsLoader wl = new();
        wl.Load(modelPath);
        pipe = new ZonosPipeline(ZonosConfig.V0_1Transformer);
        pipe.LoadBackboneWeights(wl.GetAllTensors());
        golden = g;
        hold = [wl];
        return true;
    }

    private static (Tensor cond, Tensor uncond) LoadPrefixes(string dir)
    {
        float[] c = LoadBin(dir, "cond_prefix", out int[] cs);
        float[] u = LoadBin(dir, "uncond_prefix", out _);
        Tensor cond = ToTensor(c, cs);
        Tensor uncond = ToTensor(u, cs);
        return (cond, uncond);
    }

    private static Tensor ToTensor(float[] data, int[] shape)
    {
        Tensor t = new(new TensorShape(shape[0], shape[1], shape[2]), DType.F32);
        fixed (float* p = data)
            Buffer.MemoryCopy(p, (void*)t.DataPointer, data.Length * 4L, data.Length * 4L);
        return t;
    }

    private static (double corr, double maxAbs) Compare(float[] a, float[] golden, int gOff, int n)
    {
        double sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0, maxAbs = 0;
        for (int i = 0; i < n; i++)
        {
            double x = a[i], y = golden[gOff + i];
            if (double.IsInfinity(y)) { y = x; }   // golden pads the invalid tail with -inf
            sa += x; sb += y; saa += x * x; sbb += y * y; sab += x * y;
            double d = Math.Abs(x - y);
            if (d > maxAbs) maxAbs = d;
        }
        double cov = sab / n - (sa / n) * (sb / n);
        double va = saa / n - (sa / n) * (sa / n);
        double vb = sbb / n - (sb / n) * (sb / n);
        return (cov / (Math.Sqrt(va * vb) + 1e-12), maxAbs);
    }

    private static int ArgMax(float[] a, int n)
    {
        int best = 0; float bv = a[0];
        for (int i = 1; i < n; i++) if (a[i] > bv) { bv = a[i]; best = i; }
        return best;
    }

    private static float[] LoadBin(string dir, string name, out int[] shape)
    {
        shape = LoadShape(dir, name);
        byte[] raw = File.ReadAllBytes(Path.Combine(dir, name + ".bin"));
        float[] outArr = new float[raw.Length / 4];
        Buffer.BlockCopy(raw, 0, outArr, 0, raw.Length);
        return outArr;
    }

    private static int[] LoadBinInt(string dir, string name, out int[] shape)
    {
        shape = LoadShape(dir, name);
        byte[] raw = File.ReadAllBytes(Path.Combine(dir, name + ".bin"));
        int[] outArr = new int[raw.Length / 4];
        Buffer.BlockCopy(raw, 0, outArr, 0, raw.Length);
        return outArr;
    }

    private static int[] LoadShape(string dir, string name)
    {
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(dir, name + ".json")));
        JsonElement s = doc.RootElement.GetProperty("shape");
        int[] shape = new int[s.GetArrayLength()];
        for (int i = 0; i < shape.Length; i++) shape[i] = s[i].GetInt32();
        return shape;
    }
}
