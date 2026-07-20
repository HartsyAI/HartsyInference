using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using HartsyInference.Audio.Models.Zonos;
using HartsyInference.Core.Backends;
using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.PyTorch;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Real-weight parity for the Zonos ResNet293 speaker encoder vs the reference
/// <c>SpeakerEmbeddingLDA</c> (zonos/speaker_cloning.py). Gated on the two <c>.pt</c> checkpoints
/// (<c>ZONOS_SPK_WEIGHTS</c> = ResNet293_SimAM_ASP_base.pt, <c>ZONOS_SPK_LDA</c> = ..._LDA-128.pt) and the
/// golden dump dir <c>ZONOS_GOLDEN</c> (zonos_golden.py: spk_wav16k / spk_mel / spk_lda128 raw f32 bins).</summary>
public sealed unsafe class ZonosSpeakerParityTests
{
    private readonly ITestOutputHelper _out;
    public ZonosSpeakerParityTests(ITestOutputHelper o) => _out = o;

    [Fact]
    [Trait("Category", "Integration")]
    public void SpeakerEncoder_MatchesReference()
    {
        string? wPath = Environment.GetEnvironmentVariable("ZONOS_SPK_WEIGHTS");
        string? ldaPath = Environment.GetEnvironmentVariable("ZONOS_SPK_LDA");
        string? golden = Environment.GetEnvironmentVariable("ZONOS_GOLDEN");
        if (string.IsNullOrEmpty(wPath) || !File.Exists(wPath) || string.IsNullOrEmpty(ldaPath) || !File.Exists(ldaPath)
            || string.IsNullOrEmpty(golden) || !Directory.Exists(golden))
        {
            _out.WriteLine("Skipped: set ZONOS_SPK_WEIGHTS / ZONOS_SPK_LDA / ZONOS_GOLDEN.");
            return;
        }

        using PytorchPickleLoader wl = new();
        wl.Load(wPath);
        Dictionary<string, Tensor> w = wl.GetAllTensors();
        using PytorchPickleLoader ll = new();
        ll.Load(ldaPath);
        Dictionary<string, Tensor> lda = ll.GetAllTensors();

        float[] wav = LoadBin(golden, "spk_wav16k", out _);
        float[] goldMel = LoadBin(golden, "spk_mel", out int[] melShape);
        float[] goldLda = LoadBin(golden, "spk_lda128", out _);

        ZonosSpeakerEncoder enc = new(ZonosSpeakerConfig.Default);
        enc.LoadWeights(w, lda);

        IBackend backend;
        if (Environment.GetEnvironmentVariable("ZONOS_CUDA") == "1")
        {
            string ptx = Environment.GetEnvironmentVariable("ZONOS_PTX")!;
            backend = new HartsyInference.Cuda.CudaBackend(0, ptx);
            _out.WriteLine("Backend: CUDA.");
        }
        else { backend = new CpuBackend(); _out.WriteLine("Backend: CPU."); }

        // 1) logFbank front-end parity.
        (Tensor mel, int t) = enc.BuildMel(wav);
        _out.WriteLine($"mel frames: engine={t} golden={melShape[2]}");
        (double melCorr, double melMax) = Compare(mel, goldMel);
        _out.WriteLine($"mel: corr={melCorr:F6} maxAbs={melMax:E3}");

        // 2) full 128-d embedding parity.
        Tensor emb = enc.Embed(backend, mel, t);
        (double embCorr, double embMax) = Compare(emb, goldLda);
        _out.WriteLine($"lda128: corr={embCorr:F6} maxAbs={embMax:E3}");
        mel.Dispose();
        emb.Dispose();

        Assert.True(melCorr > 0.999, $"mel corr {melCorr}");
        Assert.True(embCorr > 0.99, $"speaker embedding corr {embCorr}");
    }

    private static float[] LoadBin(string dir, string name, out int[] shape)
    {
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(dir, name + ".json")));
        JsonElement s = doc.RootElement.GetProperty("shape");
        shape = new int[s.GetArrayLength()];
        for (int i = 0; i < shape.Length; i++) shape[i] = s[i].GetInt32();
        byte[] raw = File.ReadAllBytes(Path.Combine(dir, name + ".bin"));
        float[] outArr = new float[raw.Length / 4];
        Buffer.BlockCopy(raw, 0, outArr, 0, raw.Length);
        return outArr;
    }

    private static (double corr, double maxAbs) Compare(Tensor a, float[] golden)
    {
        long n = a.ElementCount;
        float* p = (float*)a.DataPointer;
        double sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0, maxAbs = 0;
        for (long i = 0; i < n; i++)
        {
            double x = p[i], y = golden[i];
            sa += x; sb += y; saa += x * x; sbb += y * y; sab += x * y;
            double d = Math.Abs(x - y);
            if (d > maxAbs) maxAbs = d;
        }
        double cov = sab / n - (sa / n) * (sb / n);
        double va = saa / n - (sa / n) * (sa / n);
        double vb = sbb / n - (sb / n) * (sb / n);
        double corr = cov / (Math.Sqrt(va * vb) + 1e-12);
        return (corr, maxAbs);
    }
}
