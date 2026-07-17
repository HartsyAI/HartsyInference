using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using HartsyInference.Audio.Models.Zonos;
using HartsyInference.Core.Backends;
using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelHandler.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Real-weight parity for the Zonos prefix conditioner (both CFG branches) vs
/// <c>PrefixConditioner.forward</c>. Gated on the transformer checkpoint (<c>ZONOS_MODEL</c> =
/// model.safetensors) and the golden dump dir (<c>ZONOS_GOLDEN</c>: cond_prefix / uncond_prefix / spk_lda128 /
/// phonemes.json). Uses the golden phoneme ids + golden speaker embedding to isolate the conditioner from espeak
/// and the speaker encoder.</summary>
public sealed unsafe class ZonosConditioningParityTests
{
    private const int EnUsLanguageId = 24;
    private readonly ITestOutputHelper _out;
    public ZonosConditioningParityTests(ITestOutputHelper o) => _out = o;

    [Fact]
    [Trait("Category", "Integration")]
    public void PrefixConditioner_MatchesReference()
    {
        string? modelPath = Environment.GetEnvironmentVariable("ZONOS_MODEL");
        string? golden = Environment.GetEnvironmentVariable("ZONOS_GOLDEN");
        if (string.IsNullOrEmpty(modelPath) || !File.Exists(modelPath) || string.IsNullOrEmpty(golden) || !Directory.Exists(golden))
        {
            _out.WriteLine("Skipped: set ZONOS_MODEL / ZONOS_GOLDEN.");
            return;
        }

        SafeTensorsLoader wl = new();
        wl.Load(modelPath);
        IReadOnlyDictionary<string, Tensor> w = wl.GetAllTensors();

        int[] phonemes = LoadIds(golden);
        float[] goldCond = LoadBin(golden, "cond_prefix", out int[] shape);   // [1, P, 2048]
        float[] goldUncond = LoadBin(golden, "uncond_prefix", out _);
        float[] spk = LoadBin(golden, "spk_lda128", out _);
        int p = shape[1], d = shape[2];

        IBackend backend = Environment.GetEnvironmentVariable("ZONOS_CUDA") == "1"
            ? new HartsyInference.Cuda.CudaBackend(0, Environment.GetEnvironmentVariable("ZONOS_PTX")!)
            : new CpuBackend();
        _out.WriteLine($"Backend: {backend.GetType().Name}");
        using ZonosConditioning cond = new();
        cond.LoadWeights(w);

        Tensor speaker = new(new TensorShape(1, ZonosConditioning.SpeakerDim), DType.F32);
        fixed (float* sptr = spk)
            Buffer.MemoryCopy(sptr, (void*)speaker.DataPointer, spk.Length * 4L, spk.Length * 4L);

        // make_cond_dict normalizes emotion by its sum.
        float[] emotion = [0.3077f, 0.0256f, 0.0256f, 0.0256f, 0.0256f, 0.0256f, 0.2564f, 0.3077f];
        float sum = 0; foreach (float e in emotion) sum += e;
        for (int i = 0; i < emotion.Length; i++) emotion[i] /= sum;

        Tensor condPrefix = cond.BuildPrefix(backend, phonemes, speaker, emotion,
            fmax: 22050f, pitchStd: 20f, speakingRate: 15f, languageId: EnUsLanguageId, conditional: true);
        Tensor uncondPrefix = cond.BuildPrefix(backend, phonemes, speaker, emotion,
            fmax: 22050f, pitchStd: 20f, speakingRate: 15f, languageId: EnUsLanguageId, conditional: false);

        (double cc, double cm) = CompareTransposed(condPrefix, goldCond, p, d);
        (double uc, double um) = CompareTransposed(uncondPrefix, goldUncond, p, d);
        _out.WriteLine($"cond  : corr={cc:F6} maxAbs={cm:E3}");
        _out.WriteLine($"uncond: corr={uc:F6} maxAbs={um:E3}");
        speaker.Dispose(); condPrefix.Dispose(); uncondPrefix.Dispose(); (backend as IDisposable)?.Dispose();

        Assert.True(cc > 0.9995, $"cond prefix corr {cc}");
        Assert.True(uc > 0.9995, $"uncond prefix corr {uc}");
    }

    private static int[] LoadIds(string dir)
    {
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(dir, "phonemes.json")));
        JsonElement arr = doc.RootElement.GetProperty("ids");
        int[] ids = new int[arr.GetArrayLength()];
        for (int i = 0; i < ids.Length; i++) ids[i] = arr[i].GetInt32();
        return ids;
    }

    /// <summary>Compares engine channels-first <c>[1, D, P]</c> against golden channels-last <c>[1, P, D]</c>.</summary>
    private static (double corr, double maxAbs) CompareTransposed(Tensor cf, float[] golden, int p, int d)
    {
        float* q = (float*)cf.DataPointer;   // [D, P]
        double sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0, maxAbs = 0;
        long n = (long)p * d;
        for (int s = 0; s < p; s++)
            for (int c = 0; c < d; c++)
            {
                double x = q[(long)c * p + s];
                double y = golden[(long)s * d + c];
                sa += x; sb += y; saa += x * x; sbb += y * y; sab += x * y;
                double diff = Math.Abs(x - y);
                if (diff > maxAbs) maxAbs = diff;
            }
        double cov = sab / n - (sa / n) * (sb / n);
        double va = saa / n - (sa / n) * (sa / n);
        double vb = sbb / n - (sb / n) * (sb / n);
        return (cov / (Math.Sqrt(va * vb) + 1e-12), maxAbs);
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
}
