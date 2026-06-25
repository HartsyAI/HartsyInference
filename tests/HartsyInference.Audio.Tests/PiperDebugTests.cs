using HartsyInference.Audio.Models.Vits;
using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelHandler.Onnx;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Per-stage debug dumps for diagnosing the Piper VITS parity gap. Gated on <c>PIPER_ONNX</c>.</summary>
public sealed unsafe class PiperDebugTests
{
    private readonly ITestOutputHelper _out;
    public PiperDebugTests(ITestOutputHelper o) => _out = o;

    [Fact]
    public void DumpStages()
    {
        string? onnx = Environment.GetEnvironmentVariable("PIPER_ONNX");
        if (string.IsNullOrEmpty(onnx) || !File.Exists(onnx)) return;

        using OnnxWeightLoader loader = new();
        loader.Load(onnx);
        Dictionary<string, Tensor> w = loader.GetResolvedTensors();
        VitsConfig cfg = VitsConfig.PiperMedium with { NoiseScaleW = 0f };

        string[] expKeys = w.Keys.Where(k => k.Contains("flows.0") && k.Contains("Exp")).ToArray();
        File.WriteAllText("/tmp/cs_expkeys.txt", string.Join("\n", expKeys.Select(k => $"{k} count={w[k].ElementCount}")));

        int[] ids = [1, 0, 50, 0, 23, 0, 86, 0, 35, 0, 16, 0, 74, 0, 3, 0, 120, 0, 55, 0, 2];
        using CpuBackend backend = new();

        VitsTextEncoder enc = new(cfg);
        enc.LoadWeights(w);
        (Tensor hidden, Tensor mP, Tensor logsP) = enc.Forward(backend, ids);
        _out.WriteLine($"hidden {Shape(hidden)} mP {Shape(mP)} logsP {Shape(logsP)}");

        // Dump m_p [1, inter, T] -> raw f32 (channels-major).
        float* mp = (float*)mP.DataPointer;
        byte[] mpBytes = new byte[mP.ElementCount * 4];
        fixed (byte* b = mpBytes) Buffer.MemoryCopy(mp, b, mpBytes.Length, mpBytes.Length);
        File.WriteAllBytes("/tmp/cs_mp.f32", mpBytes);
        _out.WriteLine($"mP mean={Mean(mp, mP.ElementCount):F4} std={Std(mp, mP.ElementCount):F4}");

        // Durations via SDP (noise_w = 0).
        VitsStochasticDurationPredictor sdp = new(cfg);
        sdp.LoadWeights(w, cfg.SdpPrefix);
        float[] logw = sdp.Forward(backend, hidden, ids.Length, noiseScaleW: 0f, seed: 0);
        int[] dur = logw.Select(l => Math.Max(1, (int)MathF.Ceiling(MathF.Exp(l)))).ToArray();
        _out.WriteLine($"C# durations: [{string.Join(",", dur)}] sum={dur.Sum()}");
        File.WriteAllText("/tmp/cs_dur.txt", string.Join(",", dur) + "\n" + string.Join(",", logw.Select(l => l.ToString("F4"))));
    }

    private static string Shape(Tensor t) => t.Shape.ToString();
    private static float Mean(float* p, long n) { double s = 0; for (long i = 0; i < n; i++) s += p[i]; return (float)(s / n); }
    private static float Std(float* p, long n) { double m = Mean(p, n), s = 0; for (long i = 0; i < n; i++) s += (p[i] - m) * (p[i] - m); return (float)Math.Sqrt(s / n); }
}
