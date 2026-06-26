using HartsyInference.Audio.Io;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Cpu;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>End-to-end Piper: text → espeak IPA → token ids → VITS → waveform, written to a WAV for manual QA. Gated on
/// <c>PIPER_ONNX</c> (path to the voice <c>.onnx</c>, with its <c>.onnx.json</c> beside it) and <c>ESPEAK_DATA_DIR</c>.</summary>
public sealed class PiperEndToEndTests
{
    private readonly ITestOutputHelper _out;
    public PiperEndToEndTests(ITestOutputHelper o) => _out = o;

    [Fact]
    public void SynthesizesText()
    {
        string? onnx = Environment.GetEnvironmentVariable("PIPER_ONNX");
        if (string.IsNullOrEmpty(onnx) || !File.Exists(onnx)) return;
        string json = onnx + ".json";
        if (!File.Exists(json))
        {
            // scratchpad copy names the config "cfg.json"; allow a sibling override.
            string alt = Path.Combine(Path.GetDirectoryName(onnx)!, "cfg.json");
            if (File.Exists(alt)) json = alt; else return;
        }
        if (string.IsNullOrEmpty(Environment.GetEnvironmentVariable("ESPEAK_DATA_DIR"))) return;

        using PiperPipeline piper = PiperPipeline.LoadFromFiles(onnx, json);
        using CpuBackend backend = new();

        const string text = "Hello world. This is a test of the speech synthesizer.";
        float[] audio = piper.SynthesizeText(backend, text, seed: 1234);

        _out.WriteLine($"{audio.Length} samples ({audio.Length / (double)piper.SampleRate:F2}s) @ {piper.SampleRate}Hz");
        Assert.NotEmpty(audio);

        double sumSq = 0;
        float maxAbs = 0;
        foreach (float a in audio)
        {
            Assert.True(float.IsFinite(a));
            sumSq += (double)a * a;
            maxAbs = MathF.Max(maxAbs, MathF.Abs(a));
        }
        double rms = Math.Sqrt(sumSq / audio.Length);
        _out.WriteLine($"rms={rms:F4} maxAbs={maxAbs:F4}");
        Assert.True(rms > 0.01, $"audio is near-silent (rms {rms:F4})");
        Assert.True(maxAbs <= 1.5f, $"audio out of range (maxAbs {maxAbs:F3})");

        string outDir = Path.Combine(Path.GetTempPath(), "hartsyinference_piper");
        Directory.CreateDirectory(outDir);
        string outPath = Path.Combine(outDir, "piper_e2e.wav");
        WavFile.WriteMono16(outPath, audio, piper.SampleRate);
        _out.WriteLine($"wrote {outPath}");
    }
}
