using System;
using System.IO;
using System.Text.Json;
using HartsyInference.Audio.Io;
using HartsyInference.Audio.Models.Zonos;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Backends;
using HartsyInference.Cpu;
using HartsyInference.ModelHandler.PyTorch;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.Phonemizer.Espeak;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Full end-to-end <see cref="ZonosTts"/> generation (speaker encode → phonemes → cond/uncond prefix →
/// delayed-AR → DAC decode) writing a 44.1 kHz WAV for external whisper verification. Gated on the checkpoints
/// (<c>ZONOS_MODEL</c>, <c>ZONOS_DAC</c>, <c>ZONOS_SPK_WEIGHTS</c>, <c>ZONOS_SPK_LDA</c>), the reference clip
/// (<c>ZONOS_GOLDEN/spk_wav16k</c>), espeak data (<c>ESPEAK_DATA_DIR</c>) and an output path (<c>ZONOS_OUT_WAV</c>).
/// Runs on CUDA when <c>ZONOS_CUDA=1</c> + <c>ZONOS_PTX</c> (full 30 s gen is far too slow on CPU).</summary>
public sealed unsafe class ZonosE2eTests
{
    private readonly ITestOutputHelper _out;
    public ZonosE2eTests(ITestOutputHelper o) => _out = o;

    [Fact]
    [Trait("Category", "GpuIntegration")]
    public void Synthesize_WritesWav()
    {
        string? model = Environment.GetEnvironmentVariable("ZONOS_MODEL");
        string? dac = Environment.GetEnvironmentVariable("ZONOS_DAC");
        string? spk = Environment.GetEnvironmentVariable("ZONOS_SPK_WEIGHTS");
        string? lda = Environment.GetEnvironmentVariable("ZONOS_SPK_LDA");
        string? golden = Environment.GetEnvironmentVariable("ZONOS_GOLDEN");
        string? espeak = Environment.GetEnvironmentVariable("ESPEAK_DATA_DIR");
        string? outWav = Environment.GetEnvironmentVariable("ZONOS_OUT_WAV");
        if (AnyMissing(model, dac, spk, lda) || string.IsNullOrEmpty(golden) || !Directory.Exists(golden)
            || string.IsNullOrEmpty(espeak) || !Directory.Exists(espeak) || string.IsNullOrEmpty(outWav))
        {
            _out.WriteLine("Skipped: set ZONOS_MODEL/DAC/SPK_WEIGHTS/SPK_LDA + ZONOS_GOLDEN + ESPEAK_DATA_DIR + ZONOS_OUT_WAV.");
            return;
        }

        SafeTensorsLoader modelLoader = new(); modelLoader.Load(model!);
        PytorchPickleLoader dacLoader = new(); dacLoader.Load(dac!);
        PytorchPickleLoader spkLoader = new(); spkLoader.Load(spk!);
        PytorchPickleLoader ldaLoader = new(); ldaLoader.Load(lda!);

        IBackend backend;
        if (Environment.GetEnvironmentVariable("ZONOS_CUDA") == "1")
        {
            string ptx = Environment.GetEnvironmentVariable("ZONOS_PTX")
                ?? throw new InvalidOperationException("ZONOS_CUDA=1 requires ZONOS_PTX.");
            backend = new HartsyInference.Cuda.CudaBackend(0, ptx);
            _out.WriteLine($"Backend: CUDA (ptx={ptx}).");
        }
        else { backend = new CpuBackend(); _out.WriteLine("Backend: CPU."); }

        EspeakPhonemizer phonemizer = EspeakPhonemizer.FromDataDirectory(espeak!, "en-us");
        using ZonosTts tts = new(ZonosConfig.V0_1Transformer, phonemizer, "en-us");
        tts.LoadWeights(modelLoader.GetAllTensors(), dacLoader.GetAllTensors(),
            spkLoader.GetAllTensors(), ldaLoader.GetAllTensors());

        string? refWavPath = Environment.GetEnvironmentVariable("ZONOS_REF_WAV");
        float[] refWav16k;
        if (!string.IsNullOrEmpty(refWavPath) && File.Exists(refWavPath))
        {
            WavFile.DecodedAudio da = WavFile.Read(refWavPath);
            float[] mono = da.ToMono();
            refWav16k = da.SampleRate == 16000 ? mono
                : Resampler.Create(da.SampleRate, 16000).Resample(mono);
            _out.WriteLine($"Reference: {refWavPath} ({da.SampleRate} Hz → 16k, {refWav16k.Length} samp).");
        }
        else { refWav16k = LoadBin(golden!, "spk_wav16k"); }
        string text = Environment.GetEnvironmentVariable("ZONOS_TEXT")
            ?? "Hello, this is a test of the Zonos text to speech system.";
        bool greedy = Environment.GetEnvironmentVariable("ZONOS_GREEDY") == "1";
        int seed = int.TryParse(Environment.GetEnvironmentVariable("ZONOS_SEED"), out int sd) ? sd : 12345;
        ZonosControls controls = new()
        {
            CfgScale = 2.0f,
            MaxNewTokens = 600,
            Temperature = greedy ? 0f : 1.0f,
            RepetitionPenalty = greedy ? 1.0f : 3.0f,
        };
        if (greedy) _out.WriteLine("Greedy (temp=0, rep=1).");

        long t0 = Environment.TickCount64;
        float[] audio;
        if (Environment.GetEnvironmentVariable("ZONOS_GOLDEN_IDS") == "1")
        {
            int[] ids = LoadGoldenIds(golden!);
            _out.WriteLine($"Using golden phoneme ids ({ids.Length}).");
            audio = tts.SynthesizeFromPhonemes(backend, ids, refWav16k, controls, seed);
        }
        else { audio = tts.Synthesize(backend, text, refWav16k, controls, seed: seed); }
        long ms = Environment.TickCount64 - t0;

        Assert.NotEmpty(audio);
        double seconds = audio.Length / (double)tts.SampleRate;
        _out.WriteLine($"Generated {audio.Length} samples ({seconds:F2}s) in {ms} ms (RTF {ms / 1000.0 / seconds:F2}).");
        WavFile.WriteMono16(outWav!, audio, tts.SampleRate);
        _out.WriteLine($"Wrote {outWav}");
        (backend as IDisposable)?.Dispose();
    }

    private static bool AnyMissing(params string?[] paths)
    {
        foreach (string? p in paths) if (string.IsNullOrEmpty(p) || !File.Exists(p)) return true;
        return false;
    }

    private static float[] LoadBin(string dir, string name)
    {
        byte[] raw = File.ReadAllBytes(Path.Combine(dir, name + ".bin"));
        float[] outArr = new float[raw.Length / 4];
        Buffer.BlockCopy(raw, 0, outArr, 0, raw.Length);
        return outArr;
    }

    private static int[] LoadGoldenIds(string dir)
    {
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(dir, "phonemes.json")));
        JsonElement arr = doc.RootElement.GetProperty("ids");
        int[] ids = new int[arr.GetArrayLength()];
        for (int i = 0; i < ids.Length; i++) ids[i] = arr[i].GetInt32();
        return ids;
    }
}
