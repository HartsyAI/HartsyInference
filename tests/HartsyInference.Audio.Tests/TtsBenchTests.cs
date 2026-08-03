using System.Diagnostics;
using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Models.MeloTts;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Backends;
using HartsyInference.Cpu;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Pure generation-time benchmarks (exclude model load; warmup + 3 timed runs) for the SOTA TTS
/// models we support, to compare against Python reference timings. CPU by default (small models);
/// gated on cached weights.</summary>
public sealed class TtsBenchTests
{
    private readonly ITestOutputHelper _out;
    public TtsBenchTests(ITestOutputHelper o) => _out = o;

    private static IBackend MakeBackend(out string name)
    {
        if (Environment.GetEnvironmentVariable("BENCH_CUDA") == "1")
        {
            string ptx = Environment.GetEnvironmentVariable("BENCH_PTX") ?? throw new InvalidOperationException("BENCH_CUDA=1 needs BENCH_PTX");
            name = "CUDA"; return new HartsyInference.Cuda.CudaBackend(0, ptx);
        }
        name = "CPU"; return new CpuBackend();
    }

    private static (double gen, double rtf) Time(Func<float[]> run, int sr, int runs = 3)
    {
        run(); // warmup
        double best = double.MaxValue; int n = 0;
        for (int i = 0; i < runs; i++)
        {
            Stopwatch sw = Stopwatch.StartNew();
            float[] a = run();
            sw.Stop();
            best = Math.Min(best, sw.Elapsed.TotalSeconds);
            n = a.Length;
        }
        double dur = n / (double)sr;
        return (best, best / dur);
    }

    [Fact]
    public async Task Bench_Kokoro()
    {
        string w = AudioModelCache.GetRepoDirectory("Hartsy/kokoro-82m-safetensors", "tts");
        string v = AudioModelCache.GetRepoDirectory("hexgrad/Kokoro-82M", "tts");
        if (!File.Exists(Path.Combine(w, "kokoro-82m.safetensors")) || !File.Exists(Path.Combine(v, "voices", "af_heart.bin")))
        { _out.WriteLine("Kokoro not cached — skip."); return; }
        using KokoroPipeline k = await KokoroPipeline.LoadAsync();
        using IBackend b = MakeBackend(out string bn);
        const string ph = "hɛloʊ wɜːld. ðɪs ɪz ə tɛst.";
        (double gen, double rtf) = Time(() => k.Synthesize(b, ph, "af_heart", 1f), k.Config.SampleRate);
        _out.WriteLine($"OURS Kokoro ({bn}): gen {gen:F3}s | RTF {rtf:F3}");
    }

    [Fact]
    public async Task Bench_F5()
    {
        string dit = Path.Combine(AudioModelCache.GetRepoDirectory("SWivid/F5-TTS", "tts"), "F5TTS_v1_Base", "model_1250000.safetensors");
        string vocos = Path.Combine(AudioModelCache.GetRepoDirectory("lucasnewman/vocos-mel-24khz", "tts"), "model.safetensors");
        string jfk = Path.Combine(AudioModelCache.CacheRoot, "test-clips", "jfk.wav");
        if (!File.Exists(dit) || !File.Exists(vocos) || !File.Exists(jfk)) { _out.WriteLine("F5/vocos/jfk missing — skip."); return; }
        using F5TtsPipeline pipe = await F5TtsPipeline.LoadAsync();
        using IBackend b = MakeBackend(out string bn);
        HartsyInference.Audio.Io.WavFile.DecodedAudio refA = HartsyInference.Audio.Io.WavFile.Read(jfk);
        // ~3s reference trim.
        int rn = Math.Min(3 * refA.SampleRate, refA.Channels[0].Length);
        float[] refClip = refA.Channels[0][..rn];
        const string refText = "And so, my fellow Americans, ask not what your country can do for you";
        const string tgt = "This is a short benchmark of the F five speech model.";
        int steps = int.TryParse(Environment.GetEnvironmentVariable("F5_STEPS"), out int s) ? s : 16;
        (double gen, double rtf) = Time(() => pipe.GenerateFromAudio(b, refClip, refA.SampleRate, refText, tgt,
            new F5TtsOptions { Steps = steps, Seed = 7 }), 24_000, runs: 2);
        _out.WriteLine($"OURS F5 ({bn}, {steps} steps): gen {gen:F3}s | RTF {rtf:F3}");
    }

    [Fact]
    public void Bench_Piper()
    {
        string onnx = Path.Combine(AudioModelCache.GetRepoDirectory("rhasspy/piper-voices", "tts"),
            "en", "en_US", "lessac", "medium", "en_US-lessac-medium.onnx");
        if (!File.Exists(onnx) || string.IsNullOrEmpty(Environment.GetEnvironmentVariable("ESPEAK_DATA_DIR")))
        { _out.WriteLine("Piper onnx or ESPEAK_DATA_DIR missing — skip."); return; }
        using PiperPipeline p = PiperPipeline.LoadFromFiles(onnx, onnx + ".json");
        using IBackend b = MakeBackend(out string bn);
        const string text = "Hello world. This is a test of the speech synthesizer.";
        (double gen, double rtf) = Time(() => p.SynthesizeText(b, text, seed: 1234), p.SampleRate);
        _out.WriteLine($"OURS Piper ({bn}): gen {gen:F3}s | RTF {rtf:F3}");
    }

    [Fact]
    public void Bench_MeloTts()
    {
        string ckpt = Path.Combine(AudioModelCache.GetRepoDirectory("myshell-ai/MeloTTS-English-v3", "tts"), "checkpoint.pth");
        string bd = AudioModelCache.GetRepoDirectory("bert-base-uncased", "tts");
        string bert = File.Exists(Path.Combine(bd, "model.safetensors")) ? Path.Combine(bd, "model.safetensors") : Path.Combine(bd, "pytorch_model.bin");
        string vocab = Path.Combine(bd, "vocab.txt");
        if (!File.Exists(ckpt) || !File.Exists(bert)) { _out.WriteLine("MeloTTS not cached — skip."); return; }
        MeloTtsConfig cfg = MeloTtsConfig.EnglishV3 with { NoiseScaleW = 0f };
        using MeloTts m = MeloTts.LoadFromFiles(ckpt, bert, vocab, cfg);
        using IBackend b = MakeBackend(out string bn);
        const string text = "Hello world. This is a test of the speech synthesizer.";
        (double gen, double rtf) = Time(() => m.SynthesizeText(b, text, lengthScale: 1.0f, noiseScale: 0.0f, seed: 0), m.SampleRate);
        _out.WriteLine($"OURS MeloTTS ({bn}): gen {gen:F3}s | RTF {rtf:F3}");
    }
}
