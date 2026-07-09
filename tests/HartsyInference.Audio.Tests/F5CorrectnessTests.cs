using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Io;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Audio.Preprocessing;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>F5 correctness debug: gen on the EXACT input the Python reference used (full JFK ref + same texts,
/// nfe=32/cfg=2/sway=-1), STT it, and dump our pre-vocoder mel to compare against the reference mel
/// (f5_ref_mel.npy). Localizes DiT/sample-loop vs vocoder. GPU via F5_CUDA=1 + F5_PTX.</summary>
public sealed class F5CorrectnessTests
{
    private readonly ITestOutputHelper _out;
    public F5CorrectnessTests(ITestOutputHelper o) => _out = o;

    [Fact]
    public async Task GenMatchedInput_SttAndDumpMel()
    {
        string dit = Path.Combine(AudioModelCache.GetRepoDirectory("SWivid/F5-TTS"), "F5TTS_v1_Base", "model_1250000.safetensors");
        string jfk = Environment.GetEnvironmentVariable("F5_REF_WAV") ?? Path.Combine(AudioModelCache.CacheRoot, "test-clips", "jfk.wav");
        if (!File.Exists(dit) || !File.Exists(jfk)) { _out.WriteLine("F5/jfk missing — skip."); return; }

        IBackend b = Environment.GetEnvironmentVariable("F5_CUDA") == "1"
            ? new HartsyInference.Cuda.CudaBackend(0, Environment.GetEnvironmentVariable("F5_PTX")!)
            : new CpuBackend();
        using F5TtsPipeline pipe = await F5TtsPipeline.LoadAsync();

        WavFile.DecodedAudio refA = WavFile.Read(jfk);
        const string refText = "And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.";
        const string genText = "The speech synthesizer is now working correctly.";
        F5TtsOptions opts = new() { Steps = 32, CfgStrength = 2.0f, SwayCoef = -1.0f, Seed = 7 };

        float[] audio = pipe.GenerateFromAudio(b, refA.Channels[0], refA.SampleRate, refText, genText, opts);

        string outDir = "/tmp/hartsyinference_tts_to_stt"; Directory.CreateDirectory(outDir);
        string outWav = Path.Combine(outDir, "f5_OURS_matched.wav");
        WavFile.WriteMono16(outWav, audio, 24_000);
        double sumSq = 0; foreach (float v in audio) sumSq += (double)v * v;
        _out.WriteLine($"OURS F5: {audio.Length / 24000.0:F2}s | RMS {Math.Sqrt(sumSq / audio.Length):F4} | {outWav}");

        // STT
        string whisperDir = AudioModelCache.GetRepoDirectory("openai/whisper-base");
        if (File.Exists(Path.Combine(whisperDir, "model.safetensors")))
        {
            float[] s16 = Resampler.Create(24_000, 16_000).Resample(audio);
            string sttWav = Path.Combine(outDir, "f5_OURS_matched_16k.wav");
            WavFile.WriteMono16(sttWav, s16, 16_000);
            using WhisperPipeline stt = await WhisperPipeline.LoadAsync("openai/whisper-base");
            using CpuBackend sb = new();
            string heard = stt.TranscribeWav(sb, sttWav, new WhisperOptions { Language = "en" }).Trim();
            _out.WriteLine($"Target: \"{genText}\"");
            _out.WriteLine($"OURS heard: \"{heard}\"");
        }
        b.Dispose();
    }
}
