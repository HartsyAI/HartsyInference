using System.Globalization;
using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Io;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Audio.Preprocessing;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Phonemizer.Espeak;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Real-weight end-to-end StyleTTS2-LibriTTS voice-clone: a reference mel (the verified
/// <see cref="StyleEncoder"/> input) → 256-d style, then synthesize an English sentence in that voice via the
/// Kokoro backbone → WAV → Whisper. Writes the WAV to <c>{TmpPath}/hartsyinference_tts_to_stt/</c> for the human
/// listen. Gated on <c>STYLE_CKPT</c> (LibriTTS <c>.pth</c>) + <c>STYLE_TOKENIZER</c> (vocab config) +
/// <c>STYLE_REF_DIR/mel.txt</c> (the reference mel).</summary>
public sealed class StyleTts2CloneEndToEndTests
{
    private readonly ITestOutputHelper _out;
    public StyleTts2CloneEndToEndTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public async Task Clone_Synthesizes_Intelligible_Speech()
    {
        string? ckpt = Environment.GetEnvironmentVariable("STYLE_CKPT");
        string? tokCfg = Environment.GetEnvironmentVariable("STYLE_TOKENIZER");
        string? refDir = Environment.GetEnvironmentVariable("STYLE_REF_DIR");
        string whisperDir = AudioModelCache.GetRepoDirectory("openai/whisper-base");
        if (ckpt is null || !File.Exists(ckpt) || tokCfg is null || !File.Exists(tokCfg)
            || refDir is null || !File.Exists(Path.Combine(refDir, "mel.txt"))
            || !File.Exists(Path.Combine(whisperDir, "model.safetensors")))
        {
            _out.WriteLine("STYLE_CKPT / STYLE_TOKENIZER / STYLE_REF_DIR / whisper not all present — skipping.");
            return;
        }

        string text = Environment.GetEnvironmentVariable("STYLE_TEXT")
            ?? "Hello there. This is a test of the style text to speech model.";
        using StyleTts2Pipeline pipe = StyleTts2Pipeline.LoadFromCheckpoint(ckpt, tokCfg);
        using CpuBackend backend = new();

        // Phonemize the text to IPA (StyleTTS2's 178-symbol vocab; the tokenizer drops out-of-vocab chars).
        EspeakPhonemizer phon = EspeakPhonemizer.FromCache("en");
        string ipa = phon.PhonemizeToIpa(text, "en");
        _out.WriteLine($"text: \"{text}\"");
        _out.WriteLine($"ipa:  \"{ipa}\"");

        using Tensor refMel = ReadMel(Path.Combine(refDir, "mel.txt"));
        float[] pcm = pipe.SynthesizeClone(backend, ipa, refMel, speed: 1f);

        Assert.NotEmpty(pcm);
        double sumSq = 0; float peak = 0;
        foreach (float v in pcm) { Assert.True(float.IsFinite(v), "non-finite sample"); sumSq += (double)v * v; peak = Math.Max(peak, Math.Abs(v)); }
        double rms = Math.Sqrt(sumSq / pcm.Length);

        string outDir = Path.Combine(Path.GetTempPath(), "hartsyinference_tts_to_stt");
        Directory.CreateDirectory(outDir);
        string outWav = Path.Combine(outDir, $"styletts2_clone_{DateTime.Now:yyyyMMdd_HHmmss}.wav");
        WavFile.WriteMono16(outWav, pcm, 24_000);
        _out.WriteLine($"StyleTTS2 clone: {pcm.Length} samples ({pcm.Length / 24000.0:F2}s @ 24kHz) RMS={rms:F4} peak={peak:F4}");
        _out.WriteLine($"WAV (listen): {outWav}");
        Assert.True(rms > 1e-4, "output is silent");

        Resampler down = Resampler.Create(24_000, 16_000);
        float[] stt16 = down.Resample(pcm);
        string sttWav = outWav + ".16k.wav";
        WavFile.WriteMono16(sttWav, stt16, 16_000);
        using WhisperPipeline stt = await WhisperPipeline.LoadAsync("openai/whisper-base");
        using CpuBackend sttBackend = new();
        string heard = stt.TranscribeWav(sttBackend, sttWav,
            new WhisperOptions { Language = "en", Translate = false, WithTimestamps = false }).Trim();
        _out.WriteLine($"Whisper heard: \"{heard}\"");
        string[] content = ["hello", "test", "style", "text", "speech", "model", "there"];
        int hits = content.Count(x => heard.ToLowerInvariant().Contains(x));
        _out.WriteLine($"Content-word recall: {hits}/{content.Length} ({string.Join(",", content.Where(x => heard.ToLowerInvariant().Contains(x)))})");
        Assert.NotNull(heard);
    }

    private static unsafe Tensor ReadMel(string path)
    {
        string[] lines = File.ReadAllLines(path);
        string[] hdr = lines[0].Split(' ', StringSplitOptions.RemoveEmptyEntries);
        int m = int.Parse(hdr[0]), t = int.Parse(hdr[1]);
        string[] vals = lines[1].Split(' ', StringSplitOptions.RemoveEmptyEntries);
        Tensor mel = new(new TensorShape(1, m, t), DType.F32);
        float* mp = (float*)mel.DataPointer;
        for (int i = 0; i < m * t; i++) mp[i] = float.Parse(vals[i], CultureInfo.InvariantCulture);
        return mel;
    }
}
