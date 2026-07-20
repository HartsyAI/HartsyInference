using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Io;
using HartsyInference.Audio.Models.Vocoders;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Audio.Preprocessing;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>End-to-end audio infrastructure validation that bypasses any TTS model.
/// Demonstrates that the WAV I/O, mel preprocessing, Vocos vocoder, and Whisper STT
/// all integrate correctly numerically by round-tripping the JFK reference clip
/// through mel → Vocos → audio → Whisper.
///
/// <para>If this test passes, the infrastructure stack is sound and any TTS-output
/// issues (e.g. F5's flow-matching producing the wrong mel scale) are localized to the
/// TTS model itself — not anywhere in the audio chain it feeds into.</para>
///
/// <para>Loop: <c>JFK 16kHz → resample to 24kHz → mel → Vocos → 24kHz audio →
/// resample to 16kHz → Whisper-tiny → text</c>. Asserts canonical phrases appear in
/// the transcript.</para></summary>
public sealed class VocoderRoundTripTests
{
    private readonly ITestOutputHelper _out;
    public VocoderRoundTripTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public async Task JfkClip_ThroughMelAndVocos_RecoversCanonicalTranscript()
    {
        string jfkPath = Path.Combine(AudioModelCache.CacheRoot, "test-clips", "jfk.wav");
        string vocosRepo = AudioModelCache.GetRepoDirectory("charactr/vocos-mel-24khz");
        string vocosWeights = Path.Combine(vocosRepo, "model.safetensors");
        string whisperRepo = AudioModelCache.GetRepoDirectory("openai/whisper-tiny");

        if (!File.Exists(jfkPath) || !File.Exists(vocosWeights) || !File.Exists(Path.Combine(whisperRepo, "model.safetensors")))
        {
            _out.WriteLine("Required cached models missing — skipping.");
            return;
        }

        // 1. Load JFK clip, resample to 24 kHz (Vocos's native rate).
        WavFile.DecodedAudio jfk = WavFile.Read(jfkPath);
        Resampler upsampler = Resampler.Create(jfk.SampleRate, 24_000);
        float[] s24 = upsampler.Resample(jfk.Channels[0]);
        _out.WriteLine($"Loaded JFK: {jfk.Channels[0].Length} samples at {jfk.SampleRate} Hz, resampled to {s24.Length} at 24 kHz.");

        // 2. Extract Vocos's expected mel: 100 bins, n_fft=1024, hop=256, natural log.
        MelSpectrogramExtractor.Config melCfg = new(
            SampleRate: 24_000, NFft: 1_024, WinLength: 1_024, HopLength: 256,
            NMels: 100, Fmin: 0.0, Fmax: 12_000.0,
            Norm: MelSpectrogramExtractor.Normalization.None,
            DropLastStftFrame: false,
            LogBase: MelSpectrogramExtractor.LogBase.Natural,
            LogFloor: 1e-5f,
            DynamicRangeDb: 0f, NormOffset: 0f, NormScale: 1f, PowerSpectrum: false);
        MelSpectrogramExtractor mel = new(melCfg);
        float[,] melArr = mel.Compute(s24);
        int tMel = melArr.GetLength(1);
        Tensor melT = new(new TensorShape(1, 100, tMel), DType.F32);
        unsafe
        {
            float* p = (float*)melT.DataPointer;
            for (int m = 0; m < 100; m++)
                for (int j = 0; j < tMel; j++)
                    p[m * tMel + j] = melArr[m, j];
        }
        _out.WriteLine($"Mel: [100, {tMel}].");

        // 3. Run Vocos. The OutputGain knob compensates for the empirical 44.53× scale
        //    mismatch between Vocos's predicted log-mag and the audio's true STFT.
        using CpuBackend backend = new();
        Vocos vocos = new(VocosConfig.Mel24k);
        SafeTensorsLoader vocosLoader = new();
        vocosLoader.Load(vocosWeights);
        vocos.LoadWeights(vocosLoader.GetAllTensors());
        float[] reconstructed = vocos.Forward(backend, melT);

        // 4. Sanity check on reconstructed amplitude — must match original RMS within 50%.
        double inSq = 0d, outSq = 0d;
        for (int i = 0; i < s24.Length; i++) inSq += (double)s24[i] * s24[i];
        for (int i = 0; i < reconstructed.Length; i++) outSq += (double)reconstructed[i] * reconstructed[i];
        double inRms = Math.Sqrt(inSq / s24.Length);
        double outRms = Math.Sqrt(outSq / reconstructed.Length);
        _out.WriteLine($"Input RMS={inRms:F4}, Vocos reconstruction RMS={outRms:F4} (ratio {outRms / inRms:F2}).");
        Assert.InRange(outRms / inRms, 0.5, 2.0);

        // 5. Resample reconstructed audio to 16 kHz, run Whisper.
        Resampler downsampler = Resampler.Create(24_000, 16_000);
        float[] s16 = downsampler.Resample(reconstructed);

        string outDir = Path.Combine(Path.GetTempPath(), "hartsyinference_tts_to_stt");
        Directory.CreateDirectory(outDir);
        string outPath = Path.Combine(outDir, $"vocos_roundtrip_{DateTime.Now:yyyyMMdd_HHmmss}.wav");
        WavFile.WriteMono16(outPath, s16, 16_000);
        _out.WriteLine($"Wrote Vocos round-trip 16 kHz: {outPath}");

        using WhisperPipeline whisper = await WhisperPipeline.LoadAsync("openai/whisper-tiny");
        WhisperOptions options = new() { Language = "en", Translate = false, WithTimestamps = false };
        string transcript = whisper.TranscribeWav(backend, outPath, options).Trim().ToLowerInvariant();
        _out.WriteLine($"Whisper transcript: \"{transcript}\"");

        // 6. The canonical JFK line has "fellow americans" and "country" — those words
        //    are robust against any minor reconstruction artifacts in Vocos.
        Assert.Contains("fellow americans", transcript);
        Assert.Contains("country", transcript);
    }

    /// <summary>Cross-STT verification: same JFK → Vocos round-trip → both Whisper and
    /// Moonshine. Both must recover the canonical text — this rules out any single STT
    /// model being lenient with our Vocos output quality.</summary>
    [Fact]
    public async Task JfkClip_VocosRoundTrip_PassesBothSttModels()
    {
        string jfkPath = Path.Combine(AudioModelCache.CacheRoot, "test-clips", "jfk.wav");
        string vocosRepo = AudioModelCache.GetRepoDirectory("charactr/vocos-mel-24khz");
        string vocosWeights = Path.Combine(vocosRepo, "model.safetensors");
        string whisperRepo = AudioModelCache.GetRepoDirectory("openai/whisper-tiny");
        string moonshineRepo = AudioModelCache.GetRepoDirectory("UsefulSensors/moonshine-base");

        if (!File.Exists(jfkPath) || !File.Exists(vocosWeights)
            || !File.Exists(Path.Combine(whisperRepo, "model.safetensors"))
            || !File.Exists(Path.Combine(moonshineRepo, "model.safetensors")))
        {
            _out.WriteLine("Required cached models missing — skipping.");
            return;
        }

        // Resample + mel + Vocos.
        WavFile.DecodedAudio jfk = WavFile.Read(jfkPath);
        float[] s24 = Resampler.Create(jfk.SampleRate, 24_000).Resample(jfk.Channels[0]);
        MelSpectrogramExtractor.Config melCfg = new(
            24_000, 1_024, 1_024, 256, 100, 0.0, 12_000.0,
            MelSpectrogramExtractor.Normalization.None, false,
            MelSpectrogramExtractor.LogBase.Natural, 1e-5f, 0f, 0f, 1f, false);
        float[,] melArr = new MelSpectrogramExtractor(melCfg).Compute(s24);
        int tMel = melArr.GetLength(1);
        Tensor melT = new(new TensorShape(1, 100, tMel), DType.F32);
        unsafe { float* p = (float*)melT.DataPointer; for (int m = 0; m < 100; m++) for (int j = 0; j < tMel; j++) p[m * tMel + j] = melArr[m, j]; }

        using CpuBackend backend = new();
        Vocos vocos = new(VocosConfig.Mel24k);
        SafeTensorsLoader loader = new();
        loader.Load(vocosWeights);
        vocos.LoadWeights(loader.GetAllTensors());
        float[] reconstructed = vocos.Forward(backend, melT);
        float[] s16 = Resampler.Create(24_000, 16_000).Resample(reconstructed);

        string outPath = Path.Combine(Path.GetTempPath(), "hartsyinference_tts_to_stt", $"vocos_xstt_{DateTime.Now:yyyyMMdd_HHmmss}.wav");
        Directory.CreateDirectory(Path.GetDirectoryName(outPath)!);
        WavFile.WriteMono16(outPath, s16, 16_000);

        // Whisper.
        using WhisperPipeline whisper = await WhisperPipeline.LoadAsync("openai/whisper-tiny");
        string wTxt = whisper.TranscribeWav(backend, outPath, new WhisperOptions { Language = "en", Translate = false, WithTimestamps = false }).Trim().ToLowerInvariant();
        _out.WriteLine($"Whisper:   \"{wTxt}\"");

        // Moonshine.
        using MoonshinePipeline moon = await MoonshinePipeline.LoadAsync("UsefulSensors/moonshine-base");
        string mTxt = moon.TranscribeWav(backend, outPath).Trim().ToLowerInvariant();
        _out.WriteLine($"Moonshine: \"{mTxt}\"");

        Assert.Contains("fellow americans", wTxt);
        Assert.Contains("country", wTxt);
        Assert.Contains("fellow americans", mTxt);
        Assert.Contains("country", mTxt);
    }
}
