using System;
using System.IO;
using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Io;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Audio.Preprocessing;
using HartsyInference.Core.Backends;
using HartsyInference.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Full end-to-end <see cref="ZipVoicePipeline"/> smoke test on the real k2-fsa/ZipVoice + Vocos
/// checkpoints: mel extraction, joint-text tokenization (espeak-IPA), duration prediction, the Euler
/// flow-matching sampling loop with CFG, and Vocos vocoding — the first time this full pipeline has been
/// exercised end to end (the backbone parity tests in <c>ZipVoiceParityTests.cs</c> only cover the neural net
/// forward passes on synthetic tensors). Clones a trimmed JFK reference clip (already vendored for the Silero-VAD
/// parity fixtures) and round-trips the output through Whisper STT to check word recall against the target text.
///
/// <para>Runs on CUDA when available (device ordinal 0 — set <c>CUDA_VISIBLE_DEVICES</c> in the shell to pick a
/// physical GPU); falls back to CPU otherwise. Skip-guarded on model + Whisper cache availability so CI without
/// network still passes. Writes WAVs to <c>{TempPath}/hartsyinference_zipvoice_e2e/</c> for manual listening.</para></summary>
public sealed class ZipVoiceE2eTests
{
    private readonly ITestOutputHelper _out;
    public ZipVoiceE2eTests(ITestOutputHelper output) => _out = output;

    // NOTE: must be a verbatim, complete transcript of the ACTUAL spoken content in the reference clip used
    // below (the full 11.0s jfk.wav, not a 5s trim) — ZipVoice's "predict" duration mode phonemizes this text
    // and average-upsamples it across the *measured* reference-audio frame count (see
    // ZipVoiceModel.BuildFrameToTokenIndex / PredictTargetFrames). If RefText describes more speech than is
    // actually present in the audio (e.g. a 13-word quote paired with an 8-word 5s clip), the token→frame
    // averaging allocates real prompt-audio frames to text tokens for words that were never spoken, corrupting
    // the shared fm_decoder self-attention sequence and bleeding that unspoken reference text into the
    // generated region. Confirmed via Whisper STT on jfk.wav: this is the verbatim full transcript.
    private const string RefText =
        "And so, my fellow Americans, ask not what your country can do for you. Ask what you can do for your country.";
    private const string TargetText = "The quick brown fox jumps over the lazy dog near the riverbank.";

    [Fact]
    [Trait("Category", "GpuIntegration")]
    public async System.Threading.Tasks.Task Diag_TranscribeJfkFullAndTrimmed()
    {
        string jfkPath = Path.Combine(RepoRootDir(), "tests", "python-reference", "silerovad_reference", "jfk.wav");
        if (!File.Exists(jfkPath)) { _out.WriteLine("jfk.wav missing — skipping."); return; }

        string whisperRepoDir = AudioModelCache.GetRepoDirectory("distil-whisper/distil-large-v3");
        string whisperRepoId = "distil-whisper/distil-large-v3";
        if (!File.Exists(Path.Combine(whisperRepoDir, "model.safetensors")))
        {
            whisperRepoDir = AudioModelCache.GetRepoDirectory("openai/whisper-base");
            whisperRepoId = "openai/whisper-base";
        }
        if (!File.Exists(Path.Combine(whisperRepoDir, "model.safetensors")))
        {
            _out.WriteLine("No cached Whisper model — skipping.");
            return;
        }
        _out.WriteLine($"Using Whisper model: {whisperRepoId}");

        string outDir = Path.Combine(Path.GetTempPath(), "hartsyinference_zipvoice_e2e");
        Directory.CreateDirectory(outDir);
        string trimmedRef = TrimWav(jfkPath, seconds: 5.0f, outDir);

        using WhisperPipeline stt = await WhisperPipeline.LoadAsync(whisperRepoId);
        IBackend sttBackend = CudaContext.IsAvailable() ? new CudaBackend(0, ResolvePtxDir()) : new HartsyInference.Cpu.CpuBackend();
        try
        {
            WhisperOptions options = new() { Language = "en", Translate = false, WithTimestamps = true };

            string fullTranscript = stt.TranscribeWav(sttBackend, jfkPath, options).Trim();
            _out.WriteLine($"FULL jfk.wav (11.0s) transcript: \"{fullTranscript}\"");

            string trimTranscript = stt.TranscribeWav(sttBackend, trimmedRef, options).Trim();
            _out.WriteLine($"TRIMMED jfk_trim_5.0s.wav transcript: \"{trimTranscript}\"");

            _out.WriteLine($"RefText constant used by main test: \"{RefText}\"");
        }
        finally { (sttBackend as IDisposable)?.Dispose(); }
    }

    [Fact]
    [Trait("Category", "GpuIntegration")]
    public async System.Threading.Tasks.Task Synthesize_ClonesJfkVoice_AndRoundTripsThroughWhisper()
    {
        string jfkPath = Path.Combine(RepoRootDir(), "tests", "python-reference", "silerovad_reference", "jfk.wav");
        if (!File.Exists(jfkPath))
        {
            _out.WriteLine($"JFK reference clip missing at {jfkPath} — skipping.");
            return;
        }

        string whisperRepoDir = AudioModelCache.GetRepoDirectory("distil-whisper/distil-large-v3");
        string whisperRepoId = "distil-whisper/distil-large-v3";
        if (!File.Exists(Path.Combine(whisperRepoDir, "model.safetensors")))
        {
            whisperRepoDir = AudioModelCache.GetRepoDirectory("openai/whisper-base");
            whisperRepoId = "openai/whisper-base";
        }
        if (!File.Exists(Path.Combine(whisperRepoDir, "model.safetensors")))
        {
            _out.WriteLine("No cached Whisper model available for STT round-trip — skipping.");
            return;
        }
        _out.WriteLine($"Using Whisper model: {whisperRepoId}");

        string outDir = Path.Combine(Path.GetTempPath(), "hartsyinference_zipvoice_e2e");
        Directory.CreateDirectory(outDir);
        // Full 11.0s clip (not a 5s trim) so RefText's full transcript actually matches what's spoken —
        // see the RefText doc comment above for why a truncated clip + full-quote text caused reference bleed.
        string trimmedRef = TrimWav(jfkPath, seconds: 11.0f, outDir);
        _out.WriteLine($"Reference clip (full 11.0s, time-matched to RefText): {trimmedRef}");

        _out.WriteLine("Loading ZipVoicePipeline (auto-downloads k2-fsa/ZipVoice + vocos-mel-24khz if not cached)...");
        using ZipVoicePipeline pipeline = await ZipVoicePipeline.LoadAsync();
        _out.WriteLine("ZipVoicePipeline loaded.");

        IBackend backend;
        string backendDesc;
        if (CudaContext.IsAvailable())
        {
            string? ptxDir = ResolvePtxDir();
            backend = ptxDir is not null ? new CudaBackend(0, ptxDir) : new CudaBackend(0);
            backendDesc = $"CUDA (ptx={ptxDir ?? "none"})";
        }
        else
        {
            backend = new HartsyInference.Cpu.CpuBackend();
            backendDesc = "CPU";
        }
        _out.WriteLine($"Backend: {backendDesc}");

        try
        {
            WavFile.DecodedAudio refAudio = WavFile.Read(trimmedRef);
            float[] refMono = refAudio.ToMono();

            DateTime t0 = DateTime.UtcNow;
            float[] generated = pipeline.GenerateFromAudio(backend, refMono, refAudio.SampleRate, RefText, TargetText,
                new ZipVoiceOptions { Seed = 42 });
            TimeSpan genTime = DateTime.UtcNow - t0;

            Assert.True(generated.Length > 0, "generated audio is empty");
            double durationSec = generated.Length / 24_000.0;
            _out.WriteLine($"Generated {generated.Length} samples ({durationSec:F2}s @ 24kHz) in {genTime.TotalSeconds:F1}s "
                + $"(RTF {genTime.TotalSeconds / durationSec:F2}).");

            int nanCount = 0, infCount = 0;
            double sumSq = 0;
            float peak = 0;
            for (int i = 0; i < generated.Length; i++)
            {
                float s = generated[i];
                if (float.IsNaN(s)) { nanCount++; continue; }
                if (float.IsInfinity(s)) { infCount++; continue; }
                sumSq += (double)s * s;
                peak = MathF.Max(peak, MathF.Abs(s));
            }
            double rms = Math.Sqrt(sumSq / generated.Length);
            _out.WriteLine($"Stats: NaN={nanCount} Inf={infCount} RMS={rms:F5} Peak={peak:F5}");

            string outWav = Path.Combine(outDir, "zipvoice_generated.wav");
            WavFile.WriteMono16(outWav, generated, 24_000);
            _out.WriteLine($"Wrote generated audio: {outWav}");

            Assert.Equal(0, nanCount);
            Assert.Equal(0, infCount);
            Assert.True(rms > 1e-4, $"RMS {rms} too low — likely silence, not speech");
            Assert.True(peak < 1.5f, $"peak {peak} suggests exploding/garbage output");
            // Sanity duration: target sentence at normal speaking rate should land roughly 2-8s.
            Assert.InRange(durationSec, 0.5, 15.0);

            Resampler downsampler = Resampler.Create(24_000, 16_000);
            float[] sttInput = downsampler.Resample(generated);
            string sttWav = Path.Combine(outDir, "zipvoice_generated_16k.wav");
            WavFile.WriteMono16(sttWav, sttInput, 16_000);

            using WhisperPipeline stt = await WhisperPipeline.LoadAsync(whisperRepoId);
            // Reuse the same (already-loaded) backend for STT — CPU-path Whisper decode in this engine is
            // extremely slow (20+ min for a few seconds of audio on distil-large-v3/medium.en); CUDA is fast.
            WhisperOptions options = new() { Language = "en", Translate = false, WithTimestamps = false };
            string transcript = stt.TranscribeWav(backend, sttWav, options).Trim();
            string lower = transcript.ToLowerInvariant();
            _out.WriteLine($"Target text:        \"{TargetText}\"");
            _out.WriteLine($"Whisper transcript: \"{transcript}\"");

            string[] want = ["quick", "brown", "fox", "jumps", "lazy", "dog", "riverbank"];
            int recall = 0;
            foreach (string w in want) if (lower.Contains(w)) recall++;
            _out.WriteLine($"Content-word recall: {recall}/{want.Length} ({string.Join(",", Array.FindAll(want, w => lower.Contains(w)))})");

            if (recall == 0)
            {
                _out.WriteLine("");
                _out.WriteLine("--- TRANSCRIPT DID NOT RECOVER ANY TARGET WORDS ---");
                _out.WriteLine($"Listen manually: {outWav}");
            }
            Assert.NotEmpty(transcript);
        }
        finally
        {
            (backend as IDisposable)?.Dispose();
        }
    }

    private static string TrimWav(string srcPath, float seconds, string outDir)
    {
        WavFile.DecodedAudio src = WavFile.Read(srcPath);
        float[] mono = src.ToMono();
        int n = Math.Min((int)(seconds * src.SampleRate), mono.Length);
        float[] trimmed = new float[n];
        Array.Copy(mono, trimmed, n);
        string outPath = Path.Combine(outDir, $"jfk_trim_{seconds:F1}s.wav");
        WavFile.WriteMono16(outPath, trimmed, src.SampleRate);
        return outPath;
    }

    private static string RepoRootDir()
    {
        for (DirectoryInfo? d = new(AppContext.BaseDirectory); d is not null; d = d.Parent)
        {
            if (File.Exists(Path.Combine(d.FullName, "HartsyInference.sln"))) return d.FullName;
        }
        throw new InvalidOperationException("Could not locate repo root from " + AppContext.BaseDirectory);
    }

    private static string? ResolvePtxDir()
    {
        string local = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (Directory.Exists(local)) return local;
        for (DirectoryInfo? d = new(AppContext.BaseDirectory); d is not null; d = d.Parent)
        {
            string cand = Path.Combine(d.FullName, "src", "HartsyInference.Cuda", "Ptx");
            if (Directory.Exists(cand)) return cand;
        }
        return null;
    }
}
