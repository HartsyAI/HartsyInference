using System.Diagnostics;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Text;
using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Io;
using HartsyInference.Audio.Models.CosyVoice;
using HartsyInference.Audio.Phonemizer.Espeak;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.ModelAssets.PyTorch;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Real-weight A/B harness for the models that share Kokoro's blocks and the NSF vocoder DSP: StyleTTS 2
/// (predictor, decoder blocks, <c>AdaSnakeResLoader</c>, <c>BiLstm</c>) and CosyVoice 2 (<c>NsfVocoderDsp</c>
/// harmonic source, STFT and iSTFT head through <c>HiFTNetVocoder</c>). Opt-in with
/// <c>HARTSY_SHARED_BLOCK_REGRESSION=1</c>; <c>HARTSY_SHARED_BLOCK_MODELS</c> selects a comma list of
/// <c>styletts2</c> / <c>cosyvoice2</c> (default both). Run once on the baseline build with
/// <c>HARTSY_SHARED_BLOCK_OUT_DIR</c>, then on the candidate with <c>HARTSY_SHARED_BLOCK_REF_DIR</c> pointing at
/// that directory: each case reports length, waveform and log-spectral correlation against the baseline PCM, and
/// transcribes BOTH arms' audio in the same process with whisper-tiny and whisper-medium (multilingual, English
/// forced), so the transcripts are compared by one ASR build.
///
/// <para>Each model is driven the way the engine drives it. StyleTTS 2: espeak <c>en-us</c> IPA with punctuation
/// preserved (<c>StyleTts2Model</c>'s front-end) and a clean 24 kHz read-speech reference (Sesame's
/// <c>prompts/read_speech_c.wav</c>, 8 s) — not the ~3.4 kHz-bandwidth JFK clip. CosyVoice 2: the JFK clip with its
/// transcript as the prompt text, byte-level Qwen tokens, seed 7. Same device rule as <see cref="KokoroBenchTests"/>:
/// the CUDA ordinal from <c>HARTSY_SHARED_BLOCK_CUDA_ORDINAL</c> (default 1) must be the 3060.</para></summary>
[Trait("Category", "GpuIntegration")]
[Trait("Category", "RealWeights")]
public sealed class SharedBlockRegressionTests
{
    private const string GateEnvVar = "HARTSY_SHARED_BLOCK_REGRESSION";
    private const string ModelsEnvVar = "HARTSY_SHARED_BLOCK_MODELS";
    private const string OrdinalEnvVar = "HARTSY_SHARED_BLOCK_CUDA_ORDINAL";
    private const string OutEnvVar = "HARTSY_SHARED_BLOCK_OUT";
    private const string OutDirEnvVar = "HARTSY_SHARED_BLOCK_OUT_DIR";
    private const string RefDirEnvVar = "HARTSY_SHARED_BLOCK_REF_DIR";
    private const string ExactEnvVar = "HARTSY_SHARED_BLOCK_EXACT";
    private const int WhisperRate = 16_000;
    private const string WhisperTiny = "openai/whisper-tiny";
    private const string WhisperMedium = "openai/whisper-medium";
    private const string StyleTts2Repo = "yl4579/StyleTTS2-LibriTTS";
    private const string StyleTts2Language = "en-us";
    private const string ReferenceRepo = "sesame/csm-1b";
    private const string CosyVoiceRepo = "FunAudioLLM/CosyVoice2-0.5B";
    private const string ChatterboxRepo = "ResembleAI/chatterbox";
    private const string JfkTranscript =
        "And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.";

    private static readonly (int Words, string Text)[] Sentences =
    [
        (5, "Please hold while I check."),
        (15, "Thanks for calling, I can see your appointment is booked for Tuesday afternoon at three."),
        (30, "I have updated the delivery address on your order, the driver will call you when they are ten minutes "
            + "away, and you will receive a message with the tracking link."),
    ];

    private readonly ITestOutputHelper _out;

    public SharedBlockRegressionTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public async Task SharedBlocks_StyleTts2_And_CosyVoice2_On_3060()
    {
        if (Environment.GetEnvironmentVariable(GateEnvVar) != "1")
        {
            _out.WriteLine($"SKIPPED: set {GateEnvVar}=1 to run the shared-block regression harness.");
            return;
        }
        string models = Environment.GetEnvironmentVariable(ModelsEnvVar) ?? "styletts2,cosyvoice2";
        string jfk = Path.Combine(RepoRoot.Path, "tests", "python-reference", "silerovad_reference", "jfk.wav");
        if (!RealWeightGate.Require(_out.WriteLine,
                GpuBenchSupport.WhisperFiles(WhisperTiny).Concat(GpuBenchSupport.WhisperFiles(WhisperMedium)).Concat([jfk]).ToArray()))
        {
            return;
        }
        string? outDir = Environment.GetEnvironmentVariable(OutDirEnvVar);
        string? refDir = Environment.GetEnvironmentVariable(RefDirEnvVar);
        if (!string.IsNullOrEmpty(outDir))
        {
            Directory.CreateDirectory(outDir);
        }
        // Full-F32 mode: no TF32 GEMMs and the direct F32 conv kernels instead of cuDNN's TF32 engines, so an arm can
        // be compared against the model at full precision rather than only against another TF32 run.
        bool exact = Environment.GetEnvironmentVariable(ExactEnvVar) == "1";
        if (exact)
        {
            KnobStore.Set(EngineKnobs.HighPrecisionGemm, true);
            KnobStore.Set(EngineKnobs.NoTf32, true);
            KnobStore.Set(EngineKnobs.AudioConvCudnn, false);
        }
        try
        {
            using IBackend backend = GpuBenchSupport.Open3060(_out.WriteLine, OrdinalEnvVar);
            using WhisperPipeline tiny = await WhisperPipeline.LoadAsync(WhisperTiny);
            using WhisperPipeline medium = await WhisperPipeline.LoadAsync(WhisperMedium);
            Verifiers verifiers = new Verifiers(tiny, medium);

            StringBuilder table = new StringBuilder();
            string precision = exact ? ", full F32" : "";
            table.AppendLine($"### Shared-block regression — {backend.Capabilities.DeviceName}, in-process, real weights{precision}");
            table.AppendLine();
            table.AppendLine("| Model | Words | wall ms | audio s | samples | sha256[:12] | vs ref len | vs ref wave corr | vs ref log-spec corr "
                + "| tiny recall ref → new | medium recall ref → new | medium transcripts equal | medium transcript (new) |");
            table.AppendLine("|---|---:|---:|---:|---:|---|---|---:|---:|---|---|---|---|");

            if (models.Contains("styletts2", StringComparison.OrdinalIgnoreCase))
            {
                RunStyleTts2(backend, verifiers, outDir, refDir, table);
            }
            if (models.Contains("cosyvoice2", StringComparison.OrdinalIgnoreCase))
            {
                WavFile.DecodedAudio decoded = WavFile.Read(jfk);
                float[] jfk24k = Resampler.Create(decoded.SampleRate, 24_000).Resample(decoded.ToMono());
                RunCosyVoice2(backend, verifiers, jfk24k, outDir, refDir, table);
            }
            GpuBenchSupport.Emit(_out.WriteLine, table, OutEnvVar);
        }
        finally
        {
            if (exact)
            {
                KnobStore.Clear(EngineKnobs.HighPrecisionGemm);
                KnobStore.Clear(EngineKnobs.NoTf32);
                KnobStore.Clear(EngineKnobs.AudioConvCudnn);
            }
        }
    }

    /// <summary>StyleTTS 2 LibriTTS zero-shot clone through the engine's own front-end: espeak <c>en-us</c> IPA with
    /// punctuation preserved, a clean 24 kHz reference clip, the three bench sentences.</summary>
    private void RunStyleTts2(IBackend backend, Verifiers verifiers, string? outDir, string? refDir, StringBuilder table)
    {
        string checkpoint = Path.Combine(AudioModelCache.GetRepoDirectory(StyleTts2Repo, "tts"), "Models", "LibriTTS", "epochs_2nd_00020.pth");
        string referenceClip = Path.Combine(AudioModelCache.GetRepoDirectory(ReferenceRepo, "tts"), "prompts", "read_speech_c.wav");
        if (!RealWeightGate.Require(_out.WriteLine, checkpoint, referenceClip)) return;
        WavFile.DecodedAudio decoded = WavFile.Read(referenceClip);
        Assert.Equal(24_000, decoded.SampleRate);
        float[] reference = decoded.ToMono();
        EspeakPhonemizer phonemizer = EspeakPhonemizer.FromCache(StyleTts2Language);
        Stopwatch load = Stopwatch.StartNew();
        using StyleTts2Pipeline pipeline = StyleTts2Pipeline.LoadFromCheckpoint(checkpoint);
        _out.WriteLine($"StyleTTS2 loaded in {load.Elapsed.TotalSeconds:F1}s; reference {referenceClip} "
            + $"({reference.Length / 24_000.0:F2}s @ 24 kHz)");
        foreach ((int words, string text) in Sentences)
        {
            // StyleTts2Model's Ipa(): espeak IPA with punctuation kept, whitespace-normalized.
            string ipa = string.Join(' ', phonemizer.PhonemizeToIpa(text, StyleTts2Language, preservePunctuation: true)
                .Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries));
            _out.WriteLine($"styletts2 {words}w IPA: {ipa}");
            pipeline.SynthesizeCloneFromAudio(backend, ipa, reference, 24_000, 1f);
            Stopwatch sw = Stopwatch.StartNew();
            float[] wave = pipeline.SynthesizeCloneFromAudio(backend, ipa, reference, 24_000, 1f);
            sw.Stop();
            Report(backend, verifiers, "styletts2", words, text, wave, 24_000, sw.Elapsed.TotalSeconds, outDir, refDir, table);
        }
    }

    /// <summary>CosyVoice 2 zero-shot clone from the JFK clip with its transcript as the prompt text, built the way
    /// the engine's descriptor builds it (Qwen LM + flow + HiFTNet from the .pt files, S3 tokenizer + CAM++ from
    /// chatterbox's s3gen.safetensors), two sentences at a fixed seed.</summary>
    private void RunCosyVoice2(IBackend backend, Verifiers verifiers, float[] reference24k, string? outDir, string? refDir,
        StringBuilder table)
    {
        string cosyDir = AudioModelCache.GetRepoDirectory(CosyVoiceRepo, "tts");
        string llmPath = Path.Combine(cosyDir, "llm.pt");
        string flowPath = Path.Combine(cosyDir, "flow.pt");
        string hiftPath = Path.Combine(cosyDir, "hift.pt");
        string s3genPath = Path.Combine(AudioModelCache.GetRepoDirectory(ChatterboxRepo, "tts"), "s3gen.safetensors");
        if (!RealWeightGate.Require(_out.WriteLine, llmPath, flowPath, hiftPath, s3genPath)) return;

        Stopwatch load = Stopwatch.StartNew();
        using AnyFormatCheckpointLoader llmLoader = new AnyFormatCheckpointLoader();
        llmLoader.Load(llmPath);
        using AnyFormatCheckpointLoader flowLoader = new AnyFormatCheckpointLoader();
        flowLoader.Load(flowPath);
        using AnyFormatCheckpointLoader hiftLoader = new AnyFormatCheckpointLoader();
        hiftLoader.Load(hiftPath);
        using SafeTensorsLoader s3genLoader = new SafeTensorsLoader();
        s3genLoader.Load(s3genPath);
        Dictionary<string, Tensor> s3gen = s3genLoader.GetAllTensors();

        CosyVoiceConfig config = CosyVoiceConfig.V2_0_5B;
        CosyVoiceQwenLm lm = new CosyVoiceQwenLm(config);
        lm.LoadWeights(llmLoader.GetAllTensors());
        CosyVoiceFlow flow = new CosyVoiceFlow(config);
        flow.LoadWeights(flowLoader.GetAllTensors());
        HiFTNetVocoder vocoder = new HiFTNetVocoder(config.Hift);
        vocoder.LoadWeights(hiftLoader.GetAllTensors());
        CamPlusSpeakerEncoder speaker = new CamPlusSpeakerEncoder(config.Flow.SpeakerEmbedDim);
        speaker.LoadWeights(s3gen, "speaker_encoder");
        S3Tokenizer s3 = new S3Tokenizer();
        s3.LoadWeights(s3gen, "tokenizer");
        using CosyVoicePipeline pipeline = new CosyVoicePipeline(config, lm, flow, vocoder, speaker, s3);
        using Qwen2Tokenizer tokenizer = new Qwen2Tokenizer();
        _out.WriteLine($"CosyVoice 2 loaded in {load.Elapsed.TotalSeconds:F1}s");
        int[] referenceTextTokens = [.. tokenizer.EncodeRawByteLevel(JfkTranscript)];

        foreach ((int words, string text) in Sentences.Take(2))
        {
            int[] textTokens = [.. tokenizer.EncodeRawByteLevel(text)];
            Stopwatch sw = Stopwatch.StartNew();
            float[] wave = pipeline.Synthesize(backend, textTokens, referenceAudio: reference24k, referenceSampleRate: 24_000,
                referenceTextTokens: referenceTextTokens, seed: 7);
            sw.Stop();
            Report(backend, verifiers, "cosyvoice2", words, text, wave, config.SampleRate, sw.Elapsed.TotalSeconds, outDir, refDir, table);
        }
    }

    private void Report(IBackend backend, Verifiers verifiers, string model, int words, string text, float[] wave, int rate,
        double seconds, string? outDir, string? refDir, StringBuilder table)
    {
        string name = $"{model}_{words}w";
        string digest = AudioParityMetrics.Sha256Hex(wave)[..12];
        if (!string.IsNullOrEmpty(outDir))
        {
            File.WriteAllBytes(Path.Combine(outDir, name + ".f32"), MemoryMarshal.AsBytes<float>(wave).ToArray());
            WavFile.WriteMono16(Path.Combine(outDir, name + ".wav"), wave, rate);
        }
        string tinyNew = Hear(backend, verifiers.Tiny, wave, rate);
        string mediumNew = Hear(backend, verifiers.Medium, wave, rate);
        string len = "—", corr = "—", specCorr = "—", tinyRecall, mediumRecall, equal = "—";
        string tinyNewRecall = AudioParityMetrics.ContentWordRecall(text, tinyNew).ToString("P0", CultureInfo.InvariantCulture);
        string mediumNewRecall = AudioParityMetrics.ContentWordRecall(text, mediumNew).ToString("P0", CultureInfo.InvariantCulture);
        tinyRecall = tinyNewRecall;
        mediumRecall = mediumNewRecall;
        if (!string.IsNullOrEmpty(refDir) && File.Exists(Path.Combine(refDir, name + ".f32")))
        {
            float[] reference = MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(Path.Combine(refDir, name + ".f32"))).ToArray();
            len = reference.Length == wave.Length ? "identical" : $"{reference.Length}→{wave.Length}";
            corr = AudioParityMetrics.Compare(reference, wave).Corr.ToString("F6", CultureInfo.InvariantCulture);
            specCorr = AudioParityMetrics.LogSpectralCorrelation(reference, wave).ToString("F6", CultureInfo.InvariantCulture);
            string tinyRef = Hear(backend, verifiers.Tiny, reference, rate);
            string mediumRef = Hear(backend, verifiers.Medium, reference, rate);
            string tinyRefRecall = AudioParityMetrics.ContentWordRecall(text, tinyRef).ToString("P0", CultureInfo.InvariantCulture);
            string mediumRefRecall = AudioParityMetrics.ContentWordRecall(text, mediumRef).ToString("P0", CultureInfo.InvariantCulture);
            tinyRecall = $"{tinyRefRecall} → {tinyNewRecall}";
            mediumRecall = $"{mediumRefRecall} → {mediumNewRecall}";
            equal = string.Equals(mediumRef.Trim(), mediumNew.Trim(), StringComparison.Ordinal) ? "yes" : "no";
            _out.WriteLine($"{model} {words}w REF transcripts: tiny '{tinyRef.Trim()}' | medium '{mediumRef.Trim()}'");
        }
        double audioSeconds = wave.Length / (double)rate;
        table.AppendLine($"| {model} | {words} | {AudioParityMetrics.Ms(seconds)} | {audioSeconds:F2} | {wave.Length} | `{digest}` | {len} | "
            + $"{corr} | {specCorr} | {tinyRecall} | {mediumRecall} | {equal} | {AudioParityMetrics.Cell(mediumNew)} |");
        _out.WriteLine($"{model} {words}w: {AudioParityMetrics.Ms(seconds)} ms | {audioSeconds:F2}s ({wave.Length} samples) | sha {digest} | "
            + $"len {len} wave corr {corr} log-spec corr {specCorr} | tiny recall {tinyRecall} | medium recall {mediumRecall} | "
            + $"medium equal {equal} | NEW tiny '{tinyNew.Trim()}' | NEW medium '{mediumNew.Trim()}'");
    }

    private static string Hear(IBackend backend, WhisperPipeline whisper, float[] wave, int rate)
    {
        float[] audio16k = rate == WhisperRate ? wave : Resampler.Create(rate, WhisperRate).Resample(wave);
        return whisper.TranscribeAudio(backend, audio16k, WhisperRate, new WhisperOptions { Language = "en" });
    }

    /// <summary>The two ASR models every clip is transcribed with.</summary>
    private readonly record struct Verifiers(WhisperPipeline Tiny, WhisperPipeline Medium);
}
