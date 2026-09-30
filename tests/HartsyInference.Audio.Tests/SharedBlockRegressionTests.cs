using System.Diagnostics;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Text;
using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Frontends;
using HartsyInference.Audio.Io;
using HartsyInference.Audio.Models.CosyVoice;
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
/// that directory: each case reports length, waveform and log-spectral correlation against the baseline PCM and
/// the whisper-tiny transcript. Same device rule as <see cref="KokoroBenchTests"/>: the CUDA ordinal from
/// <c>HARTSY_SHARED_BLOCK_CUDA_ORDINAL</c> (default 1) must be the 3060.</summary>
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
    private const string RequiredDeviceSubstring = "3060";
    private const int WhisperRate = 16_000;
    private const string WhisperTiny = "openai/whisper-tiny";
    private const string StyleTts2Repo = "yl4579/StyleTTS2-LibriTTS";
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
        string cmudict = Path.Combine(ModelsRoot(), "audio", "cmudict.dict");
        if (!RealWeightGate.Require(_out.WriteLine, WhisperFiles(WhisperTiny).Concat([jfk, cmudict]).ToArray()))
        {
            return;
        }
        string? outDir = Environment.GetEnvironmentVariable(OutDirEnvVar);
        string? refDir = Environment.GetEnvironmentVariable(RefDirEnvVar);
        if (!string.IsNullOrEmpty(outDir))
        {
            Directory.CreateDirectory(outDir);
        }
        using IBackend backend = OpenBackend();
        using WhisperPipeline verify = await WhisperPipeline.LoadAsync(WhisperTiny);
        WavFile.DecodedAudio decoded = WavFile.Read(jfk);
        float[] reference24k = Resampler.Create(decoded.SampleRate, 24_000).Resample(decoded.ToMono());

        StringBuilder table = new StringBuilder();
        table.AppendLine($"### Shared-block regression — {backend.Capabilities.DeviceName}, in-process, real weights");
        table.AppendLine();
        table.AppendLine("| Model | Words | wall ms | audio s | samples | sha256[:12] | vs ref len | vs ref wave corr | vs ref log-spec corr | recall | transcript |");
        table.AppendLine("|---|---:|---:|---:|---:|---|---|---:|---:|---:|---|");

        if (models.Contains("styletts2", StringComparison.OrdinalIgnoreCase))
        {
            RunStyleTts2(backend, verify, reference24k, cmudict, outDir, refDir, table);
        }
        if (models.Contains("cosyvoice2", StringComparison.OrdinalIgnoreCase))
        {
            RunCosyVoice2(backend, verify, reference24k, outDir, refDir, table);
        }
        Emit(table);
    }

    /// <summary>StyleTTS 2 LibriTTS zero-shot clone of the first six seconds of the JFK clip, the three Kokoro
    /// bench sentences through <see cref="EnglishG2P"/> (the tokenizer drops symbols outside its set the same way
    /// on both arms).</summary>
    private void RunStyleTts2(IBackend backend, WhisperPipeline verify, float[] reference24k, string cmudict,
        string? outDir, string? refDir, StringBuilder table)
    {
        string checkpoint = Path.Combine(AudioModelCache.GetRepoDirectory(StyleTts2Repo, "tts"), "Models", "LibriTTS", "epochs_2nd_00020.pth");
        if (!RealWeightGate.Require(_out.WriteLine, checkpoint)) return;
        EnglishG2P g2p = new EnglishG2P(cmudict);
        Stopwatch load = Stopwatch.StartNew();
        using StyleTts2Pipeline pipeline = StyleTts2Pipeline.LoadFromCheckpoint(checkpoint);
        _out.WriteLine($"StyleTTS2 loaded in {load.Elapsed.TotalSeconds:F1}s");
        float[] clip = reference24k[..Math.Min(reference24k.Length, 6 * 24_000)];
        foreach ((int words, string text) in Sentences)
        {
            string ipa = g2p.ToIpa(text);
            pipeline.SynthesizeCloneFromAudio(backend, ipa, clip, 24_000, 1f);
            Stopwatch sw = Stopwatch.StartNew();
            float[] wave = pipeline.SynthesizeCloneFromAudio(backend, ipa, clip, 24_000, 1f);
            sw.Stop();
            Report(backend, verify, "styletts2", words, text, wave, 24_000, sw.Elapsed.TotalSeconds, outDir, refDir, table);
        }
    }

    /// <summary>CosyVoice 2 zero-shot clone from the full JFK clip with its transcript as the prompt text, built the
    /// way the engine's descriptor builds it (Qwen LM + flow + HiFTNet from the .pt files, S3 tokenizer + CAM++ from
    /// chatterbox's s3gen.safetensors), two sentences at a fixed seed.</summary>
    private void RunCosyVoice2(IBackend backend, WhisperPipeline verify, float[] reference24k, string? outDir, string? refDir,
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
            Report(backend, verify, "cosyvoice2", words, text, wave, config.SampleRate, sw.Elapsed.TotalSeconds, outDir, refDir, table);
        }
    }

    private void Report(IBackend backend, WhisperPipeline verify, string model, int words, string text, float[] wave, int rate,
        double seconds, string? outDir, string? refDir, StringBuilder table)
    {
        string name = $"{model}_{words}w";
        string digest = AudioParityMetrics.Sha256Hex(wave)[..12];
        if (!string.IsNullOrEmpty(outDir))
        {
            File.WriteAllBytes(Path.Combine(outDir, name + ".f32"), MemoryMarshal.AsBytes<float>(wave).ToArray());
            WavFile.WriteMono16(Path.Combine(outDir, name + ".wav"), wave, rate);
        }
        string len = "—", corr = "—", specCorr = "—";
        if (!string.IsNullOrEmpty(refDir) && File.Exists(Path.Combine(refDir, name + ".f32")))
        {
            float[] reference = MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(Path.Combine(refDir, name + ".f32"))).ToArray();
            len = reference.Length == wave.Length ? "identical" : $"{reference.Length}→{wave.Length}";
            corr = AudioParityMetrics.Compare(reference, wave).Corr.ToString("F6", CultureInfo.InvariantCulture);
            specCorr = AudioParityMetrics.LogSpectralCorrelation(reference, wave).ToString("F6", CultureInfo.InvariantCulture);
        }
        float[] forWhisper = rate == WhisperRate ? wave : Resampler.Create(rate, WhisperRate).Resample(wave);
        string heard = verify.TranscribeAudio(backend, forWhisper, WhisperRate, new WhisperOptions { Language = "en" });
        double recall = AudioParityMetrics.ContentWordRecall(text, heard);
        double audioSeconds = wave.Length / (double)rate;
        table.AppendLine($"| {model} | {words} | {AudioParityMetrics.Ms(seconds)} | {audioSeconds:F2} | {wave.Length} | `{digest}` | {len} | "
            + $"{corr} | {specCorr} | {recall:P0} | {AudioParityMetrics.Cell(heard)} |");
        _out.WriteLine($"{model} {words}w: {AudioParityMetrics.Ms(seconds)} ms | {audioSeconds:F2}s ({wave.Length} samples) | sha {digest} | "
            + $"len {len} wave corr {corr} log-spec corr {specCorr} | recall {recall:P0} | {heard.Trim()}");
    }

    private IBackend OpenBackend()
    {
        string? ordinalText = Environment.GetEnvironmentVariable(OrdinalEnvVar);
        int ordinal = string.IsNullOrEmpty(ordinalText) ? 1 : int.Parse(ordinalText, CultureInfo.InvariantCulture);
        Assert.True(CudaContext.IsAvailable(), $"CUDA unavailable: {CudaContext.LastUnavailableReason}");
        string? ptx = BackendGate.KernelDir("Ptx", "HartsyInference.Cuda");
        Assert.False(ptx is null, "no compiled PTX directory beside the tests or in the repo");
        CudaBackend backend = new CudaBackend(ordinal, ptx);
        string device = backend.Capabilities.DeviceName;
        _out.WriteLine($"CUDA ordinal {ordinal}: {device}; audio cache {AudioModelCache.CacheRoot}");
        if (!device.Contains(RequiredDeviceSubstring, StringComparison.Ordinal))
        {
            backend.Dispose();
            Assert.Fail($"ordinal {ordinal} is '{device}', not a {RequiredDeviceSubstring}. Set {OrdinalEnvVar} to the 3060's engine ordinal.");
        }
        return backend;
    }

    private static string ModelsRoot() =>
        EngineKnobs.ModelsRoot.Value is { Length: > 0 } root ? Path.GetFullPath(root) : TestPaths.ModelsDir;

    private static string[] WhisperFiles(string repo)
    {
        string dir = AudioModelCache.GetRepoDirectory(repo, "stt");
        return WhisperPipeline.ModelFiles.Where(f => f.Required).Select(f => Path.Combine(dir, f.Name)).ToArray();
    }

    private void Emit(StringBuilder table)
    {
        string text = table.ToString();
        _out.WriteLine(text);
        string? outPath = Environment.GetEnvironmentVariable(OutEnvVar);
        if (!string.IsNullOrEmpty(outPath))
        {
            File.AppendAllText(outPath, text + Environment.NewLine);
        }
    }
}
