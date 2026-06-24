using System;
using System.Collections.Generic;
using System.IO;
using HartsyInference.Audio.Io;
using HartsyInference.Audio.Models.QwenTts;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Cpu;
using HartsyInference.Cuda;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.Tokenizers;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>End-to-end Qwen3-TTS text -> 24 kHz WAV on real weights. Loads the talker (<c>model.safetensors</c>,
/// which carries <c>talker.*</c> + <c>speaker_encoder.*</c>) and the codec (<c>speech_tokenizer/model.safetensors</c>,
/// the <c>decoder.*</c> tree), tokenizes text with the embedded Qwen BPE, runs the dual-stream prefill + free AR,
/// and writes a WAV. Verifies the output is finite and non-silent.
///
/// <para>Gated on <c>QWEN3TTS_TALKER_PATH</c> + <c>QWEN3TTS_CODEC_PATH</c> (downloaded once from
/// <c>Qwen/Qwen3-TTS-12Hz-1.7B-Base</c>). Optional: <c>QWEN3TTS_SPEAKER</c> (codec token, e.g. 3061 Ryan, only on
/// the CustomVoice checkpoint; default -1 = voice-design), <c>QWEN3TTS_LANG</c> (default 2050 English),
/// <c>QWEN3TTS_TEXT</c>, <c>QWEN3TTS_MAXFRAMES</c> (default 60 ~= 4.8 s), <c>QWEN3TTS_OUT_WAV</c>. Audio-quality
/// parity is verified by comparing this WAV to the pure-C reference (gabriele-mastrapasqua/qwen3-tts).</para></summary>
public sealed class Qwen3TtsEndToEndTests
{
    private readonly ITestOutputHelper _out;
    public Qwen3TtsEndToEndTests(ITestOutputHelper o) => _out = o;

    [Fact]
    [Trait("Category", "Integration")]
    public void Text_To_Wav_OnRealWeights()
    {
        string? talkerPath = Environment.GetEnvironmentVariable("QWEN3TTS_TALKER_PATH");
        string? codecPath = Environment.GetEnvironmentVariable("QWEN3TTS_CODEC_PATH");
        if (string.IsNullOrEmpty(talkerPath) || !File.Exists(talkerPath)
            || string.IsNullOrEmpty(codecPath) || !File.Exists(codecPath))
        {
            return; // gated
        }

        int speaker = ParseEnv("QWEN3TTS_SPEAKER", -1);
        int lang = ParseEnv("QWEN3TTS_LANG", 2050);          // English
        int maxFrames = ParseEnv("QWEN3TTS_MAXFRAMES", 60);
        string text = Environment.GetEnvironmentVariable("QWEN3TTS_TEXT") ?? "Hello, this is a test of Qwen 3 text to speech.";
        string outWav = Environment.GetEnvironmentVariable("QWEN3TTS_OUT_WAV") ?? Path.Combine(Path.GetTempPath(), "qwen3tts_out.wav");

        SafeTensorsLoader talkerLoader = new();
        talkerLoader.Load(talkerPath);
        IReadOnlyDictionary<string, Tensor> talker = talkerLoader.GetAllTensors();
        SafeTensorsLoader codecLoader = new();
        codecLoader.Load(codecPath);
        IReadOnlyDictionary<string, Tensor> codec = codecLoader.GetAllTensors();

        Qwen3TtsConfig cfg = Qwen3TtsConfig.Default_1_7B with { MaxNewTokens = maxFrames };
        if (Environment.GetEnvironmentVariable("QWEN3TTS_GREEDY") == "1")
        {
            cfg = cfg with { TopK = 1, RepetitionPenalty = 1f };   // deterministic, for parity vs the reference
        }
        using Qwen3TtsPipeline pipe = new(cfg);

        // Backend: CUDA when QWEN3TTS_CUDA=1 (BF16 weights load directly, ~100x faster), else CPU (F32-only, so
        // the BF16 talker layer weights are cast up front).
        bool useCuda = Environment.GetEnvironmentVariable("QWEN3TTS_CUDA") == "1";
        IBackend backend;
        IReadOnlyDictionary<string, Tensor> talkerW = talker;
        if (useCuda)
        {
            string ptx = Environment.GetEnvironmentVariable("QWEN3TTS_PTX")
                ?? throw new InvalidOperationException("QWEN3TTS_CUDA=1 requires QWEN3TTS_PTX (path to the Ptx kernel dir).");
            backend = new CudaBackend(0, ptx);
            _out.WriteLine($"Backend: CUDA (ptx={ptx}).");
        }
        else
        {
            Dictionary<string, Tensor> talkerF32 = new(talker.Count);
            foreach (KeyValuePair<string, Tensor> kv in talker) talkerF32[kv.Key] = WhisperOps.EnsureF32(kv.Value);
            talkerW = talkerF32;
            backend = new CpuBackend();
            _out.WriteLine("Backend: CPU (F32 cast).");
        }
        // The talker dict serves the talker, the MTP, and the ECAPA (each reads its own prefix); the codec dict
        // serves the vocoder. refCodec/ecapa not needed for custom_voice / voice_design.
        pipe.LoadWeights(talkerW, talkerW, codec);
        _out.WriteLine($"Loaded talker ({talkerPath}) + codec ({codecPath}).");

        Qwen3Tokenizer tok = new();
        int[] textIds = new List<int>(tok.EncodeRaw(text)).ToArray();
        _out.WriteLine($"Text: \"{text}\" -> {textIds.Length} BPE tokens.");

        long t0 = Environment.TickCount64;
        float[] pcm;
        try
        {
            pcm = speaker >= 0
                ? pipe.SynthesizeCustomVoice(backend, textIds, speaker, lang)
                : pipe.SynthesizeVoiceDesign(backend, textIds, lang);
        }
        finally
        {
            (backend as IDisposable)?.Dispose();
        }
        double secs = (Environment.TickCount64 - t0) / 1000.0;

        Assert.NotEmpty(pcm);
        double sumSq = 0; float peak = 0;
        foreach (float v in pcm) { Assert.True(float.IsFinite(v)); sumSq += (double)v * v; peak = Math.Max(peak, Math.Abs(v)); }
        double rms = Math.Sqrt(sumSq / pcm.Length);
        WavFile.WriteMono16(outWav, pcm, cfg.Vocoder.SampleRate);
        _out.WriteLine($"Generated {pcm.Length} samples ({pcm.Length / (double)cfg.Vocoder.SampleRate:0.00}s) in {secs:0.0}s. RMS={rms:0.0000} peak={peak:0.0000}. WAV: {outWav}");
        Assert.True(rms > 1e-4, "output should not be silent");
    }

    private static int ParseEnv(string name, int fallback)
        => int.TryParse(Environment.GetEnvironmentVariable(name), out int v) ? v : fallback;
}
