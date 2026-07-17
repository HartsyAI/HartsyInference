using System;
using System.IO;
using HartsyInference.Audio.Io;
using HartsyInference.Audio.Models.CosyVoice;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Backends;
using HartsyInference.Cpu;
using HartsyInference.ModelHandler.PyTorch;
using HartsyInference.Tokenizers;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Full end-to-end <see cref="CosyVoicePipeline"/> generation on the real FunAudioLLM/CosyVoice2-0.5B
/// checkpoint (Qwen speech-token LM → OT-CFM flow → HiFTNet vocoder, zero-shot from a reference clip), writing a
/// 24 kHz WAV for external whisper verification. Gated on the model dir (<c>COSY_DIR</c> — holds llm.pt/flow.pt/
/// hift.pt) plus chatterbox s3gen.safetensors for the frozen S3/CAM++ encoders (auto-located beside COSY_DIR or via
/// <c>COSY_S3GEN</c>), a reference clip + its transcript (<c>COSY_REF_WAV</c> / <c>COSY_REF_TEXT</c>) and an output
/// path (<c>COSY_OUT_WAV</c>). Runs on CUDA when <c>COSY_CUDA=1</c> + <c>COSY_PTX</c> (AR LM too slow on CPU).</summary>
public sealed class CosyVoiceE2eTests
{
    private readonly ITestOutputHelper _out;
    public CosyVoiceE2eTests(ITestOutputHelper o) => _out = o;

    [Fact]
    [Trait("Category", "GpuIntegration")]
    public void Synthesize_WritesWav()
    {
        string? dir = Environment.GetEnvironmentVariable("COSY_DIR");
        string? refWav = Environment.GetEnvironmentVariable("COSY_REF_WAV");
        string? outWav = Environment.GetEnvironmentVariable("COSY_OUT_WAV");
        if (string.IsNullOrEmpty(dir) || !Directory.Exists(dir) || string.IsNullOrEmpty(refWav)
            || !File.Exists(refWav) || string.IsNullOrEmpty(outWav))
        {
            _out.WriteLine("Skipped: set COSY_DIR + COSY_REF_WAV + COSY_OUT_WAV (optional COSY_REF_TEXT/COSY_TEXT/COSY_CUDA/COSY_PTX/COSY_SEED).");
            return;
        }

        string llmPath = Path.Combine(dir!, "llm.pt");
        string flowPath = Path.Combine(dir!, "flow.pt");
        string hiftPath = Path.Combine(dir!, "hift.pt");
        // The two frozen input encoders (S3 speech tokenizer + CAM++ speaker) are shipped by CosyVoice2 only as
        // ONNX, whose export fuses Conv+BN (CAM++) and mangles names (S3). Both are the same frozen pretrained models
        // ResembleAI/chatterbox ships in clean PyTorch form inside s3gen.safetensors (tokenizer.* / speaker_encoder.*).
        string s3genPath = Environment.GetEnvironmentVariable("COSY_S3GEN")
            ?? Path.Combine(Path.GetDirectoryName(dir!.TrimEnd('/'))!, "ResembleAI--chatterbox", "s3gen.safetensors");

        PytorchPickleLoader llmL = new(); llmL.Load(llmPath);
        PytorchPickleLoader flowL = new(); flowL.Load(flowPath);
        PytorchPickleLoader hiftL = new(); hiftL.Load(hiftPath);
        HartsyInference.ModelHandler.SafeTensors.SafeTensorsLoader s3genL = new(); s3genL.Load(s3genPath);

        CosyVoiceConfig cfg = CosyVoiceConfig.V2_0_5B;
        CosyVoiceQwenLm lm = new(cfg); lm.LoadWeights(llmL.GetAllTensors());
        CosyVoiceFlow flow = new(cfg); flow.LoadWeights(flowL.GetAllTensors());
        HiFTNetVocoder vocoder = new(cfg.Hift); vocoder.LoadWeights(hiftL.GetAllTensors());
        CamPlusSpeakerEncoder speaker = new(192); speaker.LoadWeights(s3genL.GetAllTensors(), "speaker_encoder");
        S3Tokenizer s3 = new(); s3.LoadWeights(s3genL.GetAllTensors(), "tokenizer");
        using CosyVoicePipeline pipeline = new(cfg, lm, flow, vocoder, speaker, s3);
        _out.WriteLine("Loaded CosyVoice2-0.5B (LM + flow + HiFTNet + CAM++ + S3).");

        IBackend backend;
        if (Environment.GetEnvironmentVariable("COSY_CUDA") == "1")
        {
            string ptx = Environment.GetEnvironmentVariable("COSY_PTX")
                ?? throw new InvalidOperationException("COSY_CUDA=1 requires COSY_PTX.");
            backend = new HartsyInference.Cuda.CudaBackend(0, ptx);
            _out.WriteLine($"Backend: CUDA (ptx={ptx}).");
        }
        else { backend = new CpuBackend(); _out.WriteLine("Backend: CPU."); }

        WavFile.DecodedAudio da = WavFile.Read(refWav!);
        float[] mono = da.ToMono();
        float[] ref24k = da.SampleRate == 24000 ? mono : Resampler.Create(da.SampleRate, 24000).Resample(mono);
        _out.WriteLine($"Reference: {refWav} ({da.SampleRate} Hz → 24k, {ref24k.Length} samp, {ref24k.Length / 24000.0:F2}s).");

        Qwen2Tokenizer tok = new();
        string text = Environment.GetEnvironmentVariable("COSY_TEXT")
            ?? "Hello, this is a test of the Cosy Voice two text to speech system running natively.";
        string refText = Environment.GetEnvironmentVariable("COSY_REF_TEXT") ?? "";
        int[] textIds = [.. tok.EncodeRawByteLevel(text)];
        int[] refIds = string.IsNullOrWhiteSpace(refText) ? [] : [.. tok.EncodeRawByteLevel(refText)];
        int seed = int.TryParse(Environment.GetEnvironmentVariable("COSY_SEED"), out int sd) ? sd : 12345;
        _out.WriteLine($"Text ids: {textIds.Length}, ref-text ids: {refIds.Length}, seed {seed}.");

        long t0 = Environment.TickCount64;
        float[] audio = pipeline.Synthesize(backend, textIds, referenceAudio: ref24k, referenceSampleRate: 24000,
            referenceTextTokens: refIds, seed: seed);
        long ms = Environment.TickCount64 - t0;

        Assert.NotEmpty(audio);
        double seconds = audio.Length / (double)cfg.SampleRate;
        _out.WriteLine($"Generated {audio.Length} samples ({seconds:F2}s) in {ms} ms (RTF {ms / 1000.0 / seconds:F2}).");
        WavFile.WriteMono16(outWav!, audio, cfg.SampleRate);
        _out.WriteLine($"Wrote {outWav}");
        (backend as IDisposable)?.Dispose();
    }
}
