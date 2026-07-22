using Xunit;
using Xunit.Abstractions;
using HartsyInference.Audio.Models.Codecs.Oobleck;
using HartsyInference.Core.Models;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Tests.Common;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Diffusion.Tests;

/// <summary>End-to-end real-weight generation through <see cref="StableAudioPipeline"/>: T5-base prompt encode →
/// <see cref="StableAudioNumberEmbedder"/> timing token → <see cref="StableAudioDit"/> pingpong denoise →
/// <see cref="OobleckVae"/> decode → stereo WAV. The individual components are separately real-weight-verified
/// bit-exact (StableAudioDitParityTests / OobleckVaeParityTests / StableAudioNumberEmbedderParityTests); this
/// proves the plumbing runs end-to-end and produces finite, non-silent audio (no Python reference for the full
/// composed pipeline — sanity-checked the same way the audio-verify campaign checks other models: finite
/// samples + RMS envelope). Runs only 2 steps (not the distilled model's real 8) — this test targets plumbing,
/// not audio quality; the DiT always denoises the full 256-token trained latent regardless of step count, and
/// on this engine's naive (non-BLAS) CPU backend that is ~90-100B MACs per forward call, so even 2 steps takes
/// several minutes — <c>[Trait("Category","Slow")]</c> like <c>AceStepGenerationTests</c>. Skip-guarded when the
/// checkpoint or T5-base weights are absent.</summary>
[Trait("Category", "Integration")]
[Trait("Category", "Slow")]
public unsafe class StableAudioGenerationTests
{
    private readonly ITestOutputHelper _output;
    public StableAudioGenerationTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void Generate_ShortSfx_WritesWav()
    {
        string root = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".cache", "hartsyinference", "models", "stable-audio-open-small");
        if (!Directory.Exists(root)) { _output.WriteLine($"SKIPPED: Stable Audio checkpoint not found at {root}."); return; }

        string? t5Path = FindT5BaseSafetensors();
        if (t5Path is null) { _output.WriteLine("SKIPPED: google-t5/t5-base model.safetensors not found in the HF cache."); return; }

        using CpuBackend backend = new();
        StableAudioDitConfig cfg = StableAudioDitConfig.OpenSmall;

        T5Tokenizer t5Tokenizer = new(maxLength: 64);
        T5TextEncoder t5 = new(T5TextEncoderConfig.T5Base);
        using SafeTensorsLoader t5Loader = new();
        t5Loader.Load(t5Path);
        t5.LoadWeights(t5Loader.GetAllTensors());
        int[] promptIds = t5Tokenizer.Encode("a dog barking");
        Tensor textEmbeds = t5.Encode(backend, [promptIds]);
        _output.WriteLine($"T5: [{textEmbeds.Shape[0]}, {textEmbeds.Shape[1]}, {textEmbeds.Shape[2]}]");

        using SafeTensorsLoader ditLoader = new();
        ditLoader.Load(Path.Combine(root, "transformer", "diffusion_pytorch_model.safetensors"));
        StableAudioDit dit = new(cfg);
        dit.LoadWeights(ditLoader.GetAllTensors());

        using SafeTensorsLoader vaeLoader = new();
        vaeLoader.Load(Path.Combine(root, "vae", "diffusion_pytorch_model.safetensors"));
        OobleckConfig vaeCfg = OobleckConfig.StableAudioOpen;
        Dictionary<string, Tensor> vaeWeights = OobleckKeyRemap.ToFlatSequentialLayout(vaeLoader.GetAllTensors(), vaeCfg);
        OobleckVae vae = new(vaeCfg);
        vae.LoadWeights(vaeWeights);

        using SafeTensorsLoader condLoader = new();
        condLoader.Load(Path.Combine(root, "conditioner", "diffusion_pytorch_model.safetensors"));
        StableAudioNumberEmbedder timing = new(minVal: 0f, maxVal: (float)cfg.TimingMaxSeconds);
        timing.LoadWeights(condLoader.GetAllTensors(), "conditioners.seconds_total");

        StableAudioPipeline pipeline = new(backend, dit, (IAudioLatentDecoder)vae, timing, cfg);
        (float[] left, float[] right, int rate, int seed) = pipeline.Generate(
            textEmbeds, secondsTotal: 3.0, steps: 2, seed: 0,
            onProgress: p => _output.WriteLine($"step {p.Step}/{p.TotalSteps} ({p.ElapsedMs:F0}ms)"));

        Directory.CreateDirectory(TestPaths.OutputDir);
        string wavPath = Path.Combine(TestPaths.OutputDir, $"stable_audio_seed{seed}.wav");
        WriteWavStereo(wavPath, left, right, rate);

        double rms = 0;
        foreach (float a in left) rms += (double)a * a;
        rms = Math.Sqrt(rms / left.Length);
        _output.WriteLine($"Wrote {wavPath} ({left.Length / (double)rate:F2}s, rms={rms:F4})");

        Assert.All(left, a => Assert.True(float.IsFinite(a)));
        Assert.All(right, a => Assert.True(float.IsFinite(a)));
        Assert.True(rms > 1e-4, "output is silent");
        Assert.True(left.Length > rate * 2);
        Assert.Equal(44_100, rate);
    }

    private static string? FindT5BaseSafetensors()
    {
        string hub = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".cache", "huggingface", "hub", "models--google-t5--t5-base");
        if (!Directory.Exists(hub)) return null;
        return Directory.GetFiles(hub, "model.safetensors", SearchOption.AllDirectories).FirstOrDefault();
    }

    private static void WriteWavStereo(string path, float[] left, float[] right, int sampleRate)
    {
        int samples = left.Length;
        using BinaryWriter wr = new(File.Create(path));
        int dataBytes = samples * 2 * 2;
        wr.Write("RIFF"u8); wr.Write(36 + dataBytes); wr.Write("WAVE"u8);
        wr.Write("fmt "u8); wr.Write(16); wr.Write((short)1); wr.Write((short)2);
        wr.Write(sampleRate); wr.Write(sampleRate * 4); wr.Write((short)4); wr.Write((short)16);
        wr.Write("data"u8); wr.Write(dataBytes);
        for (int i = 0; i < samples; i++)
        {
            wr.Write((short)Math.Clamp(left[i] * 32767f, short.MinValue, short.MaxValue));
            wr.Write((short)Math.Clamp(right[i] * 32767f, short.MinValue, short.MaxValue));
        }
    }
}
