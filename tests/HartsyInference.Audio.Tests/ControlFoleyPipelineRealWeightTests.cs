using HartsyInference.Audio.Models.ControlFoley;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelAssets.PyTorch;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Text-to-audio with the released ControlFoley weights. Needs <c>HARTSY_CONTROLFOLEY_E2E_DIR</c> holding
/// <c>controlfoley_bf16.safetensors</c> (or <c>controlfoley.pth</c>), <c>audio_decoder.safetensors</c> (VAE plus vocoder, or
/// <c>v1-44.pth</c> and <c>bigvgan_generator.pt</c>) and <c>open_clip_pytorch_model.bin</c>; without it the test does nothing.
/// <c>HARTSY_CONTROLFOLEY_PROMPT</c>, <c>_STEPS</c>, <c>_DURATION</c> and <c>_OUT</c> (a wav path) tune the run.</summary>
public sealed class ControlFoleyPipelineRealWeightTests(ITestOutputHelper output)
{
    [Fact]
    public void TextToAudio_ProducesFiniteNonSilentAudio()
    {
        string? dir = Environment.GetEnvironmentVariable("HARTSY_CONTROLFOLEY_E2E_DIR");
        if (string.IsNullOrEmpty(dir))
        {
            return;
        }

        string prompt = Environment.GetEnvironmentVariable("HARTSY_CONTROLFOLEY_PROMPT") ?? "rain falling on a tin roof";
        int steps = int.Parse(Environment.GetEnvironmentVariable("HARTSY_CONTROLFOLEY_STEPS") ?? "10");
        double duration = double.Parse(Environment.GetEnvironmentVariable("HARTSY_CONTROLFOLEY_DURATION") ?? "4");
        List<IDisposable> owned = [];
        try
        {
            string bf16 = Path.Combine(dir, "controlfoley_bf16.safetensors");
            IReadOnlyDictionary<string, Tensor> networkWeights = Open(File.Exists(bf16) ? bf16 : Path.Combine(dir, "controlfoley.pth"), owned);
            ControlFoleyNetwork network = new(ControlFoleyNetworkConfig.Large44k);
            owned.Add(network);
            network.LoadWeights(networkWeights);
            ControlFoleyClip clip = new(ControlFoleyClipConfig.Dfn5bViTH14);
            clip.LoadWeights(Open(Path.Combine(dir, "open_clip_pytorch_model.bin"), owned));
            ControlFoleyAudioDecoder decoder = new();
            owned.Add(decoder);
            string combined = Path.Combine(dir, "audio_decoder.safetensors");
            if (File.Exists(combined))
            {
                decoder.LoadFromCombinedFile(combined);
            }
            else
            {
                decoder.LoadFromFiles(Path.Combine(dir, "v1-44.pth"), Path.Combine(dir, "bigvgan_generator.pt"));
            }

            using CpuBackend backend = new();
            ControlFoleyPipeline pipeline = new(network, clip, decoder);
            float[] audio = pipeline.Generate(backend, new ControlFoleyPipeline.Request
            {
                Prompt = prompt, DurationSeconds = duration, Steps = steps, Seed = 7,
            });

            double energy = 0;
            foreach (float s in audio)
            {
                Assert.True(float.IsFinite(s));
                energy += (double)s * s;
            }

            double rms = Math.Sqrt(energy / audio.Length);
            output.WriteLine($"{audio.Length} samples ({audio.Length / (double)pipeline.SampleRate:0.00} s), rms {rms:0.0000}, peak {audio.Max(MathF.Abs):0.000}");
            Assert.True(audio.Length >= (int)(duration * pipeline.SampleRate) - pipeline.SampleRate / 10);
            Assert.True(rms > 1e-3, $"output is silent (rms {rms})");
            string? outPath = Environment.GetEnvironmentVariable("HARTSY_CONTROLFOLEY_OUT");
            if (!string.IsNullOrEmpty(outPath))
            {
                WriteWav(outPath, audio, pipeline.SampleRate);
            }
        }
        finally
        {
            foreach (IDisposable d in owned)
            {
                d.Dispose();
            }
        }
    }

    private static IReadOnlyDictionary<string, Tensor> Open(string path, List<IDisposable> owned)
    {
        if (path.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase))
        {
            SafeTensorsLoader loader = new();
            loader.Load(path);
            owned.Add(loader);
            return loader.GetAllTensors();
        }

        AnyFormatCheckpointLoader pickle = new();
        pickle.Load(path);
        owned.Add(pickle);
        return pickle.GetAllTensors();
    }

    private static void WriteWav(string path, float[] samples, int rate)
    {
        using BinaryWriter w = new(File.Create(path));
        int bytes = samples.Length * 2;
        w.Write("RIFF"u8); w.Write(36 + bytes); w.Write("WAVEfmt "u8); w.Write(16); w.Write((short)1); w.Write((short)1);
        w.Write(rate); w.Write(rate * 2); w.Write((short)2); w.Write((short)16); w.Write("data"u8); w.Write(bytes);
        foreach (float s in samples)
        {
            w.Write((short)Math.Clamp(s * 32767f, -32768f, 32767f));
        }
    }
}
