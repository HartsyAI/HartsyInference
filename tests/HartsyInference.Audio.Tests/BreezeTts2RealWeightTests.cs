using System.Text.Json;
using HartsyInference.Audio.Models.BreezeTts;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;
using Xunit.Abstractions;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Real-weight checks for Breeze TTS 2; they need <c>HARTSY_BREEZE_DIR</c> to name the downloaded
/// <c>BreezeBlue/Breeze-TTS-2</c> folder (plus <c>ref_ids.json</c> from HF <c>tokenizers</c>) and otherwise do nothing.</summary>
[Trait("Category", "RealWeights")]
public sealed class BreezeTts2RealWeightTests(ITestOutputHelper output)
{
    private static string? Dir => Environment.GetEnvironmentVariable("HARTSY_BREEZE_DIR");

    [Fact]
    public void Tokenizer_MatchesHuggingFaceTokenizers()
    {
        if (Dir is not { Length: > 0 } dir) return;
        using FileStream json = File.OpenRead(Path.Combine(dir, "tokenizer.json"));
        SentencePieceBpeJson tokenizer = new(json);
        using JsonDocument reference = JsonDocument.Parse(File.ReadAllText(Path.Combine(dir, "ref_ids.json")));
        foreach (JsonElement item in reference.RootElement.EnumerateArray())
        {
            int[] expected = item.GetProperty("ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();
            Assert.Equal(expected, tokenizer.Encode(item.GetProperty("text").GetString()!));
        }
    }

    [Fact]
    public void Generate_ProducesSpeechAudio()
    {
        if (Dir is not { Length: > 0 } dir) return;
        using FileStream json = File.OpenRead(Path.Combine(dir, "tokenizer.json"));
        SentencePieceBpeJson tokenizer = new(json);

        Dictionary<string, Tensor> model = new();
        List<SafeTensorsLoader> loaders = [];
        foreach (string shard in Directory.GetFiles(dir, "model-*-of-*.safetensors"))
        {
            SafeTensorsLoader l = new(); l.Load(shard); loaders.Add(l);
            foreach (string name in l.Descriptors.Keys) model[name] = l.GetTensor(name);
        }
        using SafeTensorsLoader tokLoader = new();
        tokLoader.Load(Path.Combine(dir, "audio_tokenizer", "model.safetensors"));
        Dictionary<string, Tensor> audioTokenizer = new();
        foreach (string name in tokLoader.Descriptors.Keys) audioTokenizer[name] = tokLoader.GetTensor(name);

        using IBackend backend = new CpuBackend();
        using BreezeTts2Pipeline pipeline = new(BreezeTts2Config.Default, tokenizer);
        pipeline.LoadWeights(model, audioTokenizer);

        string text = Environment.GetEnvironmentVariable("HARTSY_BREEZE_TEXT") ?? "Hello, this is a test of the Breeze speech model.";
        string? instruction = Environment.GetEnvironmentVariable("HARTSY_BREEZE_INSTRUCTION");
        float cfg = float.TryParse(Environment.GetEnvironmentVariable("HARTSY_BREEZE_CFG"), out float c) ? c : 1f;
        int maxFrames = int.TryParse(Environment.GetEnvironmentVariable("HARTSY_BREEZE_MAX_FRAMES"), out int m) ? m : 60;
        int[][]? refFrames = null;
        string? refText = Environment.GetEnvironmentVariable("HARTSY_BREEZE_REF_TEXT");
        if (Environment.GetEnvironmentVariable("HARTSY_BREEZE_REF_WAV") is { Length: > 0 } refWav)
            refFrames = pipeline.EncodeReference(backend, ReadWav(refWav));

        System.Diagnostics.Stopwatch watch = System.Diagnostics.Stopwatch.StartNew();
        float[] audio = pipeline.Synthesize(backend, new BreezeTts2Pipeline.Request
        {
            Text = text, Instruction = instruction, CfgScale = cfg, MaxFrames = maxFrames, Seed = 7,
            ReferenceFrames = refFrames, ReferenceText = refText,
        });
        output.WriteLine($"{audio.Length} samples ({audio.Length / (double)pipeline.SampleRate:F2} s) in {watch.Elapsed.TotalSeconds:F1} s");
        if (Environment.GetEnvironmentVariable("HARTSY_BREEZE_OUT") is { Length: > 0 } outPath) WriteWav(outPath, audio, pipeline.SampleRate);
        Assert.NotEmpty(audio);
        foreach (SafeTensorsLoader l in loaders) l.Dispose();
    }

    private static float[] ReadWav(string path)
    {
        byte[] bytes = File.ReadAllBytes(path);
        float[] samples = new float[(bytes.Length - 44) / 2];
        for (int i = 0; i < samples.Length; i++) samples[i] = BitConverter.ToInt16(bytes, 44 + 2 * i) / 32768f;
        return samples;
    }

    private static void WriteWav(string path, float[] samples, int rate)
    {
        using BinaryWriter w = new(File.Create(path));
        int bytes = samples.Length * 2;
        w.Write("RIFF"u8); w.Write(36 + bytes); w.Write("WAVEfmt "u8); w.Write(16); w.Write((short)1); w.Write((short)1);
        w.Write(rate); w.Write(rate * 2); w.Write((short)2); w.Write((short)16); w.Write("data"u8); w.Write(bytes);
        foreach (float s in samples) w.Write((short)Math.Clamp(s * 32767f, -32768f, 32767f));
    }
}
