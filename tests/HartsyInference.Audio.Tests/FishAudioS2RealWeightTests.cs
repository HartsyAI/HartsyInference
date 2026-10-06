using System.Text.Json;
using HartsyInference.Audio.Models.FishAudio;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Real-weight checks for Fish Audio S2 Pro. They need <c>HARTSY_FISH_S2_DIR</c> to name a folder with the
/// released <c>tokenizer.json</c>, <c>model-0000N-of-00002.safetensors</c> + index, and <c>codec.safetensors</c> (the
/// released <c>codec.pth</c> converted without its rope/mask buffers); without it they do nothing.</summary>
[Trait("Category", "RealWeights")]
public sealed class FishAudioS2RealWeightTests(ITestOutputHelper output)
{
    private static string? Dir => Environment.GetEnvironmentVariable("HARTSY_FISH_S2_DIR");

    private static GgufTokenizer LoadTokenizer(string dir)
    {
        using FileStream json = File.OpenRead(Path.Combine(dir, "tokenizer.json"));
        return HfTokenizerJson.LoadByteLevelBpe(json);
    }

    [Fact]
    public void Tokenizer_MatchesHuggingFaceTokenizers()
    {
        if (Dir is not { Length: > 0 } dir) return;
        GgufTokenizer tokenizer = LoadTokenizer(dir);
        using JsonDocument reference = JsonDocument.Parse(File.ReadAllText(Path.Combine(dir, "ref_ids.json")));
        foreach (JsonElement item in reference.RootElement.EnumerateArray())
        {
            string text = item.GetProperty("text").GetString()!;
            int[] expected = item.GetProperty("ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();
            Assert.Equal(expected, tokenizer.Encode(text.Normalize(System.Text.NormalizationForm.FormC), addSpecial: true));
        }
    }

    [Fact]
    public void Generate_ProducesSpeechAudio()
    {
        if (Dir is not { Length: > 0 } dir) return;
        GgufTokenizer tokenizer = LoadTokenizer(dir);

        Dictionary<string, Tensor> model = new();
        List<SafeTensorsLoader> loaders = [];
        foreach (string shard in Directory.GetFiles(dir, "model-*-of-*.safetensors"))
        {
            SafeTensorsLoader l = new(); l.Load(shard); loaders.Add(l);
            foreach (string name in l.Descriptors.Keys) model[name] = l.GetTensor(name);
        }
        using SafeTensorsLoader codecLoader = new();
        codecLoader.Load(Path.Combine(dir, "codec.safetensors"));
        Dictionary<string, Tensor> codec = new();
        foreach (string name in codecLoader.Descriptors.Keys) codec[name] = codecLoader.GetTensor(name);

        using IBackend backend = new CpuBackend();
        using FishAudioS2Pipeline pipeline = new(FishAudioS2Config.S2Pro, ModifiedDacConfig.S2,
            text => tokenizer.Encode(text, addSpecial: true));
        pipeline.LoadWeights(model, codec);

        string text = Environment.GetEnvironmentVariable("HARTSY_FISH_S2_TEXT") ?? "Hello, this is a test of the Fish Audio speech model.";
        System.Diagnostics.Stopwatch watch = System.Diagnostics.Stopwatch.StartNew();
        int[,]? referenceCodes = null;
        string? referenceText = Environment.GetEnvironmentVariable("HARTSY_FISH_S2_REF_TEXT");
        if (Environment.GetEnvironmentVariable("HARTSY_FISH_S2_REF_WAV") is { Length: > 0 } refWav)
            referenceCodes = pipeline.EncodeReference(backend, ReadWav(refWav));
        int maxFrames = int.TryParse(Environment.GetEnvironmentVariable("HARTSY_FISH_S2_MAX_FRAMES"), out int m) ? m : 120;
        float[] audio = pipeline.Synthesize(backend, new FishAudioS2Pipeline.Request
        {
            Text = text, Seed = 7, MaxFrames = maxFrames, ReferenceCodes = referenceCodes, ReferenceText = referenceText,
        });
        output.WriteLine($"{audio.Length} samples ({audio.Length / (double)pipeline.SampleRate:F2} s) in {watch.Elapsed.TotalSeconds:F1} s");
        string? outPath = Environment.GetEnvironmentVariable("HARTSY_FISH_S2_OUT");
        if (outPath is not null) WriteWav(outPath, audio, pipeline.SampleRate);
        Assert.NotEmpty(audio);
        foreach (SafeTensorsLoader l in loaders) l.Dispose();
    }

    private static float[] ReadWav(string path)
    {
        byte[] bytes = File.ReadAllBytes(path);
        int data = 44;   // canonical 44-byte header (the writer below, or any plain 16-bit mono PCM WAV)
        float[] samples = new float[(bytes.Length - data) / 2];
        for (int i = 0; i < samples.Length; i++) samples[i] = BitConverter.ToInt16(bytes, data + 2 * i) / 32768f;
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
