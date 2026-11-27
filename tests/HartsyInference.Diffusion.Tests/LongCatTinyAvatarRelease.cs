using HartsyInference.Audio.Io;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Engine.Recipes.Video;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Diffusion.Tests;

/// <summary>A tiny LongCat-Video-Avatar release on disk in the release layout (the DiT folder, with the 1.5 LoRA in <c>lora/</c> or the 1.6 pair in
/// <c>loraA/</c> and <c>loraB/</c> beside it), with the real audio window, plus the request inputs a run of it takes.</summary>
internal sealed class LongCatTinyAvatarRelease : IDisposable
{
    private const int SampleRate = 16_000;

    /// <summary>Writes the release under a new temporary folder; <paramref name="onePointSix"/> writes the INT8 DiT and the two LoRAs.</summary>
    public LongCatTinyAvatarRelease(bool onePointSix, bool? int8 = null, Dictionary<string, Tensor>? dense = null,
        Action<string>? writeDit = null)
    {
        bool useInt8 = int8 ?? onePointSix;
        Root = Path.Combine(Path.GetTempPath(), "longcat-avatar-" + Guid.NewGuid().ToString("N"));
        Checkpoint = Path.Combine(Root, useInt8 ? "base_model_int8" : "base_model");
        Directory.CreateDirectory(Checkpoint);
        if (writeDit is not null)
            writeDit(Checkpoint);
        else if (dense is not null)
            LongCatTestData.WriteCheckpoint(Checkpoint, dense, writeConfig: false, indexFileName: LongCatCheckpoint.IndexFileName);
        else
            WriteCheckpoint(Checkpoint, useInt8);
        if (dense is not null || writeDit is not null)
            File.WriteAllText(Path.Combine(Checkpoint, "config.json"), ConfigJson(Config));
        foreach ((string folder, string file, int seed) in onePointSix
            ? new[] { ("loraA", "lora.safetensors", 81), ("loraB", "lora.safetensors", 82) }
            : [("lora", "dmd_lora.safetensors", 81)])
        {
            Directory.CreateDirectory(Path.Combine(Root, folder));
            using LongCatLoraFixture lora = new(Path.Combine(Root, folder, file), Config, seed);
        }
    }

    /// <summary>The tiny avatar DiT with the real audio window: 5 blocks of 1280 channels.</summary>
    public static LongCatConfig Config => LongCatTestData.TinyAvatar with { AudioBlock = 5, AudioChannel = 1280 };

    public string Root { get; }

    /// <summary>The DiT folder a recipe is pointed at.</summary>
    public string Checkpoint { get; }

    /// <summary>Stand-in driving features: seeded values of the Whisper features' shape. Each call's speech is added to <paramref name="calls"/>.</summary>
    public static LongCatDrivingFeatures Features(int seed, List<(float[] Speech, int Fps)>? calls = null) => (speech, fps) =>
    {
        calls?.Add(([.. speech], fps));
        Random rng = new(seed);
        return LongCatTestData.RandomValues(rng, (int)(speech.Length / (double)SampleRate * fps) * Config.AudioBlock * Config.AudioChannel);
    };

    /// <summary>A square gradient portrait of side <paramref name="size"/>.</summary>
    public static ImageData Portrait(int size)
    {
        byte[] rgb = new byte[size * size * 3];
        for (int i = 0; i < rgb.Length; i++)
            rgb[i] = (byte)(i % 251);
        return new ImageData { Rgb = rgb, Width = size, Height = size };
    }

    /// <summary>A 16 kHz mono tone of <paramref name="seconds"/> as a WAV clip.</summary>
    public static AudioClip Speech(double seconds)
    {
        float[] samples = new float[(int)(seconds * SampleRate)];
        for (int i = 0; i < samples.Length; i++)
            samples[i] = 0.2f * MathF.Sin(2f * MathF.PI * 220f * i / SampleRate);
        using MemoryStream stream = new();
        WavFile.WriteMono16(stream, samples, SampleRate);
        return new AudioClip { Data = stream.ToArray(), Format = "wav" };
    }

    public void Dispose() => Directory.Delete(Root, recursive: true);

    private static void WriteCheckpoint(string directory, bool int8)
    {
        LongCatConfig config = Config;
        Dictionary<string, Tensor> weights = LongCatTestData.AvatarWeights(config, seed: 71);
        Dictionary<string, Tensor> written = int8 ? LongCatTestData.QuantizePerRow(weights) : weights;
        try
        {
            LongCatTestData.WriteCheckpoint(directory, written, writeConfig: false,
                indexFileName: int8 ? LongCatCheckpoint.Int8IndexFileName : LongCatCheckpoint.IndexFileName);
            File.WriteAllText(Path.Combine(directory, "config.json"), ConfigJson(config));
        }
        finally
        {
            foreach ((string key, Tensor tensor) in written)
                if (!weights.TryGetValue(key, out Tensor? dense) || !ReferenceEquals(dense, tensor))
                    tensor.Dispose();
            LongCatTestData.DisposeAll(weights);
        }
    }

    /// <summary>The config the recipe reads, for the tiny avatar DiT.</summary>
    internal static string ConfigJson(LongCatConfig config) => $$"""
                {"in_channels":{{config.InChannels}},"out_channels":{{config.OutChannels}},"hidden_size":{{config.HiddenSize}},
                 "depth":{{config.Depth}},"num_heads":{{config.NumHeads}},"caption_channels":{{config.CaptionChannels}},"mlp_ratio":{{config.MlpRatio}},
                 "adaln_tembed_dim":{{config.AdalnTembedDim}},"frequency_embedding_size":{{config.FrequencyEmbeddingSize}},"patch_size":[1,2,2],
                 "text_tokens_zero_pad":true,"output_dim":{{config.AudioOutputDim}},"audio_window":{{config.AudioWindow}},
                 "audio_block":{{config.AudioBlock}},"audio_channel":{{config.AudioChannel}},"intermediate_dim":{{config.AudioIntermediateDim}},
                 "context_tokens":{{config.AudioContextTokens}},"vae_scale":{{config.AudioVaeScale}}}
                """;
}
