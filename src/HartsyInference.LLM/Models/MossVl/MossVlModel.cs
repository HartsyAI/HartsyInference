using System.Text.Json;
using HartsyInference.Core.Exceptions;
using HartsyInference.LLM.Generation;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.Gguf;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.LLM.Models.MossVl;

/// <summary>A loaded MOSS-VL text side: the decoder, its config, tokenizer and image processor, from either source the model ships
/// in. A checkpoint folder is read with its <c>config.json</c>, <c>tokenizer.json</c> and <c>preprocessor_config.json</c>. A GGUF
/// uses those files when they sit beside it and otherwise falls back to the GGUF's own metadata and tokenizer, so a lone GGUF loads.
/// The vision tower lives in the Diffusion package and is loaded by the caller (see <c>MossVlVisionTower</c>).</summary>
public sealed class MossVlModel : IDisposable
{
    /// <summary>The Hugging Face repository whose tokenizer and JSON configs a lone GGUF needs.</summary>
    public const string TokenizerRepo = "OpenMOSS-Team/MOSS-VL-Realtime-SGLANG";

    private MossVlModel(MossVlConfig config, MossVlTextDecoder decoder, ILlmTokenizer tokenizer, MossVlImageProcessor processor,
        Core.Imaging.VideoFrameSamplingOptions videoSampling, string path, bool gguf)
    {
        VideoSampling = videoSampling;
        Config = config;
        Decoder = decoder;
        Tokenizer = tokenizer;
        Processor = processor;
        Path = path;
        IsGguf = gguf;
    }

    public MossVlConfig Config { get; }

    public MossVlTextDecoder Decoder { get; }

    public ILlmTokenizer Tokenizer { get; }

    public MossVlImageProcessor Processor { get; }

    /// <summary>How a video is sampled and sized (<c>video_preprocessor_config.json</c>, or the released checkpoint's values).</summary>
    public Core.Imaging.VideoFrameSamplingOptions VideoSampling { get; }

    /// <summary>The GGUF file or checkpoint folder the model came from.</summary>
    public string Path { get; }

    /// <summary>True when <see cref="Path"/> is a GGUF (its vision tower is inside the same file).</summary>
    public bool IsGguf { get; }

    /// <summary>Whether <paramref name="path"/> is a MOSS-VL checkpoint folder (<c>model_type</c> <c>moss_vl</c>) or a GGUF whose
    /// architecture is <c>moss_vl</c>. Reads only the config or the GGUF header.</summary>
    public static bool IsMossVl(string path)
    {
        if (Directory.Exists(path))
            return HfCheckpointDirectory.TryProbe(path) is { } info && string.Equals(info.ModelType, MossVlConfig.ModelType, StringComparison.Ordinal);
        if (!File.Exists(path) || !path.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase)) return false;
        using GgufLoader probe = new();
        probe.Load(path);
        return string.Equals(probe.Metadata.GetString("general.architecture"), MossVlConfig.ModelType, StringComparison.Ordinal);
    }

    /// <summary>Loads the text side from <paramref name="path"/> for <see cref="MossVlLoadOptions.Backend"/>.</summary>
    /// <exception cref="HartsyInferenceException">The path is not a MOSS-VL checkpoint, or a file it needs is missing.</exception>
    public static MossVlModel Open(string path, MossVlLoadOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        bool gguf = File.Exists(path);
        string folder = gguf ? System.IO.Path.GetDirectoryName(System.IO.Path.GetFullPath(path))! : path;
        string configPath = System.IO.Path.Combine(folder, HfCheckpointDirectory.ConfigFileName);
        string tokenizerPath = System.IO.Path.Combine(folder, "tokenizer.json");
        string processorPath = System.IO.Path.Combine(folder, "preprocessor_config.json");

        ILlmTokenizer tokenizer;
        MossVlConfig config;
        if (gguf && (!File.Exists(configPath) || !File.Exists(tokenizerPath)))
        {
            // A lone GGUF: its metadata stands in for config.json, and its embedded vocabulary (when the converter wrote one)
            // for tokenizer.json. The released GGUF embeds none, so it needs tokenizer.json beside it.
            using GgufLoader probe = new();
            probe.Load(path);
            if (File.Exists(tokenizerPath))
                tokenizer = LoadTokenizerJson(tokenizerPath);
            else if (probe.Metadata.ContainsKey("tokenizer.ggml.tokens"))
                tokenizer = GgufLanguageModel.BuildTokenizer(probe.Metadata);
            else
                throw new HartsyInferenceException(
                    $"'{System.IO.Path.GetFileName(path)}' embeds no tokenizer, and there is no tokenizer.json beside it. Put the checkpoint's "
                    + $"tokenizer.json (from {TokenizerRepo}) in '{folder}'.");
            config = File.Exists(configPath) ? ReadConfig(configPath) : MossVlConfig.FromGguf(probe.Metadata, tokenizer);
        }
        else
        {
            if (!File.Exists(configPath) || !File.Exists(tokenizerPath))
                throw new HartsyInferenceException($"'{folder}' is missing config.json or tokenizer.json, which a MOSS-VL checkpoint folder needs.");
            config = ReadConfig(configPath);
            tokenizer = LoadTokenizerJson(tokenizerPath);
        }
        MossVlImageProcessor processor = File.Exists(processorPath) ? MossVlImageProcessor.FromFile(processorPath) : MossVlImageProcessor.Default;
        string videoPath = System.IO.Path.Combine(folder, "video_preprocessor_config.json");
        Core.Imaging.VideoFrameSamplingOptions videoSampling = new();
        if (File.Exists(videoPath))
        {
            using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(videoPath));
            videoSampling = MossVlImageProcessor.VideoOptionsFromJson(doc.RootElement);
        }

        MossVlTextDecoder decoder = gguf ? MossVlTextDecoder.OpenGguf(path, config, options) : MossVlTextDecoder.Open(path, config, options);
        return new MossVlModel(config, decoder, tokenizer, processor, videoSampling, path, gguf);
    }

    /// <inheritdoc/>
    public void Dispose() => Decoder.Dispose();

    private static MossVlConfig ReadConfig(string path)
    {
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(path));
        return MossVlConfig.FromJson(doc.RootElement);
    }

    private static ILlmTokenizer LoadTokenizerJson(string path)
    {
        using FileStream stream = File.OpenRead(path);
        return HfTokenizerJson.LoadByteLevelBpe(stream);
    }
}
