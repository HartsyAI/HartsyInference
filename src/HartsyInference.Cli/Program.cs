using HartsyInference.Cli.Commands;
using HartsyInference.Cli.Repl;
using HartsyInference.Core.Logging;
using Spectre.Console.Cli;

namespace HartsyInference.Cli;

/// <summary>Entry point for the unified <c>hartsy</c> CLI: a Spectre.Console command app over every engine modality.
/// Phase 1 wires the shared infrastructure and the catalog/cache commands (list, models, pull); the interactive
/// REPL and per-modality generate commands land in later phases.</summary>
public static class Program
{
    /// <summary>Parses <paramref name="args"/> and dispatches to a command; with no args, shows the banner and usage.</summary>
    public static int Main(string[] args)
    {
        Logs.MinLevel = LogLevel.Warning;

        if (args.Length == 0)
        {
            using ReplSession repl = new ReplSession();
            return repl.Run();
        }

        CommandApp app = new CommandApp();
        app.Configure(config =>
        {
            config.SetApplicationName("hartsy");
            config.AddCommand<TextCommand>("text")
                .WithDescription("Generate text from a prompt with a local LLM (streams tokens).")
                .WithExample("text", "\"Explain transformers in one sentence.\"", "-m", "qwen3", "--model-path", "/models/qwen3")
                .WithExample("text", "\"Hello\"", "--model-path", "model.gguf", "-b", "cpu");
            config.AddCommand<ImageCommand>("image")
                .WithDescription("Generate an image from a prompt with a diffusion checkpoint (SDXL).")
                .WithExample("image", "\"a fox in snow\"", "--model-path", "sdxl.safetensors", "--steps", "30");
            config.AddCommand<TranscribeCommand>("transcribe")
                .WithDescription("Transcribe a WAV file to text with Whisper.")
                .WithExample("transcribe", "speech.wav", "-m", "whisper-base");
            config.AddCommand<SpeechCommand>("speak")
                .WithDescription("Synthesize speech from text with Piper (saves a WAV).")
                .WithExample("speak", "\"Hello world\"", "-m", "en_US-lessac-medium");
            config.AddCommand<ThreeDCommand>("3d")
                .WithDescription("Generate a 3D mesh (GLB) from an image with TripoSR or Hunyuan3D.")
                .WithExample("3d", "photo.png", "-m", "triposr", "--model-path", "/models/triposr");
            config.AddCommand<VisionCommand>("vision")
                .WithDescription("Run CLIP embedding or YOLO detection on an image.")
                .WithExample("vision", "photo.png", "-m", "yolo11", "--model-path", "yolo11n.safetensors");
            config.AddCommand<MusicCommand>("music")
                .WithDescription("Generate music from a prompt with MusicGen (saves a WAV).")
                .WithExample("music", "\"lofi hip hop, mellow piano\"", "--model-path", "musicgen-small.safetensors");
            config.AddCommand<ConvertCommand>("convert")
                .WithDescription("Re-voice a source clip with RVC or OpenVoice (saves a WAV).")
                .WithExample("convert", "source.wav", "-m", "openvoice", "--target", "reference.wav");
            config.AddBranch("fx", fx =>
            {
                fx.SetDescription("Audio effects: stem separation (Demucs) and speech enhancement (Resemble-Enhance).");
                fx.AddCommand<FxSeparateCommand>("separate")
                    .WithDescription("Split a mix into stems with Demucs (saves one WAV per stem).")
                    .WithExample("fx", "separate", "mix.wav", "-m", "demucs");
                fx.AddCommand<FxEnhanceCommand>("enhance")
                    .WithDescription("Denoise and enhance a recording with Resemble-Enhance (saves a WAV).")
                    .WithExample("fx", "enhance", "noisy.wav");
            });
            config.AddCommand<VideoCommand>("video")
                .WithDescription("Generate a video (frame sequence) from a prompt with any registered video family (CUDA).")
                .WithExample("video", "a cat walking through a sunlit garden", "-m", "ltx-video");
            config.AddCommand<InteractiveCommand>("world")
                .WithDescription("Roll out an Oasis world model from a first-frame image (canned action plan).");
            config.AddCommand<PreviewCommand>("preview")
                .WithDescription("Display an image (PNG/BMP) inline in the terminal.")
                .WithExample("preview", "output/a-fox-in-snow-0001.png");
            config.AddCommand<ListCommand>("list")
                .WithDescription("List models in the catalog, optionally filtered by modality.")
                .WithExample("list", "image")
                .WithExample("list", "--verified");
            config.AddCommand<ModelsCommand>("models")
                .WithDescription("Show models downloaded into the local cache.");
            config.AddCommand<PullCommand>("pull")
                .WithDescription("Download a model from HuggingFace (or register a local path) into the cache.")
                .WithExample("pull", "stabilityai/stable-diffusion-xl-base-1.0");
        });

        return app.Run(args);
    }
}
