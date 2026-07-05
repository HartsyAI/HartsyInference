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
            config.AddCommand<VideoCommand>("video")
                .WithDescription("Generate a video (BMP frame sequence) from a prompt with LTX-Video (CUDA).");
            config.AddCommand<InteractiveCommand>("world")
                .WithDescription("Roll out an Oasis world model from a first-frame image (canned action plan).");
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
