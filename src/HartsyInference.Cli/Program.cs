using HartsyInference.Cli.Commands;
using HartsyInference.Cli.Repl;
using HartsyInference.Core.Logging;
using Spectre.Console.Cli;

namespace HartsyInference.Cli;

/// <summary>Entry point for the unified <c>hartsy</c> CLI: registers every per-modality generate command plus the catalog/cache commands.</summary>
/// <remarks>Launches the interactive REPL when run with no arguments.</remarks>
public static class Program
{
    /// <summary>Parses <paramref name="args"/> and dispatches to a command; with no args, shows the banner and usage.</summary>
    public static int Main(string[] args)
    {
        // Warning by default keeps the REPL/one-shot output clean; HARTSY_LOG_LEVEL exposes the engine's
        // per-phase / per-step diagnostics (D2H sync counts, phase timings) without a rebuild.
        Logs.MinLevel = Enum.TryParse(Environment.GetEnvironmentVariable("HARTSY_LOG_LEVEL"), ignoreCase: true, out LogLevel level)
            ? level : LogLevel.Warning;

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
            config.AddCommand<WakeTrainCommand>("wake-train")
                .WithDescription("Train a new wake word from its text using the engine's own TTS voices.")
                .WithExample("wake-train", "\"hey hartsy\"", "--negative-audio", "~/recordings");
            config.AddCommand<WakeClientCommand>("wake-client")
                .WithDescription("Stream a WAV to the wake-word listener as if it were a voice satellite.")
                .WithExample("wake-client", "speech.wav", "--host", "127.0.0.1", "-p", "10800");
            config.AddCommand<ThreeDCommand>("3d")
                .WithDescription("Generate a 3D mesh (GLB) or Gaussian-splat cloud (PLY) from an image with TripoSR, Hunyuan3D, or TRELLIS.")
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
            config.AddCommand<RestoreCommand>("restore")
                .WithDescription("Restore a degraded video or image with SeedVR2 (upscale, deartifact, denoise).")
                .WithExample("restore", "old_clip.mp4", "-m", "seedvr2-3b");
            config.AddCommand<WorldCommand>("world")
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
