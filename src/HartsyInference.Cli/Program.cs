using HartsyInference.Cli.Commands;
using HartsyInference.Cli.Infra;
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
            CliTheme.RenderBanner("auto");

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
