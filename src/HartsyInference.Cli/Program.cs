using HartsyInference.Cli.Commands;
using HartsyInference.Cli.Infra;
using HartsyInference.Cli.Repl;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Logging;
using Spectre.Console;
using Spectre.Console.Cli;

namespace HartsyInference.Cli;

/// <summary>Entry point for the unified <c>hartsy</c> CLI: registers every per-modality generate command plus the catalog/cache commands.</summary>
/// <remarks>Launches the interactive REPL when run with no arguments.</remarks>
public static class Program
{
    /// <summary>The configured minimum log level, or Warning when unset or unparsable.</summary>
    private static LogLevel ResolveLogLevel() =>
        Enum.TryParse(EngineKnobs.LogLevel.Value, ignoreCase: true, out LogLevel level) ? level : LogLevel.Warning;

    /// <summary>Parses <paramref name="args"/> and dispatches to a command; with no args, shows the banner and usage.</summary>
    public static int Main(string[] args)
    {
        // Warning by default keeps the REPL/one-shot output clean; diagnostics.logLevel exposes the engine's
        // per-phase / per-step diagnostics (D2H sync counts, phase timings) without a rebuild. Re-read after the
        // --set profile is pushed below, since that is where a caller raising it for one run supplies it.
        Logs.MinLevel = ResolveLogLevel();



        // Applied here rather than per command: it must be in force before the engine is constructed, since
        // construction-scoped settings are read while the model loads. The CLI is one run per process, so a
        // process-lifetime scope cannot leak into anyone else's generation.
        IDisposable? knobScope = null;
        try
        {
            KnobProfile? profile = KnobCli.Build(ArgValue(args, "--profile"), ArgValues(args, "--set"));
            if (profile is not null)
            {
                knobScope = profile.Push();
                // The level was resolved before the profile existed, so a --set that raises it only binds here.
                Logs.MinLevel = ResolveLogLevel();
                AnsiConsole.MarkupLine($"[#9aa4af]settings[/] [#2ea5e0]{Markup.Escape(profile.ToString())}[/]");
            }
        }
        catch (ArgumentException ex)
        {
            AnsiConsole.MarkupLine($"[red]{Markup.Escape(ex.Message)}[/]");
            return 1;
        }
        using IDisposable? scopedSettings = knobScope;

        if (args.Length == 0)
        {
            using ReplSession repl = new ReplSession();
            return repl.Run();
        }

        CommandApp app = new CommandApp();
        app.Configure(config =>
        {
            config.SetApplicationName("hartsy");
            // Refuse an option no command declares instead of collecting it as a "remaining" argument. The default
            // is to accept it silently, which means a typo does not fail — it runs the generation with a DIFFERENT
            // setting than the one asked for and reports success. `--cfgscale 2` (the option is `--cfg`) and
            // `--detect` (it is `--mode detect`) both did exactly that, and the second one produced a confusing
            // failure three layers down, in a model loader complaining about a missing weight.
            config.Settings.StrictParsing = true;
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
                .WithDescription("Re-voice audio using a target speaker reference.")
                .WithExample("convert", "source.wav", "-m", "openvoice", "--target", "reference.wav");
            config.AddBranch("settings", settings =>
            {
                settings.SetDescription("Read and change engine settings. They live in one file; 'settings path' prints it.");
                settings.AddCommand<SettingsListCommand>("list")
                    .WithDescription("List every setting with its type, default and scope.")
                    .WithExample("settings", "list");
                settings.AddCommand<SettingsGetCommand>("get")
                    .WithDescription("Show one setting's effective value and which layer supplied it.")
                    .WithExample("settings", "get", "paths.modelsRoot");
                settings.AddCommand<SettingsSetCommand>("set")
                    .WithDescription("Write one setting to the settings file so it survives a restart.")
                    .WithExample("settings", "set", "paths.modelsRoot", "/mnt/models");
                settings.AddCommand<SettingsPathCommand>("path")
                    .WithDescription("Print the settings file this process reads and writes.")
                    .WithExample("settings", "path");
            });
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
            config.AddCommand<InspectCommand>("inspect")
                .WithDescription("Inspect a checkpoint header and resolve its execution profile without loading weights.")
                .WithExample("inspect", "--modality", "video", "--model-path", "/models/h3.safetensors", "--json");
            config.AddCommand<QuantizeCommand>("quantize")
                .WithDescription("Write a quantized GGUF copy of a checkpoint. Offline — the engine never quantizes at load.")
                .WithExample("quantize", "--model-path", "/models/flux2-klein.safetensors", "-o", "/models/klein-Q8_0.gguf", "--quant", "Q8_0");
        });

        // Without this the run always fails: StrictParsing above refuses any option no command declares, and no
        // command declares --profile/--set because they are read here, before the parser exists. They have to be
        // taken out of what the parser sees or every invocation that uses one dies on "Unexpected option 'set'".
        return app.Run(WithoutKnobArgs(args));
    }

    /// <summary>Drops each <c>--profile</c>/<c>--set</c> flag and the value that follows it, leaving the command line the parser should see.</summary>
    /// <remarks>A trailing flag with no value is left in place on purpose: <see cref="ArgValues"/> cannot have read
    /// it, so silently swallowing it here would run the generation while ignoring what the operator asked for —
    /// the same failure mode <c>StrictParsing</c> exists to prevent. The parser rejects it by name instead.</remarks>
    private static string[] WithoutKnobArgs(string[] args)
    {
        List<string> kept = new List<string>(args.Length);
        for (int i = 0; i < args.Length; i++)
        {
            bool isKnobFlag = string.Equals(args[i], "--profile", StringComparison.Ordinal)
                || string.Equals(args[i], "--set", StringComparison.Ordinal);
            if (isKnobFlag && i + 1 < args.Length)
            {
                i++;
                continue;
            }
            kept.Add(args[i]);
        }
        return [.. kept];
    }

    /// <summary>Last value of a repeated <c>--flag value</c> pair, or null when absent.</summary>
    private static string? ArgValue(string[] args, string flag)
    {
        for (int i = args.Length - 2; i >= 0; i--)
        {
            if (string.Equals(args[i], flag, StringComparison.Ordinal))
            {
                return args[i + 1];
            }
        }
        return null;
    }

    /// <summary>Every value of a repeated <c>--flag value</c> pair, in order.</summary>
    private static string[] ArgValues(string[] args, string flag)
    {
        List<string> values = [];
        for (int i = 0; i < args.Length - 1; i++)
        {
            if (string.Equals(args[i], flag, StringComparison.Ordinal))
            {
                values.Add(args[i + 1]);
            }
        }
        return [.. values];
    }

}
