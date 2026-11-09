using System.ComponentModel;
using System.Globalization;
using HartsyInference.Cli.Infra;
using Spectre.Console;
using Spectre.Console.Cli;

namespace HartsyInference.Cli.Commands;

/// <summary>Generates text from a prompt with a local LLM (GGUF or safetensors), streaming tokens as they arrive.</summary>
public sealed class TextCommand : Command<TextCommand.Settings>
{
    /// <summary>Options for <c>hartsy text</c>.</summary>
    public sealed class Settings : CommandSettings
    {
        /// <summary>The prompt to complete.</summary>
        [CommandArgument(0, "<prompt>")]
        [Description("The prompt to complete.")]
        public string Prompt { get; init; } = "";

        /// <summary>Model id (catalog) or path. Optional when <c>--model-path</c> is given.</summary>
        [CommandOption("-m|--model")]
        [Description("Model id (e.g. qwen2, qwen3) or a local path. Optional when --model-path is given.")]
        public string Model { get; init; } = "";

        /// <summary>Explicit checkpoint path (.gguf file or safetensors directory).</summary>
        [CommandOption("--model-path")]
        [Description("Path to a .gguf file or a safetensors model directory.")]
        public string? ModelPath { get; init; }

        /// <summary>Compute backend selector.</summary>
        [CommandOption("-b|--backend")]
        [Description("Backend: auto, cpu, cuda, or vulkan.")]
        public string Backend { get; init; } = "auto";

        /// <summary>Device selector for the text service, including the layer-split composite form.</summary>
        [CommandOption("--device")]
        [Description("Device for the LLM: 'cuda:0' pins an ordinal; 'cuda:0+cuda:1' splits the layers across both GPUs by free VRAM (VRAM pooling for models that don't fit one card).")]
        public string? Device { get; init; }

        /// <summary>Maximum new tokens to generate.</summary>
        [CommandOption("--max-tokens")]
        [Description("Maximum number of tokens to generate.")]
        public int MaxTokens { get; init; } = 256;

        /// <summary>Sampling temperature; &lt;= 0 selects greedy decoding.</summary>
        [CommandOption("--temperature")]
        [Description("Sampling temperature; 0 or less means greedy.")]
        public float Temperature { get; init; } = 0.7f;

        /// <summary>Nucleus sampling cutoff.</summary>
        [CommandOption("--top-p")]
        [Description("Nucleus (top-p) sampling cutoff.")]
        public float TopP { get; init; } = 0.95f;

        /// <summary>RNG seed; &lt; 0 uses a reproducible default.</summary>
        [CommandOption("--seed")]
        [Description("RNG seed; negative uses a reproducible default.")]
        public int Seed { get; init; } = -1;

        /// <summary>System prompt applied ahead of the user turn; unset lets the model/template default apply.</summary>
        [CommandOption("--system")]
        [Description("System prompt applied ahead of the user turn.")]
        public string? System { get; init; }

        /// <summary>Image(s) to attach to the prompt, for vision-language models. Repeat the flag for more than one.</summary>
        [CommandOption("-i|--image")]
        [Description("Attach an image (PNG/BMP) for vision-language models. Repeat for multiple images.")]
        public string[] Image { get; init; } = [];

        /// <summary>A video file to attach, for video-capable models (decoded with ffmpeg and sampled per the model's settings).</summary>
        [CommandOption("--video")]
        [Description("Attach a video file for video-capable models (MOSS-VL-Realtime). Needs ffmpeg on PATH.")]
        public string? Video { get; init; }

        /// <summary>A folder of PNG/BMP frames to attach as a video, in name order, at <see cref="FrameFps"/>.</summary>
        [CommandOption("--frames-dir")]
        [Description("Attach a folder of PNG/BMP frames (sorted by name) as a video; no ffmpeg needed. See --frame-fps.")]
        public string? FramesDir { get; init; }

        /// <summary>Frames per second of a <see cref="FramesDir"/> video, which sets each frame's timestamp.</summary>
        [CommandOption("--frame-fps")]
        [Description("Frame rate of a --frames-dir video (frame i is at i / fps seconds). Default 1.")]
        public double FrameFps { get; init; } = 1.0;

        /// <summary>Frames sampled per second from a <see cref="Video"/>; unset keeps the model's setting.</summary>
        [CommandOption("--video-fps")]
        [Description("Frames sampled per second from --video; unset keeps the model's setting (MOSS-VL: 1).")]
        public double? VideoFps { get; init; }

        /// <summary>The most frames sampled from a <see cref="Video"/>; unset keeps the model's setting.</summary>
        [CommandOption("--video-max-frames")]
        [Description("Most frames sampled from --video; unset keeps the model's setting (MOSS-VL: 256).")]
        public int? VideoMaxFrames { get; init; }

        /// <summary>Runs a live session: <see cref="Video"/> or <see cref="FramesDir"/> is replayed at its own pace, the prompt opens the chat,
        /// and each stdin line is a further prompt.</summary>
        [CommandOption("--realtime")]
        [Description("Live session (MOSS-VL-Realtime): replay --video or --frames-dir at its pace; each stdin line is a new prompt. Greedy unless --temperature > 0 is given explicitly.")]
        public bool Realtime { get; init; }

        /// <summary>Seconds of silence after the last frame that end a <see cref="Realtime"/> run.</summary>
        [CommandOption("--idle-exit")]
        [Description("With --realtime: end the run after this many seconds of model silence once every frame is in. Default 3.")]
        public double? IdleExit { get; init; }

        /// <summary>Caps the live decode rate.</summary>
        [CommandOption("--tokens-per-second")]
        [Description("With --realtime: the most tokens decoded per second; unset is unpaced.")]
        public double? TokensPerSecond { get; init; }

        /// <summary>Prints the model's silence markers in a live run.</summary>
        [CommandOption("--show-silence")]
        [Description("With --realtime: print [silence] when the model stays quiet, and its other control markers (e.g. [response]).")]
        public bool ShowSilence { get; init; }

        /// <summary>Top-K sampling cutoff; unset uses the model default.</summary>
        [CommandOption("--top-k")]
        [Description("Top-K sampling cutoff; unset uses the model default.")]
        public int? TopK { get; init; }

        /// <summary>Min-P sampling cutoff; unset uses the model default.</summary>
        [CommandOption("--min-p")]
        [Description("Min-P sampling cutoff; unset uses the model default.")]
        public float? MinP { get; init; }

        /// <summary>Repetition penalty; unset uses the model default.</summary>
        [CommandOption("--repetition-penalty")]
        [Description("Repetition penalty; unset uses the model default.")]
        public float? RepetitionPenalty { get; init; }

        /// <summary>Enable the model's reasoning/"thinking" mode (Qwen3-family); unset lets the model/template default apply.</summary>
        [CommandOption("--thinking")]
        [Description("Enable chat-template reasoning/\"thinking\" mode (Qwen3-family). Unset lets the model/template decide.")]
        public bool? Thinking { get; init; }

        /// <summary>Disable the model's reasoning/"thinking" mode. Mutually exclusive with <see cref="Thinking"/>.</summary>
        [CommandOption("--no-thinking")]
        [Description("Disable chat-template reasoning/\"thinking\" mode.")]
        public bool NoThinking { get; init; }

        /// <summary>Reasoning effort for models that take one (DeepSeek-V4.1 thinking mode): <c>low</c>, <c>high</c>, <c>max</c>, or an integer in [1, 100]. Unset uses the template default.</summary>
        [CommandOption("--reasoning-effort")]
        [Description("Reasoning effort for models that take one (DeepSeek-V4.1 thinking): low, high, max, or an integer in [[1, 100]].")]
        public string? ReasoningEffort { get; init; }

        /// <summary>Recurrent-block passes for a looped model (Dory: 1 to 4); unset uses the checkpoint's default. Ignored by other models.</summary>
        [CommandOption("--loops")]
        [Description("Recurrent-block passes for a looped model (Dory: 1 fastest, 4 most accurate). Unset uses the checkpoint default.")]
        public int? Loops { get; init; }

        /// <summary>Keep weights compressed at their on-disk quant instead of dequantizing to full precision.</summary>
        [CommandOption("--low-vram-quant")]
        [Description("Keep weights compressed at their on-disk quant to reduce VRAM use.")]
        public bool LowVramQuant { get; init; }

        /// <summary>Re-quantizes a checkpoint folder's plain float projections on load.</summary>
        [CommandOption("--precision")]
        [Description("For a checkpoint folder: re-quantize its plain BF16 projections to q8_0, q6_k, q5_k, q4_k or q4_0 on load (cached on disk). NF4/FP8 folders load packed either way.")]
        public string? Precision { get; init; }

        /// <summary>Where the model is placed when this run loads it.</summary>
        [CommandOption("--placement")]
        [Description("Model placement for a GGUF on CUDA: auto (one GPU, else split across GPUs, else expert offload), gpu, split or offload.")]
        public string? Placement { get; init; }

        /// <summary>Free the model's device memory after this request completes.</summary>
        [CommandOption("--always-free-memory")]
        [Description("Free the model's device memory after this request completes.")]
        public bool AlwaysFreeMemory { get; init; }

        /// <summary>Directory to save the generated text to (as a .txt file).</summary>
        [CommandOption("-o|--output")]
        [Description("Directory to save the generated text (.txt).")]
        public string? Output { get; init; }

        /// <summary>Suppress progress and stream output; print only the final text.</summary>
        [CommandOption("-q|--quiet")]
        [Description("Suppress staging/streaming; print only the final text.")]
        public bool Quiet { get; init; }

        /// <summary>Reads the prompt from a file instead of the positional argument (long prompts, prefill measurements).</summary>
        [CommandOption("--prompt-file")]
        [Description("Read the prompt from this file; the positional prompt is then ignored.")]
        public string? PromptFile { get; init; }

        /// <summary>Writes each completion token's log-probability and the top token's to a TSV (eager decode, single-request route).</summary>
        [CommandOption("--scores-out")]
        [Description("Write per-token log-probabilities (index, token_id, logprob, top_id, top_logprob) to this TSV; decodes eagerly.")]
        public string? ScoresOut { get; init; }

        /// <summary>Teacher forcing: the completion is the token_id column of a scores TSV written by --scores-out.</summary>
        [CommandOption("--teacher-force")]
        [Description("Force the completion to the token ids in this --scores-out TSV and score them; requires --scores-out.")]
        public string? TeacherForce { get; init; }

        /// <summary>Force CUDA-graph decode on; requires greedy sampling and an eligible dense GQA/RoPE model.</summary>
        /// <remarks>Forces --temperature to 0 if a positive temperature was also given.</remarks>
        [CommandOption("--graph-decode")]
        [Description("Force CUDA-graph decode (forces greedy sampling). Falls back silently if the model/backend isn't eligible.")]
        public bool GraphDecode { get; init; }
    }

    /// <inheritdoc/>
    public override int Execute(CommandContext context, Settings settings)
    {
        string prompt = settings.Prompt;
        if (settings.PromptFile is { Length: > 0 } promptFile)
        {
            if (!File.Exists(promptFile))
            {
                AnsiConsole.MarkupLine($"[red]Prompt file not found:[/] {Markup.Escape(promptFile)}");
                return 1;
            }
            prompt = File.ReadAllText(promptFile);
        }
        int exitCode;
        // A live session may open with an empty prompt (the reference's default) and take its questions from stdin.
        if (!settings.Realtime && !CommandRunner.RequireNonEmpty(prompt, "A prompt is required.", out exitCode))
            return exitCode;
        if (settings.Realtime && settings.Video is null && settings.FramesDir is null)
        {
            AnsiConsole.MarkupLine("[red]--realtime needs --video or --frames-dir to replay.[/]");
            return 1;
        }
        if (settings.Realtime && settings.Image.Length > 0)
        {
            AnsiConsole.MarkupLine("[red]--realtime takes frames from --video or --frames-dir, not --image.[/]");
            return 1;
        }
        if (settings.TeacherForce is { Length: > 0 } && settings.ScoresOut is not { Length: > 0 })
        {
            AnsiConsole.MarkupLine("[red]--teacher-force needs --scores-out to write the scores of the forced tokens.[/]");
            return 1;
        }

        if (!CommandRunner.RequireModelOrPath(settings.Model, settings.ModelPath, "--model", "--model-path", out exitCode))
            return exitCode;

        if (settings.Thinking == true && settings.NoThinking)
        {
            AnsiConsole.MarkupLine("[red]--thinking and --no-thinking are mutually exclusive.[/]");
            return 1;
        }

        foreach (string imagePath in settings.Image)
        {
            if (!File.Exists(imagePath))
            {
                AnsiConsole.MarkupLine($"[red]Image not found:[/] {imagePath}");
                return 1;
            }
        }

        if (settings.Video is { } videoPath && !File.Exists(videoPath))
        {
            AnsiConsole.MarkupLine($"[red]Video not found:[/] {videoPath}");
            return 1;
        }
        if (settings.FramesDir is { } framesDir && !Directory.Exists(framesDir))
        {
            AnsiConsole.MarkupLine($"[red]Frames folder not found:[/] {framesDir}");
            return 1;
        }
        if (settings.Video is not null && settings.FramesDir is not null)
        {
            AnsiConsole.MarkupLine("[red]--video and --frames-dir are mutually exclusive.[/]");
            return 1;
        }
        if (settings.FrameFps <= 0)
        {
            AnsiConsole.MarkupLine("[red]--frame-fps must be positive.[/]");
            return 1;
        }

        float temperature = settings.Temperature;
        if (settings.GraphDecode && temperature > 0f)
        {
            AnsiConsole.MarkupLine("[yellow]--graph-decode requires greedy sampling; forcing --temperature to 0.[/]");
            temperature = 0f;
        }

        ParamState parameters = new ParamState(Modality.Text) { Backend = settings.Backend, Model = settings.Model, OutputDir = settings.Output };
        if (settings.Device is { Length: > 0 } device) parameters.Put("device", device);
        parameters.Put("max-tokens", settings.MaxTokens.ToString(CultureInfo.InvariantCulture));
        // A live run is greedy like the reference unless a temperature is asked for; the one-shot default (0.7) does not apply.
        if (!settings.Realtime || Environment.GetCommandLineArgs().Contains("--temperature"))
            parameters.Put("temperature", temperature.ToString(CultureInfo.InvariantCulture));
        if (settings.Realtime)
        {
            parameters.Put("realtime", "true");
            parameters.PutIfSet("idle-exit", settings.IdleExit);
            parameters.PutIfSet("tokens-per-second", settings.TokensPerSecond);
            parameters.Put("show-silence", settings.ShowSilence ? "true" : "false");
        }
        parameters.Put("top-p", settings.TopP.ToString(CultureInfo.InvariantCulture));
        parameters.Put("seed", settings.Seed.ToString(CultureInfo.InvariantCulture));
        parameters.Put("graph-decode", settings.GraphDecode ? "true" : "false");
        if (settings.System is { Length: > 0 } system)
            parameters.Put("system", system);
        if (settings.Image.Length > 0)
            parameters.Put("image", string.Join('\n', settings.Image));
        if (settings.Video is { } video) parameters.Put("video", video);
        if (settings.FramesDir is { } frames)
        {
            parameters.Put("frames-dir", frames);
            parameters.Put("frame-fps", settings.FrameFps.ToString(CultureInfo.InvariantCulture));
        }
        parameters.PutIfSet("video-fps", settings.VideoFps);
        parameters.PutIfSet("video-max-frames", settings.VideoMaxFrames);
        parameters.PutIfSet("top-k", settings.TopK);
        parameters.PutIfSet("min-p", settings.MinP);
        parameters.PutIfSet("repetition-penalty", settings.RepetitionPenalty);
        if (settings.Thinking is { } thinking)
            parameters.Put("thinking", thinking ? "true" : "false");
        else if (settings.NoThinking)
            parameters.Put("thinking", "false");
        if (settings.ReasoningEffort is { Length: > 0 } effort)
            parameters.Put("reasoning-effort", effort);
        if (settings.Loops is { } loops)
        {
            if (loops < 1)
            {
                AnsiConsole.MarkupLine("[red]--loops must be at least 1.[/]");
                return 1;
            }
            parameters.Put("loops", loops.ToString(System.Globalization.CultureInfo.InvariantCulture));
        }
        parameters.Put("low-vram-quant", settings.LowVramQuant ? "true" : "false");
        if (settings.Precision is { Length: > 0 } precision) parameters.Put("precision", precision);
        parameters.Put("always-free-memory", settings.AlwaysFreeMemory ? "true" : "false");
        if (settings.Placement is { Length: > 0 } placement)
            parameters.Put("placement", placement);
        if (settings.ScoresOut is { Length: > 0 } scoresOut)
            parameters.Put("scores-out", scoresOut);
        if (settings.TeacherForce is { Length: > 0 } teacherForce)
        {
            if (!File.Exists(teacherForce))
            {
                AnsiConsole.MarkupLine($"[red]Teacher-force file not found:[/] {Markup.Escape(teacherForce)}");
                return 1;
            }
            parameters.Put("teacher-force", teacherForce);
        }

        ModelSpec spec = ModelResolver.Resolve(settings.Model, settings.ModelPath, Modality.Text);
        string label = CommandRunner.ResolveLabel(spec, settings.Model, settings.ModelPath);

        return CommandRunner.Run(Modality.Text, spec, prompt, parameters, settings.Backend, settings.Quiet,
            settings.Output, label, showResponseRule: true);
    }
}
