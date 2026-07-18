using HartsyInference.Cli.Commands;
using HartsyInference.Cli.Dispatch;
using HartsyInference.Cli.Infra;
using HartsyInference.Core.Backends;
using HartsyInference.Vision.Codec;
using Spectre.Console;

namespace HartsyInference.Cli.Repl;

/// <summary>The interactive <c>hartsy</c> session: a persistent, editable parameter state, slash commands, and a
/// per-session cache of loaded models so successive generations reuse the same backend and weights.</summary>
public sealed class ReplSession : IDisposable
{
    private static readonly IReadOnlyList<SlashCommand> SlashCommands = new SlashCommand[]
    {
        new("help", "show commands"),
        new("text", "switch to text generation"),
        new("image", "switch to image generation"),
        new("speak", "switch to speech synthesis"),
        new("music", "switch to music generation"),
        new("transcribe", "switch to speech-to-text"),
        new("vision", "switch to vision (embed / detect)"),
        new("video", "switch to video generation"),
        new("3d", "switch to 3D mesh generation"),
        new("world", "switch to world-model rollout"),
        new("model", "set the model (id or path)"),
        new("backend", "set the compute backend"),
        new("set", "set a generation parameter"),
        new("output", "set the artifact output directory"),
        new("show", "show the current state"),
        new("reset", "reset parameters to defaults"),
        new("list", "browse the model catalog"),
        new("models", "show the local model cache"),
        new("preview", "display an image inline"),
        new("clear", "clear the screen"),
        new("quit", "exit"),
    };

    private readonly ModalityDispatch _dispatch = CommandRunner.Dispatch;
    private readonly Dictionary<string, IModalityRunner> _runners = new(StringComparer.OrdinalIgnoreCase);

    private IBackend? _backend;
    private string _backendSelector = "auto";
    private Modality _modality = Modality.Text;
    private ParamState _params = new ParamState(Modality.Text);
    private string _model = "";
    private string? _outputDir;

    /// <summary>Runs the read-eval-print loop until the user exits (or EOF). Always returns 0.</summary>
    public int Run()
    {
        CliTheme.RenderBanner(_backendSelector);
        AnsiConsole.MarkupLine($"[grey]Type a prompt to generate, or[/] [{CliTheme.Accent}]/help[/] [grey]for commands.[/] [grey]Ctrl+C or[/] [{CliTheme.Accent}]/quit[/] [grey]to exit.[/]");
        AnsiConsole.WriteLine();

        while (true)
        {
            string? line = LineEditor.ReadLine(Modalities.ToCliName(_modality), SlashCommands);
            if (line is null)
                break;
            line = line.Trim();
            if (line.Length == 0)
                continue;

            if (line[0] == '/')
            {
                if (!HandleSlash(line))
                    break;
            }
            else
            {
                Generate(line);
            }
        }

        AnsiConsole.MarkupLine("[grey]bye.[/]");
        return 0;
    }

    private bool HandleSlash(string line)
    {
        string[] parts = line.Split(' ', 2, StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries);
        string cmd = parts[0].TrimStart('/').ToLowerInvariant();
        string arg = parts.Length > 1 ? parts[1] : "";

        // A bare /image, /text, … switches modality.
        if (Modalities.TryParse(cmd, out Modality quick))
        {
            SwitchModality(quick);
            return true;
        }

        switch (cmd)
        {
            case "help" or "?":
                PrintHelp();
                break;
            case "quit" or "exit" or "q":
                return false;
            case "mode":
                if (Modalities.TryParse(arg, out Modality m))
                    SwitchModality(m);
                else
                    AnsiConsole.MarkupLine($"[red]Unknown mode '{Markup.Escape(arg)}'.[/] Options: {string.Join(", ", Modalities.All.Select(Modalities.ToCliName))}.");
                break;
            case "model":
                _model = arg;
                AnsiConsole.MarkupLine($"[grey]model →[/] [{CliTheme.Accent}]{Markup.Escape(arg.Length == 0 ? "(default)" : arg)}[/]");
                break;
            case "backend":
                SetBackend(arg);
                break;
            case "output":
                _outputDir = arg.Length == 0 ? null : arg;
                AnsiConsole.MarkupLine($"[grey]output →[/] [{CliTheme.Accent}]{Markup.Escape(_outputDir ?? "(default)")}[/]");
                break;
            case "set":
                SetParam(arg);
                break;
            case "show":
                ShowState();
                break;
            case "reset":
                _params.Reset();
                AnsiConsole.MarkupLine("[grey]parameters reset to defaults.[/]");
                break;
            case "list":
                CatalogView.Render(Modalities.TryParse(arg, out Modality lf) ? lf : null, verifiedOnly: false);
                break;
            case "models":
                CacheView.Render();
                break;
            case "preview":
                RenderPreview(arg);
                break;
            case "clear":
                AnsiConsole.Clear();
                break;
            default:
                AnsiConsole.MarkupLine($"[red]Unknown command '/{Markup.Escape(cmd)}'.[/] Try [{CliTheme.Accent}]/help[/].");
                break;
        }
        return true;
    }

    private void SwitchModality(Modality modality)
    {
        _modality = modality;
        _params = new ParamState(modality);
        _model = "";
        AnsiConsole.MarkupLine($"[grey]mode →[/] [{CliTheme.Accent}]{Modalities.ToCliName(modality)}[/]" +
            (_dispatch.IsSupported(modality) ? "" : " [yellow](not wired yet)[/]"));
    }

    private void SetBackend(string selector)
    {
        if (selector.Length == 0)
        {
            AnsiConsole.MarkupLine($"[grey]backend is[/] [{CliTheme.Accent}]{Markup.Escape(BackendFactory.Describe(_backendSelector))}[/]");
            return;
        }
        if (!BackendFactory.ValidSelectors.Contains(selector, StringComparer.OrdinalIgnoreCase))
        {
            AnsiConsole.MarkupLine($"[red]Unknown backend '{Markup.Escape(selector)}'.[/] Valid: {string.Join(", ", BackendFactory.ValidSelectors)}.");
            return;
        }
        _backendSelector = selector.ToLowerInvariant();
        DisposeLoaded();
        AnsiConsole.MarkupLine($"[grey]backend →[/] [{CliTheme.Accent}]{Markup.Escape(BackendFactory.Describe(_backendSelector))}[/]");
    }

    private void SetParam(string arg)
    {
        string[] kv = arg.Split(' ', 2, StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries);
        if (kv.Length < 2)
        {
            AnsiConsole.MarkupLine("[red]Usage:[/] /set <key> <value>");
            return;
        }
        if (_params.TrySet(kv[0], kv[1]))
            AnsiConsole.MarkupLine($"[grey]{Markup.Escape(kv[0])} →[/] [{CliTheme.Accent}]{Markup.Escape(kv[1])}[/]");
        else
            AnsiConsole.MarkupLine($"[red]'{Markup.Escape(kv[0])}' is not a parameter for {Modalities.ToCliName(_modality)}.[/] Try [{CliTheme.Accent}]/show[/].");
    }

    private static void RenderPreview(string arg)
    {
        string path = arg.Trim().Trim('"');
        if (path.Length == 0)
        {
            AnsiConsole.MarkupLine("[red]Usage:[/] /preview <image.png>");
            return;
        }
        if (!File.Exists(path))
        {
            AnsiConsole.MarkupLine($"[red]Image not found:[/] {Markup.Escape(path)}");
            return;
        }
        try
        {
            (byte[] rgb, int width, int height) = path.EndsWith(".bmp", StringComparison.OrdinalIgnoreCase)
                ? BmpEncoder.Decode(File.ReadAllBytes(path))
                : PngDecoder.DecodeFromFile(path);
            AnsiConsole.WriteLine();
            TerminalImage.Render(rgb, width, height);
            AnsiConsole.MarkupLine($"[grey]{width}x{height} · {Markup.Escape(Path.GetFileName(path))}[/]");
        }
        catch (Exception ex)
        {
            AnsiConsole.MarkupLine($"[red]Could not preview:[/] {Markup.Escape(ex.Message)}");
        }
    }

    private void ShowState()
    {
        Table table = new Table().Border(TableBorder.Rounded).BorderColor(Color.Grey);
        table.AddColumn("[grey]setting[/]");
        table.AddColumn($"[{CliTheme.Accent}]value[/]");
        table.AddRow("mode", Modalities.ToCliName(_modality));
        table.AddRow("model", _model.Length == 0 ? "(default)" : Markup.Escape(_model));
        table.AddRow("backend", Markup.Escape(BackendFactory.Describe(_backendSelector)));
        table.AddRow("output", Markup.Escape(_outputDir ?? "(default)"));
        foreach (KeyValuePair<string, string> kv in _params.Values)
            table.AddRow(Markup.Escape(kv.Key), Markup.Escape(kv.Value));
        AnsiConsole.Write(table);
    }

    private void Generate(string prompt)
    {
        if (!_dispatch.IsSupported(_modality))
        {
            AnsiConsole.MarkupLine($"[yellow]The '{Modalities.ToCliName(_modality)}' modality is not wired yet.[/]");
            return;
        }

        try
        {
            ModelSpec spec = ModelResolver.Resolve(_model, null, _modality);
            IBackend backend = EnsureBackend();
            IModalityRunner runner = GetOrLoadRunner(spec, backend);

            IProgressSink sink = new ConsoleProgressSink();
            if (_modality == Modality.Text)
                AnsiConsole.Write(new Rule($"[{CliTheme.Accent}]response[/]").LeftJustified().RuleStyle("grey"));

            GeneratedArtifact artifact = _dispatch.Get(_modality).Run(runner, prompt, _params, sink, CancellationToken.None);
            ResultPresenter.Present(artifact, quiet: false);

            string? saved = ArtifactWriter.Write(artifact, _outputDir, prompt, force: _outputDir is not null);
            if (saved is not null)
                AnsiConsole.MarkupLine($"[grey]saved[/] [{CliTheme.Accent}]{Markup.Escape(saved)}[/]");
        }
        catch (Exception ex)
        {
            CliErrors.Report(ex, _modality);
        }
        AnsiConsole.WriteLine();
    }

    private IModalityRunner GetOrLoadRunner(ModelSpec spec, IBackend backend)
    {
        string key = $"{_modality}:{spec.LocalPath ?? spec.Requested}";
        if (_runners.TryGetValue(key, out IModalityRunner? cached))
            return cached;

        IModalityRunner runner = _dispatch.Get(_modality).Load(spec, backend, new ConsoleProgressSink());
        _runners[key] = runner;
        return runner;
    }

    private IBackend EnsureBackend() => _backend ??= BackendFactory.Create(_backendSelector);

    private void DisposeLoaded()
    {
        foreach (IModalityRunner runner in _runners.Values)
            runner.Dispose();
        _runners.Clear();
        _backend?.Dispose();
        _backend = null;
    }

    private static void PrintHelp()
    {
        AnsiConsole.MarkupLine($"[{CliTheme.Accent}]Commands[/]");
        (string, string)[] rows =
        {
            ("<prompt>", "generate with the current mode/model/params"),
            ("/text /image /speak /music /transcribe /vision /video /3d /world", "switch mode"),
            ("/mode <name>", "switch mode explicitly"),
            ("/model <id|path>", "set the model (empty = default)"),
            ("/backend <auto|cpu|cuda|vulkan>", "set the compute backend"),
            ("/set <key> <value>", "set a generation parameter"),
            ("/output <dir>", "set the artifact output directory"),
            ("/show", "show the current state"),
            ("/reset", "reset parameters to defaults"),
            ("/list [modality]", "browse the model catalog"),
            ("/models", "show the local model cache"),
            ("/preview <image>", "display an image inline"),
            ("/clear", "clear the screen"),
            ("/quit", "exit"),
        };
        foreach ((string key, string desc) in rows)
            AnsiConsole.MarkupLine($"  [{CliTheme.Accent}]{Markup.Escape(key)}[/]  [grey]{Markup.Escape(desc)}[/]");
    }

    /// <inheritdoc/>
    public void Dispose() => DisposeLoaded();
}
