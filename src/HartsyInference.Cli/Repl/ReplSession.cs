using HartsyInference.Cli.Commands;
using HartsyInference.Cli.Dispatch;
using HartsyInference.Cli.Infra;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using Spectre.Console;

namespace HartsyInference.Cli.Repl;

/// <summary>The interactive <c>hartsy</c> session: persistent parameter state, slash commands, and a per-session cache of loaded models.</summary>
/// <remarks>The cache lets successive generations reuse the same backend and weights.</remarks>
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
        new("convert", "switch to voice conversion"),
        new("fx", "switch to audio effects (separate/enhance)"),
        new("model", "pick a model (or set id/path)"),
        new("params", "set generation parameters interactively"),
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

    private readonly InferenceEngine _engine = new InferenceEngine();

    private Modality _modality = Modality.Text;
    private ParamState _params = new ParamState(Modality.Text);
    private string _model = "";
    private string? _outputDir;

    /// <summary>Runs the read-eval-print loop until the user exits (or EOF). Always returns 0.</summary>
    public int Run()
    {
        CliTheme.RenderBanner(_engine.BackendSelector);
        AnsiConsole.MarkupLine($"[#9aa4af]Type a prompt to generate, or[/] [{CliTheme.Accent}]/help[/] [#9aa4af]for commands.[/] [#9aa4af]Ctrl+C or[/] [{CliTheme.Accent}]/quit[/] [#9aa4af]to exit.[/]");
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

        AnsiConsole.MarkupLine("[#9aa4af]bye.[/]");
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
                if (arg.Length == 0)
                {
                    PickModel();
                }
                else
                {
                    _model = arg;
                    AnsiConsole.MarkupLine($"[#9aa4af]model →[/] [{CliTheme.Accent}]{Markup.Escape(arg)}[/]");
                }
                break;
            case "params":
                EditParams();
                break;
            case "backend":
                SetBackend(arg);
                break;
            case "output":
                _outputDir = arg.Length == 0 ? null : arg;
                AnsiConsole.MarkupLine($"[#9aa4af]output →[/] [{CliTheme.Accent}]{Markup.Escape(_outputDir ?? "(default)")}[/]");
                break;
            case "set":
                SetParam(arg);
                break;
            case "show":
                ShowState();
                break;
            case "reset":
                _params.Reset();
                AnsiConsole.MarkupLine("[#9aa4af]parameters reset to defaults.[/]");
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
        AnsiConsole.MarkupLine($"[#9aa4af]mode →[/] [{CliTheme.Accent}]{Modalities.ToCliName(modality)}[/]" +
            (_engine.IsSupported(modality) ? "" : " [yellow](not wired yet)[/]"));
    }

    private void SetBackend(string selector)
    {
        if (selector.Length == 0)
        {
            AnsiConsole.MarkupLine($"[#9aa4af]backend is[/] [{CliTheme.Accent}]{Markup.Escape(_engine.BackendDescription)}[/]");
            return;
        }
        if (!BackendFactory.IsValidSelector(selector))
        {
            AnsiConsole.MarkupLine($"[red]Unknown backend '{Markup.Escape(selector)}'.[/] " +
                $"Valid: {string.Join(", ", BackendFactory.ValidSelectors)} (device backends also accept ':{{ordinal}}', e.g. cuda:1).");
            return;
        }
        _engine.SetBackend(selector.ToLowerInvariant());
        AnsiConsole.MarkupLine($"[#9aa4af]backend →[/] [{CliTheme.Accent}]{Markup.Escape(_engine.BackendDescription)}[/]");
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
            AnsiConsole.MarkupLine($"[#9aa4af]{Markup.Escape(kv[0])} →[/] [{CliTheme.Accent}]{Markup.Escape(kv[1])}[/]");
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
            (byte[] rgb, int width, int height) = ImageIo.DecodeFile(path);
            AnsiConsole.WriteLine();
            TerminalImage.Render(rgb, width, height);
            AnsiConsole.MarkupLine($"[#9aa4af]{width}x{height} · {Markup.Escape(Path.GetFileName(path))}[/]");
        }
        catch (Exception ex)
        {
            Logs.Warning($"Could not preview '{path}': {ex.Message}");
            AnsiConsole.MarkupLine($"[red]Could not preview:[/] {Markup.Escape(ex.Message)}");
        }
    }

    // Transcribe (Whisper) and Speak (Piper) download a sensible default on first use, so they don't need a local pick.
    private static bool NeedsModelSelection(Modality modality) =>
        modality is not (Modality.Transcribe or Modality.Speech);

    /// <summary>Scans the models folder for the current modality and lets the user pick one (or enter a path/id), then walk its parameters.</summary>
    /// <remarks>Returns false when nothing is chosen (empty folder or cancelled).</remarks>
    private bool PickModel()
    {
        IReadOnlyList<DiscoveredModel> models = LocalModelScanner.Scan(_modality);
        if (models.Count == 0)
        {
            string folders = string.Join(", ", LocalModelScanner.SearchedFolders(_modality));
            AnsiConsole.MarkupLine(
                $"[yellow]No {Modalities.ToCliName(_modality)} models found[/] under [{CliTheme.Accent}]{Markup.Escape(RepoPaths.ModelsRoot())}[/] " +
                $"[{CliTheme.Muted}]({Markup.Escape(folders)})[/].");
            AnsiConsole.MarkupLine($"[{CliTheme.Muted}]Set one explicitly with[/] [{CliTheme.Accent}]/model <path|id>[/][{CliTheme.Muted}], or download with[/] [{CliTheme.Accent}]hartsy pull[/][{CliTheme.Muted}].[/]");
            return false;
        }

        List<string> choices = models.Select(m => m.IsDirectory ? $"{m.Label}/" : $"{m.Label}   {CliTheme.FormatBytes(m.SizeBytes)}").ToList();
        choices.Add("✎ enter a path or model id manually…");

        string name = Modalities.ToCliName(_modality);
        string article = "aeiou".Contains(char.ToLowerInvariant(name[0])) ? "an" : "a";
        int idx = InteractivePrompt.SelectIndex($"Select {article} {name} model", choices);
        if (idx < 0)
        {
            AnsiConsole.MarkupLine($"[{CliTheme.Muted}]cancelled.[/]");
            return false;
        }

        if (idx == models.Count)
        {
            string? manual = InteractivePrompt.Ask("model path or id");
            if (string.IsNullOrWhiteSpace(manual))
                return false;
            _model = manual.Trim();
        }
        else
        {
            _model = models[idx].Path;
        }

        AnsiConsole.MarkupLine($"[#9aa4af]model →[/] [{CliTheme.Accent}]{Markup.Escape(_model)}[/]");
        if (_params.Values.Count > 0 && InteractivePrompt.Confirm("Set generation parameters now?", defaultYes: false))
            EditParams();
        return true;
    }

    /// <summary>Walks each tunable for the current modality, letting the user keep (Enter) or change it.</summary>
    private void EditParams()
    {
        if (_params.Values.Count == 0)
        {
            AnsiConsole.MarkupLine($"[{CliTheme.Muted}]The '{Modalities.ToCliName(_modality)}' mode has no tunable parameters.[/]");
            return;
        }

        AnsiConsole.MarkupLine($"[{CliTheme.Muted}]Enter a new value or press Enter to keep the current one.[/]");
        foreach (string key in _params.Values.Keys.ToList())
        {
            string current = _params.Get(key) ?? "";
            string? value = InteractivePrompt.Ask(key, current);
            if (value is not null && value != current)
                _params.TrySet(key, value);
        }
        ShowState();
    }

    private void ShowState()
    {
        Table table = new Table().Border(TableBorder.Rounded).BorderColor(Color.Grey);
        table.AddColumn("[#9aa4af]setting[/]");
        table.AddColumn($"[{CliTheme.Accent}]value[/]");
        table.AddRow("mode", Modalities.ToCliName(_modality));
        table.AddRow("model", _model.Length == 0 ? "(default)" : Markup.Escape(_model));
        table.AddRow("backend", Markup.Escape(_engine.BackendDescription));
        table.AddRow("output", Markup.Escape(_outputDir ?? "(default)"));
        foreach (KeyValuePair<string, string> kv in _params.Values)
            table.AddRow(Markup.Escape(kv.Key), kv.Value.Length == 0 ? "(model default)" : Markup.Escape(kv.Value));
        AnsiConsole.Write(table);
    }

    private void Generate(string prompt)
    {
        if (!_engine.IsSupported(_modality))
        {
            AnsiConsole.MarkupLine($"[yellow]The '{Modalities.ToCliName(_modality)}' modality is not wired yet.[/]");
            return;
        }

        // No model chosen yet for a modality that needs one: offer the on-disk models to pick from instead of erroring.
        if (_model.Length == 0 && NeedsModelSelection(_modality) && !PickModel())
            return;

        try
        {
            ModelSpec spec = ModelAcquisition.EnsurePresent(ModelResolver.Resolve(_model, null, _modality));
            if (_modality == Modality.Text)
                AnsiConsole.Write(new Rule($"[{CliTheme.Accent}]response[/]").LeftJustified().RuleStyle("grey"));

            GeneratedArtifact artifact = GenerationDispatch
                .RunAsync(_engine, spec, prompt, _params, _outputDir, quiet: false, CancellationToken.None)
                .GetAwaiter().GetResult();
            ResultPresenter.Present(artifact, quiet: false);

            string? saved = ArtifactWriter.Write(artifact, _outputDir, prompt, force: _outputDir is not null);
            if (saved is not null)
                AnsiConsole.MarkupLine($"[#9aa4af]saved[/] [{CliTheme.Accent}]{Markup.Escape(saved)}[/]");
        }
        catch (Exception ex)
        {
            CliErrors.Report(ex, _modality);
        }
        AnsiConsole.WriteLine();
    }

    private static void PrintHelp()
    {
        AnsiConsole.MarkupLine($"[{CliTheme.Accent}]Commands[/]");
        (string, string)[] rows =
        {
            ("<prompt>", "generate with the current mode/model/params"),
            ("/text /image /speak /music /transcribe /vision /video /3d /world /convert /fx", "switch mode"),
            ("/mode <name>", "switch mode explicitly"),
            ("/model [id|path]", "pick a model from disk, or set one explicitly"),
            ("/params", "step through and set generation parameters"),
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
            AnsiConsole.MarkupLine($"  [{CliTheme.Accent}]{Markup.Escape(key)}[/]  [#9aa4af]{Markup.Escape(desc)}[/]");
    }

    /// <inheritdoc/>
    public void Dispose() => _engine.Dispose();
}
