using System.ComponentModel;
using System.Text.Json;
using System.Text.Json.Serialization;
using HartsyInference.Cli.Infra;
using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Planning;
using HartsyInference.Engine.Requests;
using Spectre.Console;
using Spectre.Console.Cli;

namespace HartsyInference.Cli.Commands;

/// <summary>Inspects a checkpoint header and prints the resolved execution plan without constructing weights.</summary>
public sealed class InspectCommand : Command<InspectCommand.Settings>
{
    /// <summary>Options for <c>hartsy inspect</c>.</summary>
    public sealed class Settings : CommandSettings
    {
        /// <summary>Modality to inspect; video is the first header-profile implementation.</summary>
        [CommandOption("--modality")]
        [Description("Modality to inspect (currently video).")]
        public string Modality { get; init; } = "video";

        /// <summary>Catalog id, repository id, or local model token.</summary>
        [CommandOption("-m|--model")]
        public string Model { get; init; } = "";

        /// <summary>Explicit local checkpoint path.</summary>
        [CommandOption("--model-path")]
        public string? ModelPath { get; init; }

        /// <summary>Optional profile id to confirm detection.</summary>
        [CommandOption("--model-profile")]
        public string? ModelProfile { get; init; }

        /// <summary>Backend used for hardware preflight.</summary>
        [CommandOption("-b|--backend")]
        public string Backend { get; init; } = "auto";

        /// <summary>Emit the complete plan as JSON.</summary>
        [CommandOption("--json")]
        public bool Json { get; init; }
    }

    /// <inheritdoc/>
    public override int Execute(CommandContext context, Settings settings)
    {
        if (!string.Equals(settings.Modality, "video", StringComparison.OrdinalIgnoreCase))
        {
            AnsiConsole.MarkupLine($"[red]Inspect currently supports only --modality video, not '{Markup.Escape(settings.Modality)}'.[/]");
            return 1;
        }
        if (!CommandRunner.RequireModelOrPath(settings.Model, settings.ModelPath, "--model", "--model-path", out int exitCode))
        {
            return exitCode;
        }
        ModelSpec spec = ModelResolver.Resolve(settings.Model, settings.ModelPath, Modality.Video);
        if (!string.IsNullOrWhiteSpace(settings.ModelProfile))
        {
            spec = spec with { ProfileId = settings.ModelProfile };
        }
        using InferenceEngine engine = new InferenceEngine(settings.Backend);
        VideoPlan plan = engine.VideoPlanning.PlanAsync(spec, new VideoRequest { Prompt = "" }).GetAwaiter().GetResult();
        if (settings.Json)
        {
            JsonSerializerOptions options = new JsonSerializerOptions
            {
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
                WriteIndented = true,
                Converters = { new JsonStringEnumConverter(JsonNamingPolicy.CamelCase) },
            };
            Console.WriteLine(JsonSerializer.Serialize(plan, options));
        }
        else
        {
            AnsiConsole.MarkupLine($"[bold]Profile:[/] {Markup.Escape(plan.Profile.Id)} ({Markup.Escape(plan.Profile.DisplayName)})");
            AnsiConsole.MarkupLine($"[bold]Task:[/] {plan.Profile.Task}  [bold]Acceleration:[/] {plan.Profile.Acceleration}  [bold]Attention:[/] {plan.Profile.Attention}");
            VideoEffectiveSettings effective = plan.EffectiveSettings;
            string shifts = effective.FlowShift is float videoShift
                ? effective.AudioFlowShift is float audioShift ? $"{videoShift}/{audioShift}" : videoShift.ToString()
                : "family-default";
            string sampler = effective.Sampler ?? "family-default";
            AnsiConsole.MarkupLine($"[bold]Effective:[/] {effective.Width}x{effective.Height}, {effective.Frames} frames @ {effective.Fps} fps, "
                + $"{effective.Steps} steps, CFG {effective.CfgScale}, shifts {shifts}, {Markup.Escape(sampler)}");
            foreach (VideoPlanIssue issue in plan.Issues)
            {
                string color = issue.Severity switch
                {
                    VideoPlanIssueSeverity.Error => "red",
                    VideoPlanIssueSeverity.Warning => "yellow",
                    _ => "grey",
                };
                AnsiConsole.MarkupLine($"[{color}]{Markup.Escape(issue.Severity.ToString())} {Markup.Escape(issue.Code)}: {Markup.Escape(issue.Message)}[/]");
            }
        }
        return plan.IsValid ? 0 : 2;
    }
}
