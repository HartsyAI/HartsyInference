using System.ComponentModel;
using HartsyInference.Cli.Infra;
using HartsyInference.Engine.Recipes.Image;
using Spectre.Console;
using Spectre.Console.Cli;

namespace HartsyInference.Cli.Commands;

/// <summary>
/// <c>hartsy remove-people &lt;image&gt; --mask &lt;mask&gt;</c>: removes the people under the mask with the recipe in
/// <see cref="PeopleRemoval"/>. It runs <c>hartsy image</c> with the recipe's flags filled in, so the output and the Fill
/// handling are the same as <c>image</c>'s.
/// </summary>
public sealed class RemovePeopleCommand : Command<RemovePeopleCommand.Settings>
{
    public sealed class Settings : CommandSettings
    {
        [CommandArgument(0, "<image>")]
        [Description("Photo to edit (PNG or JPEG).")]
        public string Image { get; init; } = "";

        [CommandOption("--mask")]
        [Description("Mask image, white where the people are. Same size as the photo.")]
        public string? Mask { get; init; }

        [CommandOption("--precision")]
        [Description("fp8 (the default) or bf16.")]
        public string Precision { get; init; } = "fp8";

        [CommandOption("--seed")]
        [Description("Seed; -1 picks one.")]
        public int Seed { get; init; } = -1;

        [CommandOption("-b|--backend")]
        [Description("Backend: auto, cuda:0, cpu, ...")]
        public string Backend { get; init; } = "auto";

        [CommandOption("-o|--output")]
        [Description("Directory for the result. Defaults to Output/.")]
        public string? Output { get; init; }
    }

    public override int Execute(CommandContext context, Settings settings)
    {
        if (string.IsNullOrWhiteSpace(settings.Mask))
        {
            AnsiConsole.MarkupLine("[red]--mask is required: a mask that is white where the people are.[/]");
            return 2;
        }
        if (!CommandRunner.RequireFilePath(settings.Mask, "--mask", out int exitCode))
        {
            return exitCode;
        }
        try
        {
            PeopleRemoval.ModelId(settings.Precision);
        }
        catch (ArgumentException ex)
        {
            AnsiConsole.MarkupLine($"[red]{Markup.Escape(ex.Message)}[/]");
            return 2;
        }

        // The request runs at the photo's size: the recipe default (1024²) would stretch a 768×1024 photo.
        (_, int width, int height) = ImageIo.DecodeFile(settings.Image);
        (_, int maskWidth, int maskHeight) = ImageIo.DecodeFile(settings.Mask);
        if (maskWidth != width || maskHeight != height)
        {
            AnsiConsole.MarkupLine($"[red]The mask is {maskWidth}×{maskHeight} but the photo is {width}×{height}. They must match.[/]");
            return 2;
        }
        return new ImageCommand().Execute(context, ToImageSettings(settings, width, height));
    }

    /// <summary>The <c>image</c> settings this command runs: the recipe's flags, the photo's size, and the caller's photo, mask, seed and output.</summary>
    internal static ImageCommand.Settings ToImageSettings(Settings settings, int width, int height) => new()
    {
        Prompt = PeopleRemoval.Prompt,
        Model = PeopleRemoval.ModelId(settings.Precision),
        Negative = PeopleRemoval.NegativePrompt,
        TrueCfg = PeopleRemoval.TrueCfgScale,
        FluxGuidance = PeopleRemoval.FluxGuidance,
        Steps = PeopleRemoval.Steps,
        Creativity = PeopleRemoval.FillCreativity,
        Loras = [PeopleRemoval.LoraId],
        LoraWeights = [PeopleRemoval.LoraWeight],
        Width = width,
        Height = height,
        InitImage = settings.Image,
        Mask = settings.Mask,
        Seed = settings.Seed,
        Backend = settings.Backend,
        Output = settings.Output,
    };
}
