using HartsyInference.Engine.Variants;

namespace HartsyInference.Engine.Recipes;

/// <summary>Checks recipe declarations against their own contract: every variant's defaults are positive, aligned and
/// within the family's constraints, and every recipe answers for its own name. The types make a missing default a
/// compile error; this catches the values a type cannot (a zero frame rate, an off-grid frame count, a shadowed name).</summary>
public static class RecipeRegistryValidator
{
    /// <summary>Pixel sizes must be a multiple of this: the smallest spatial compression any registered VAE applies.</summary>
    public const int SizeMultiple = 8;

    /// <summary>Every problem with <paramref name="recipe"/>'s declarations; empty when it is valid.</summary>
    public static IReadOnlyList<string> Problems(IArchitectureRecipe recipe)
    {
        ArgumentNullException.ThrowIfNull(recipe);
        List<string> problems = [];
        CheckName(recipe.Name, recipe.Matches, problems);
        foreach ((string label, ResolvedModelVariant? variant) in VariantsOf(recipe.Variants))
        {
            CheckImage($"{recipe.Name}{label}", recipe.DefaultsFor(variant), problems);
        }
        return problems;
    }

    /// <summary>Every problem with <paramref name="recipe"/>'s declarations; empty when it is valid.</summary>
    public static IReadOnlyList<string> Problems(IVideoRecipe recipe)
    {
        ArgumentNullException.ThrowIfNull(recipe);
        List<string> problems = [];
        CheckName(recipe.Name, recipe.Matches, problems);
        FrameGrid? grid = recipe.Constraints?.Frames;
        if (grid is null)
        {
            problems.Add($"{recipe.Name}: declares no frame grid.");
        }
        foreach ((string label, ResolvedModelVariant? variant) in VariantsOf(recipe.Variants))
        {
            CheckVideo($"{recipe.Name}{label}", recipe.DefaultsFor(variant), grid, problems);
        }
        return problems;
    }

    /// <summary>Every problem with <paramref name="recipe"/>'s declarations; empty when it is valid.</summary>
    public static IReadOnlyList<string> Problems(IMusicRecipe recipe)
    {
        ArgumentNullException.ThrowIfNull(recipe);
        List<string> problems = [];
        CheckName(recipe.Name, recipe.Matches, problems);
        CheckMusic(recipe.Name, recipe.Defaults, recipe.Constraints, problems);
        foreach (string variant in recipe.Variants)
        {
            CheckMusic($"{recipe.Name} (variant '{variant}')", recipe.DefaultsFor(variant), recipe.Constraints, problems);
        }
        return problems;
    }

    /// <summary>Throws naming every problem in <paramref name="recipes"/>, including a name an earlier recipe shadows.</summary>
    /// <exception cref="InvalidOperationException">Any recipe is invalid, or a name resolves to the wrong recipe.</exception>
    public static void ThrowIfInvalid(IReadOnlyList<IArchitectureRecipe> recipes)
    {
        ArgumentNullException.ThrowIfNull(recipes);
        List<string> problems = [.. recipes.SelectMany(Problems)];
        CheckResolution([.. recipes.Select(recipe => (recipe.Name, (Func<string, bool>)recipe.Matches))], "image", problems);
        ThrowIfAny("image", problems);
    }

    /// <summary>Throws naming every problem in <paramref name="recipes"/>, including a name an earlier recipe shadows.</summary>
    /// <exception cref="InvalidOperationException">Any recipe is invalid, or a name resolves to the wrong recipe.</exception>
    public static void ThrowIfInvalid(IReadOnlyList<IVideoRecipe> recipes)
    {
        ArgumentNullException.ThrowIfNull(recipes);
        List<string> problems = [.. recipes.SelectMany(Problems)];
        CheckResolution([.. recipes.Select(recipe => (recipe.Name, (Func<string, bool>)recipe.Matches))], "video", problems);
        ThrowIfAny("video", problems);
    }

    /// <summary>Throws naming every problem in <paramref name="recipes"/>, including a name an earlier recipe shadows.</summary>
    /// <exception cref="InvalidOperationException">Any recipe is invalid, or a name resolves to the wrong recipe.</exception>
    public static void ThrowIfInvalid(IReadOnlyList<IMusicRecipe> recipes)
    {
        ArgumentNullException.ThrowIfNull(recipes);
        List<string> problems = [.. recipes.SelectMany(Problems)];
        CheckResolution([.. recipes.Select(recipe => (recipe.Name, (Func<string, bool>)recipe.Matches))], "music", problems);
        ThrowIfAny("music", problems);
    }

    /// <summary>The family-level answer, then each declared variant resolved as if the caller had named it.</summary>
    private static IEnumerable<(string Label, ResolvedModelVariant? Variant)> VariantsOf(ModelVariantCatalog? catalog)
    {
        yield return ("", null);
        if (catalog is null)
        {
            yield break;
        }
        foreach (ModelVariant variant in catalog.Variants)
        {
            yield return ($" (variant '{variant.Id}')",
                new ResolvedModelVariant(catalog.FamilyId, variant, ModelVariantSource.CallerHint, "registry validation"));
        }
    }

    private static void CheckName(string name, Func<string, bool> matches, List<string> problems)
    {
        if (string.IsNullOrWhiteSpace(name))
        {
            problems.Add("A recipe has no name.");
        }
        else if (!matches(name))
        {
            problems.Add($"{name}: does not match its own name, so it cannot be resolved by it.");
        }
    }

    private static void CheckImage(string owner, ImageDefaults defaults, List<string> problems)
    {
        CheckRevision(owner, defaults.Revision, problems);
        CheckSampling(owner, defaults.Steps, defaults.CfgScale, problems);
        CheckSize(owner, defaults.Width, defaults.Height, problems);
        if (defaults.SigmaShift is double shift && !double.IsFinite(shift))
        {
            problems.Add($"{owner}: sigma shift {shift} is not finite.");
        }
        CheckOptionalName(owner, "sampler", defaults.Sampler, problems);
        CheckOptionalName(owner, "scheduler", defaults.Scheduler, problems);
    }

    private static void CheckVideo(string owner, VideoDefaults defaults, FrameGrid? grid, List<string> problems)
    {
        CheckRevision(owner, defaults.Revision, problems);
        CheckSampling(owner, defaults.Steps, defaults.CfgScale, problems);
        CheckSize(owner, defaults.Width, defaults.Height, problems);
        if (defaults.Fps < 1)
        {
            problems.Add($"{owner}: frame rate {defaults.Fps} is below 1.");
        }
        if (grid is not null && !grid.Contains(defaults.Frames))
        {
            problems.Add($"{owner}: default frame count {defaults.Frames} is not on the family's {grid} grid.");
        }
        if (defaults.FlowShift is float flow && !float.IsFinite(flow))
        {
            problems.Add($"{owner}: flow shift {flow} is not finite.");
        }
        if (defaults.AudioFlowShift is float audio && !float.IsFinite(audio))
        {
            problems.Add($"{owner}: audio flow shift {audio} is not finite.");
        }
        CheckOptionalName(owner, "sampler", defaults.Sampler, problems);
        CheckOptionalName(owner, "scheduler", defaults.Scheduler, problems);
    }

    private static void CheckMusic(string owner, MusicDefaults defaults, MusicConstraints constraints, List<string> problems)
    {
        CheckRevision(owner, defaults.Revision, problems);
        if (!double.IsFinite(defaults.Duration) || defaults.Duration <= 0)
        {
            problems.Add($"{owner}: duration {defaults.Duration} must be finite and above 0.");
        }
        else if (constraints.MaxDurationSeconds is double max && defaults.Duration > max)
        {
            problems.Add($"{owner}: duration {defaults.Duration}s is above the family's {max}s maximum.");
        }
        if (defaults.Steps is int steps && steps < 1)
        {
            problems.Add($"{owner}: {steps} steps is below 1.");
        }
        CheckPositiveGuidance(owner, "guidance scale", defaults.CfgScale, problems);
        CheckPositiveGuidance(owner, "LM guidance scale", defaults.LmCfgScale, problems);
        CheckFinite(owner, "shift", defaults.Shift, problems);
        CheckFinite(owner, "CFG interval start", defaults.CfgIntervalStart, problems);
        CheckFinite(owner, "CFG interval end", defaults.CfgIntervalEnd, problems);
        CheckFinite(owner, "cover strength", defaults.CoverStrength, problems);
        CheckFinite(owner, "reference start", defaults.ReferenceStartSeconds, problems);
        CheckFinite(owner, "reference end", defaults.ReferenceEndSeconds, problems);
        CheckFinite(owner, "LM temperature", defaults.LmTemperature, problems);
        CheckFinite(owner, "LM top-p", defaults.LmTopP, problems);
        CheckOptionalName(owner, "infer method", defaults.InferMethod, problems);
        CheckOptionalName(owner, "guidance type", defaults.GuidanceType, problems);
    }

    private static void CheckPositiveGuidance(string owner, string what, double? value, List<string> problems)
    {
        if (value is double scale && (!double.IsFinite(scale) || scale <= 0))
        {
            problems.Add($"{owner}: {what} {scale} must be finite and above 0.");
        }
    }

    private static void CheckFinite(string owner, string what, double? value, List<string> problems)
    {
        if (value is double number && !double.IsFinite(number))
        {
            problems.Add($"{owner}: {what} {number} is not finite.");
        }
    }

    private static void CheckRevision(string owner, int revision, List<string> problems)
    {
        if (revision < 1)
        {
            problems.Add($"{owner}: revision {revision} is below 1.");
        }
    }

    private static void CheckSampling(string owner, int steps, float cfgScale, List<string> problems)
    {
        if (steps < 1)
        {
            problems.Add($"{owner}: {steps} steps is below 1.");
        }
        if (!float.IsFinite(cfgScale) || cfgScale <= 0f)
        {
            problems.Add($"{owner}: guidance scale {cfgScale} must be finite and above 0.");
        }
    }

    private static void CheckSize(string owner, int width, int height, List<string> problems)
    {
        if (width < SizeMultiple || height < SizeMultiple || width % SizeMultiple != 0 || height % SizeMultiple != 0)
        {
            problems.Add($"{owner}: {width}x{height} is not a positive multiple of {SizeMultiple} on both sides.");
        }
    }

    private static void CheckOptionalName(string owner, string what, string? value, List<string> problems)
    {
        if (value is not null && string.IsNullOrWhiteSpace(value))
        {
            problems.Add($"{owner}: {what} is blank; leave it null to let the family choose.");
        }
    }

    /// <summary>Each name must resolve to the recipe that declares it: registry lookup takes the first match, so an
    /// earlier recipe matching a later one's name (or two recipes sharing a name) silently shadows it.</summary>
    private static void CheckResolution(IReadOnlyList<(string Name, Func<string, bool> Matches)> recipes, string kind,
        List<string> problems)
    {
        for (int i = 0; i < recipes.Count; i++)
        {
            string name = recipes[i].Name;
            for (int earlier = 0; earlier < i; earlier++)
            {
                if (recipes[earlier].Matches(name))
                {
                    problems.Add($"{name}: shadowed by the earlier {kind} recipe '{recipes[earlier].Name}', which also matches it.");
                    break;
                }
            }
        }
    }

    private static void ThrowIfAny(string kind, List<string> problems)
    {
        if (problems.Count > 0)
        {
            throw new InvalidOperationException(
                $"The built-in {kind} recipe registry is invalid:{Environment.NewLine}  " + string.Join(Environment.NewLine + "  ", problems));
        }
    }
}
