using HartsyInference.Engine.Variants;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>Fibo 2.0 dev (text-to-image) vs edit-turbo. Both share one architecture, so the only evidence is the file name; the
/// edit-turbo build is selected by its <c>edit</c> token and dev is the default. Both edit; edit-turbo cannot generate from text
/// alone.</summary>
public static class Fibo2Variants
{
    /// <summary>The undistilled text-to-image build (default).</summary>
    public static ModelVariant Dev { get; } = new ModelVariant
    {
        Id = "dev",
        DisplayName = "Fibo 2.0 Dev",
        FilenameTokenSets = [["fibo2", "!edit"]],
    };

    /// <summary>The edit and turbo build: edit-only, 4 steps, no guidance.</summary>
    public static ModelVariant EditTurbo { get; } = new ModelVariant
    {
        Id = "edit-turbo",
        DisplayName = "Fibo 2.0 Edit Turbo",
        HintAliases = ["edit"],
        FilenameTokenSets = [["edit"]],
    };

    /// <summary>The catalog <see cref="Fibo2Recipe"/> declares.</summary>
    public static ModelVariantCatalog Catalog { get; } = new ModelVariantCatalog("fibo2", Dev, [EditTurbo, Dev]);
}
