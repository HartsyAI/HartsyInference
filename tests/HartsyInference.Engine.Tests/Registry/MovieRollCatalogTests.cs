using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Registry;

namespace HartsyInference.Engine.Tests.Registry;

public sealed class MovieRollCatalogTests
{
    [Fact]
    public void MovieRoll_IsAVideoRowAwaitingValidationAndDrivableFromTheCli()
    {
        CatalogEntry? entry = ModelCatalog.Find("movieroll");

        Assert.NotNull(entry);
        Assert.Equal(Modality.Video, entry.Modality);
        Assert.Equal(ModelStatus.ValidationPending, entry.Status);
        Assert.True(entry.CliDrivable);
    }

    [Fact]
    public void MovieRoll_CarriesOnlyTheGatedDiTAndNoPin()
    {
        CatalogEntry entry = ModelCatalog.Find("movieroll")!;

        ModelAsset asset = Assert.Single(entry.Assets);
        Assert.Equal("yunyangge/MovieRoll", asset.Repo);
        Assert.Equal("distilled/t2v/model.safetensors", asset.RepoPath);
        Assert.Equal("transformer", asset.Role);
        // The LFS object id is masked in the public listing of this gated repo, so nothing is pinned.
        Assert.Null(asset.Sha256);
    }

    [Fact]
    public void MovieRoll_ResolvesItsRecipeByFamily()
    {
        IVideoRecipe? recipe = VideoRecipeRegistry.Resolve("movieroll");

        Assert.NotNull(recipe);
        Assert.Equal("movieroll", recipe.Name);
    }

    [Fact]
    public void MovieRoll_DefaultFramesLieOnItsGridAndTheGridStartsAt17()
    {
        IVideoRecipe recipe = VideoRecipeRegistry.Resolve("movieroll")!;

        Assert.True(recipe.Constraints.Frames.Contains(recipe.Defaults.Frames));
        Assert.Equal(81, recipe.Defaults.Frames);
        Assert.Equal(17, recipe.Constraints.Frames.Normalize(1));
        Assert.Equal(17, recipe.Constraints.Frames.Offset);
    }
}
