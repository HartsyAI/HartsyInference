using HartsyInference.Engine;
using HartsyInference.Engine.Registry;
using Xunit;

namespace HartsyInference.LLM.Tests;

public sealed class ClefCatalogTests
{
    [Fact]
    public void Clef_CatalogPinsCompatibleTextAndVisionAssets()
    {
        CatalogEntry entry = ModelCatalog.Find("clef")!;

        Assert.Equal(Modality.Text, entry.Modality);
        Assert.True(entry.CliDrivable);
        Assert.Equal(ModelStatus.ValidationPending, entry.Status);
        Assert.Collection(entry.Assets,
            text =>
            {
                Assert.Equal("bartowski/Cloudflare_clef-GGUF", text.Repo);
                Assert.Equal("Cloudflare_clef-IQ2_XXS.gguf", text.RepoPath);
                Assert.Equal("transformer", text.Role);
            },
            vision =>
            {
                Assert.Equal("mmproj-Cloudflare_clef-f16.gguf", vision.RepoPath);
                Assert.Equal("mmproj", vision.Role);
            });
    }
}
