using System.Text.RegularExpressions;
using HartsyInference.Engine.Registry;

namespace HartsyInference.Engine.Tests.Registry;

public sealed class KimiK31CatalogTests
{
    private static readonly Regex Commit = new("^[0-9a-f]{40}$", RegexOptions.Compiled);

    [Fact]
    public void KimiK31_IsAStructuralTextRowThatIsNotCliDrivable()
    {
        CatalogEntry? entry = ModelCatalog.Find("kimi-k3.1-pro");

        Assert.NotNull(entry);
        Assert.Equal(Modality.Text, entry.Modality);
        Assert.Equal(ModelStatus.Structural, entry.Status);
        Assert.False(entry.CliDrivable);
        Assert.Equal("moonshotai/Kimi-K3.1-Pro", entry.HuggingFaceRepo);
        Assert.Empty(entry.Assets);
    }

    [Fact]
    public void KimiK31_IsPinnedToACommitAndRefusesTheVisionComponent()
    {
        CatalogVariant variant = Assert.Single(ModelCatalog.Find("kimi-k3.1-pro")!.Variants);

        Assert.Matches(Commit, variant.Source.Revision!);
        Assert.Equal(96, variant.Source.FileCount);
        Assert.Equal(0, (int)(variant.Components & CatalogComponents.Vision));
        Assert.Contains(CatalogComponents.Vision, variant.Refusals.Keys);
    }

    [Fact]
    public void KimiK31_CarriesItsLicenseInTheRow()
    {
        CatalogVariant variant = Assert.Single(ModelCatalog.Find("kimi-k3.1-pro")!.Variants);

        Assert.Contains("Kimi K3.1-Pro License", variant.Notes);
        Assert.Contains("US$20M", variant.Notes);
    }
}
