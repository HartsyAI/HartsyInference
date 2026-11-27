using System.Text.RegularExpressions;
using HartsyInference.Engine.Registry;

namespace HartsyInference.Engine.Tests.Registry;

public sealed class SageS21FlashCatalogTests
{
    private static readonly Regex Commit = new("^[0-9a-f]{40}$", RegexOptions.Compiled);

    [Fact]
    public void SageS21Flash_IsAStructuralTextRowThatIsNotCliDrivable()
    {
        CatalogEntry? entry = ModelCatalog.Find("sage-s2.1-flash");

        Assert.NotNull(entry);
        Assert.Equal(Modality.Text, entry.Modality);
        Assert.Equal(ModelStatus.Structural, entry.Status);
        Assert.False(entry.CliDrivable);
        Assert.Equal("internlm/Sage-S2.1-Flash", entry.HuggingFaceRepo);
        Assert.Empty(entry.Assets);
    }

    [Fact]
    public void SageS21Flash_PinsBothCheckpointsToCommitsAndRefusesVision()
    {
        CatalogEntry entry = ModelCatalog.Find("sage-s2.1-flash")!;
        CatalogVariant bf16 = entry.Variants.Single(v => v.Id == "bf16"), fp8 = entry.Variants.Single(v => v.Id == "fp8");

        Assert.Equal(("internlm/Sage-S2.1-Flash", "eacc61b34a1af2ecb0c764e099152c46e076169b", 120), (bf16.Source.Repo, bf16.Source.Revision, bf16.Source.FileCount));
        Assert.Equal(("internlm/Sage-S2.1-Flash-FP8", "5717cf2fb5c2a54e9c123334f78029d5eadc09cd", 62), (fp8.Source.Repo, fp8.Source.Revision, fp8.Source.FileCount));
        Assert.All(new[] { bf16, fp8 }, v =>
        {
            Assert.Matches(Commit, v.Source.Revision!);
            Assert.Contains(CatalogComponents.Vision, v.Refusals.Keys);
        });
        Assert.NotEqual(bf16.Source.TargetSubdir, fp8.Source.TargetSubdir);
    }

    [Fact]
    public void SageS21Flash_CarriesItsLicenseAndTheProjectorCaveatInTheRow()
    {
        CatalogVariant variant = ModelCatalog.Find("sage-s2.1-flash")!.Variants.First();

        Assert.Contains("MIT License", variant.Notes);
        Assert.Contains("License metadata has not yet been specified", variant.Notes);
        Assert.Contains("not shipped or ported", variant.Notes);
    }
}
