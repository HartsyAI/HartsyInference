using System.Text.RegularExpressions;
using HartsyInference.Engine.Registry;

namespace HartsyInference.Engine.Tests.Registry;

public sealed class MaxwellCatalogTests
{
    private static readonly Regex Commit = new("^[0-9a-f]{40}$", RegexOptions.Compiled);

    [Fact]
    public void Maxwell_IsACliDrivableTextRowAwaitingReferenceValidation()
    {
        CatalogEntry? entry = ModelCatalog.Find("maxwell-v1-flash");

        Assert.NotNull(entry);
        Assert.Equal(Modality.Text, entry.Modality);
        Assert.Equal(ModelStatus.ValidationPending, entry.Status);
        Assert.True(entry.CliDrivable);
        Assert.Equal("maxwell-model/Maxwell-V1-Flash", entry.HuggingFaceRepo);
    }

    [Fact]
    public void Maxwell_IsPinnedToACommitAndCarriesOneFp8Checkpoint()
    {
        CatalogVariant variant = Assert.Single(ModelCatalog.Find("maxwell-v1-flash")!.Variants);

        Assert.Matches(Commit, variant.Source.Revision!);
        Assert.Equal("maxwell-model/Maxwell-V1-Flash-FP8", variant.Source.Repo);
        Assert.Equal(["model.safetensors"], variant.Source.Files);
        Assert.Equal(38_588_885_068, variant.Source.TotalBytes);
        Assert.Contains("chat_template.jinja", variant.Source.SupportFiles);
    }
}
