using System.Linq;
using HartsyInference.Core.Configuration;
using Xunit;

namespace HartsyInference.Core.Tests.Configuration;

/// <summary>The lookup order for a model folder: a setting first, then the catalog's spelling, then the default name.</summary>
public sealed class ModelFolderSpellingsTests : IDisposable
{
    public void Dispose() => KnobStore.ResetOverrides();

    [Fact]
    public void WithoutASetting_TheCatalogSpellingComesFirst() =>
        Assert.Equal(new[] { "LLM/nanbeige43", "llm/nanbeige43" }, ModelFolders.Spellings("LLM/nanbeige43").ToArray());

    [Fact]
    public void WithASetting_TheSettingComesFirst()
    {
        KnobStore.Set(EngineKnobs.ModelFolderLlm, "my-llms");

        Assert.Equal(new[] { "my-llms/nanbeige43", "LLM/nanbeige43" }, ModelFolders.Spellings("LLM/nanbeige43").ToArray());
    }

    [Fact]
    public void AnEmptySettingIsNoSetting()
    {
        KnobStore.Set(EngineKnobs.ModelFolderLlm, "");

        Assert.Equal(new[] { "LLM/nanbeige43", "llm/nanbeige43" }, ModelFolders.Spellings("LLM/nanbeige43").ToArray());
    }

    [Fact]
    public void WhenTheCatalogSpellingIsTheDefault_OneSpellingRemains() =>
        Assert.Equal(new[] { "VAE/x" }, ModelFolders.Spellings("VAE/x").ToArray());

    [Fact]
    public void AnUnknownFolderIsReturnedAlone() =>
        Assert.Equal(new[] { "Unknown/x" }, ModelFolders.Spellings("Unknown/x").ToArray());
}
