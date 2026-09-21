using HartsyInference.ModelAssets.Metadata;
using Xunit;

namespace HartsyInference.ModelAssets.Tests.Metadata;

/// <summary>Shape checks on the identity table. These cannot prove an id still exists in SwarmUI — that needs a live
/// scan — but they do catch the mistakes that produce a file nothing can classify.</summary>
public sealed class ModelIdentityCatalogTests
{
    [Fact]
    public void EveryRowHasTheFieldsAStampNeeds()
    {
        foreach ((string engineId, ArtifactIdentity identity) in ModelIdentityCatalog.All)
        {
            Assert.False(string.IsNullOrWhiteSpace(identity.SwarmClassId), $"{engineId} has no class id");
            Assert.False(string.IsNullOrWhiteSpace(identity.DisplayName), $"{engineId} has no display name");
            Assert.False(string.IsNullOrWhiteSpace(identity.Author), $"{engineId} has no author");
            Assert.False(string.IsNullOrWhiteSpace(identity.License), $"{engineId} has no license");
            Assert.Equal(engineId, identity.EngineId);
            Assert.NotEmpty(identity.Tags);
        }
    }

    /// <summary>Two families stamping the same class id is legitimate (LTX-2.5 and its distilled build share one),
    /// but two rows for one engine id is not — which row wins would depend on declaration order.</summary>
    [Fact]
    public void EngineIdsAreUnique()
    {
        Assert.Equal(ModelIdentityCatalog.All.Count,
            ModelIdentityCatalog.All.Keys.Distinct(StringComparer.OrdinalIgnoreCase).Count());
    }

    /// <summary>A class id is looked up lowercased by the sorter, so a stray space or slash-prefix silently misses.</summary>
    [Fact]
    public void ClassIdsAreTrimmedAndUnspaced()
    {
        foreach (ArtifactIdentity identity in ModelIdentityCatalog.All.Values)
        {
            Assert.Equal(identity.SwarmClassId.Trim(), identity.SwarmClassId);
            Assert.DoesNotContain(' ', identity.SwarmClassId);
        }
    }

    /// <summary>A resolution is either absent or a parseable <c>WxH</c>; the sorter calls int.Parse on it and any
    /// other shape throws during a model scan rather than here.</summary>
    [Fact]
    public void ResolutionsAreWellFormedWhenPresent()
    {
        foreach ((string engineId, ArtifactIdentity identity) in ModelIdentityCatalog.All)
        {
            if (identity.StandardResolution is null)
            {
                continue;
            }
            string[] parts = identity.StandardResolution.Split('x');
            Assert.True(parts.Length == 2, $"{engineId} resolution '{identity.StandardResolution}' is not WxH");
            Assert.True(int.TryParse(parts[0], out int width) && width > 0, $"{engineId} width");
            Assert.True(int.TryParse(parts[1], out int height) && height > 0, $"{engineId} height");
        }
    }

    /// <summary>Audio, voice-conversion and effects rows must declare no resolution at all — the class they name
    /// has no standard size, so any value disables its matcher.</summary>
    [Fact]
    public void AudioRowsDeclareNoResolution()
    {
        string[] audioCategories = ["stt", "tts", "music", "clone", "fx"];
        foreach ((string engineId, ArtifactIdentity identity) in ModelIdentityCatalog.All)
        {
            if (identity.Tags.Count > 0 && audioCategories.Contains(identity.Tags[0]))
            {
                Assert.True(identity.StandardResolution is null, $"{engineId} must not declare a resolution");
            }
        }
    }

    /// <summary>Negative control for the test above: image and video rows do declare one, so a table that lost every
    /// resolution could not pass both.</summary>
    [Fact]
    public void ImageAndVideoRowsDoDeclareAResolution()
    {
        ArtifactIdentity[] spatial = [.. ModelIdentityCatalog.All.Values
            .Where(x => x.Tags.Count > 0 && (x.Tags[0] == "image" || x.Tags[0] == "video"))];
        Assert.NotEmpty(spatial);
        Assert.All(spatial, x => Assert.False(string.IsNullOrWhiteSpace(x.StandardResolution)));
    }

    [Fact]
    public void Find_IsCaseInsensitiveAndReturnsNullForAnUnknownFamily()
    {
        Assert.NotNull(ModelIdentityCatalog.Find("KREA2"));
        Assert.Null(ModelIdentityCatalog.Find("no-such-model"));
        Assert.Null(ModelIdentityCatalog.Find(""));
    }
}
