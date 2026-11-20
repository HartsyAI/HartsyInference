using Xunit;
using HartsyInference.ModelAssets.Metadata;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>The prompt block's vocabulary checks and the flat <c>modelspec.prompt_*</c> header keys it becomes.</summary>
public sealed class PromptProfileMetadataTests
{
    private static readonly PromptProfile _ideogram = new()
    {
        Kind = "image", Encoders = ["vlm", "clip"], Format = "json", Negative = "none",
        Supports = ["boxes", "text-rendering"], Boxes = "yxyx-1000", Guide = "docs/prompting/ideogram-4.md",
    };

    private static ArtifactIdentity Identity(PromptProfile? prompt) => new()
    {
        EngineId = "ideogram4", SwarmClassId = "ideogram-4", DisplayName = "Ideogram 4", Author = "Ideogram", License = "other",
        Tags = ["image", "ideogram4"], Prompt = prompt,
    };

    private static ArtifactProvenance Provenance(string component) => new() { Converter = "test", Component = component };

    [Fact]
    public void TheFlatKeysCarryEachSetFieldAsAString()
    {
        IReadOnlyDictionary<string, string> keys = _ideogram.ToModelSpecKeys();

        Assert.Equal("json", keys["modelspec.prompt_format"]);
        Assert.Equal("none", keys["modelspec.prompt_negative"]);
        Assert.Equal("boxes,text-rendering", keys["modelspec.prompt_supports"]);
        Assert.Equal("yxyx-1000", keys["modelspec.prompt_boxes"]);
        Assert.Equal("vlm,clip", keys["modelspec.prompt_encoder"]);
        Assert.Equal(5, keys.Count);
    }

    [Fact]
    public void UnsetFieldsWriteNoKeyAndAnEmptySupportsListWritesAnEmptyValue()
    {
        PromptProfile sparse = new() { Kind = "video", Supports = [] };

        IReadOnlyDictionary<string, string> keys = sparse.ToModelSpecKeys();

        Assert.Equal(["modelspec.prompt_supports"], keys.Keys);
        Assert.Equal("", keys["modelspec.prompt_supports"]);
        Assert.Empty(new PromptProfile { Kind = "image" }.ToModelSpecKeys());
    }

    [Fact]
    public void ThePrimaryWeightsGetTheKeysAndACompanionDoesNot()
    {
        Dictionary<string, string> primary = ArtifactMetadata.ForRepack(Identity(_ideogram), Provenance(ArtifactProvenance.MainComponent));
        Dictionary<string, string> companion = ArtifactMetadata.ForRepack(Identity(_ideogram), Provenance("text-encoder"));

        Assert.Equal("yxyx-1000", primary["modelspec.prompt_boxes"]);
        Assert.DoesNotContain(companion.Keys, key => key.StartsWith("modelspec.prompt_", StringComparison.Ordinal));
    }

    [Fact]
    public void AnIdentityWithoutAProfileWritesNoPromptKeys()
    {
        Dictionary<string, string> metadata = ArtifactMetadata.ForRepack(Identity(null), Provenance(ArtifactProvenance.MainComponent));

        Assert.DoesNotContain(metadata.Keys, key => key.StartsWith("modelspec.prompt_", StringComparison.Ordinal));
    }

    [Fact]
    public void AValidProfileHasNoProblems() => Assert.Empty(_ideogram.Problems());

    [Theory]
    [InlineData("kind", "audio")]
    [InlineData("format", "yaml")]
    [InlineData("negative", "optional")]
    public void AValueOutsideTheVocabularyIsNamed(string field, string value)
    {
        PromptProfile bad = field switch
        {
            "kind" => _ideogram with { Kind = value },
            "format" => _ideogram with { Format = value },
            _ => _ideogram with { Negative = value },
        };

        Assert.Contains(bad.Problems(), problem => problem.StartsWith(field + " ", StringComparison.Ordinal));
    }

    [Fact]
    public void ListsAreCheckedItemByItemAndForDuplicates()
    {
        PromptProfile bad = _ideogram with { Encoders = ["t5", "t5", "bert"], Supports = ["boxes", "Boxes"] };

        string[] problems = [.. bad.Problems()];

        Assert.Contains("encoders 'bert' is not one of clip|t5|llm|vlm.", problems);
        Assert.Contains("encoders lists 't5' twice.", problems);
        Assert.Contains(problems, problem => problem.StartsWith("supports 'Boxes'", StringComparison.Ordinal));
    }

    [Theory]
    [InlineData("1000")]
    [InlineData("zyxx-1000")]
    [InlineData("yxyx-0")]
    [InlineData("yxyx-abc")]
    public void ABoxFormMustBeAxesThenAPositiveScale(string boxes)
    {
        Assert.NotEmpty((_ideogram with { Boxes = boxes }).Problems());
    }

    [Fact]
    public void ABoxFormWithoutBoxSupportIsAProblem()
    {
        PromptProfile bad = _ideogram with { Supports = ["text-rendering"] };

        Assert.Contains("boxes is set but supports does not list 'boxes'.", bad.Problems());
    }

    [Fact]
    public void ANonPositiveTokenLimitIsAProblem()
    {
        Assert.Contains("maxTokens must be positive.", (_ideogram with { MaxTokens = 0 }).Problems());
    }
}
