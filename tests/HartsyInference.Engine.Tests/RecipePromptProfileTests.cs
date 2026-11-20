using System.Text.Json;
using System.Text.Json.Serialization;
using HartsyInference.Diffusion.Prompting;
using HartsyInference.Engine.Catalog;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Variants;
using HartsyInference.ModelAssets.Metadata;
using Xunit;

namespace HartsyInference.Engine.Tests;

/// <summary>The prompt block every recipe declares: it stays inside the vocabulary the SwarmUI-LLMAssistant extension reads,
/// reaches the catalog descriptors, and serializes the way the API writes it.</summary>
public sealed class RecipePromptProfileTests
{
    private static readonly JsonSerializerOptions _wire = new(JsonSerializerDefaults.Web)
    {
        Converters = { new JsonStringEnumConverter() },
    };

    private static IEnumerable<(string Family, string Kind, PromptWeightingMode Weighting, IReadOnlyList<PromptProfile?> Profiles)> EveryRecipe()
    {
        foreach (string name in RecipeRegistry.DefaultNames)
        {
            IArchitectureRecipe recipe = RecipeRegistry.Resolve(name)!;
            yield return (name, "image", recipe.PromptWeighting, ProfilesOf(recipe.Prompt, recipe.Variants, recipe.PromptFor));
        }
        foreach (string name in VideoRecipeRegistry.DefaultNames)
        {
            IVideoRecipe recipe = VideoRecipeRegistry.Resolve(name)!;
            yield return (name, "video", recipe.PromptWeighting, ProfilesOf(recipe.Prompt, recipe.Variants, recipe.PromptFor));
        }
    }

    private static IReadOnlyList<PromptProfile?> ProfilesOf(PromptProfile? family, ModelVariantCatalog? catalog,
        Func<ResolvedModelVariant?, PromptProfile?> promptFor)
    {
        List<PromptProfile?> profiles = [family, promptFor(null)];
        if (catalog is not null)
        {
            foreach (ModelVariant variant in catalog.Variants)
            {
                profiles.Add(promptFor(new ResolvedModelVariant(catalog.FamilyId, variant, ModelVariantSource.CallerHint, "test")));
            }
        }
        return profiles;
    }

    [Fact]
    public void EveryRecipeAndVariantDeclaresAValidProfileOfItsOwnKind()
    {
        foreach ((string family, string kind, _, IReadOnlyList<PromptProfile?> profiles) in EveryRecipe())
        {
            foreach (PromptProfile? profile in profiles)
            {
                Assert.True(profile is not null, $"{family} declares no prompt profile, so a client cannot even tell its kind.");
                Assert.Empty(profile.Problems());
                Assert.Equal(kind, profile.Kind);
            }
        }
    }

    [Fact]
    public void ClaimingWeightsRequiresTheEngineToApplyThemOnEveryVariant()
    {
        foreach ((string family, _, PromptWeightingMode weighting, IReadOnlyList<PromptProfile?> profiles) in EveryRecipe())
        {
            if (profiles.Any(profile => profile?.Supports?.Contains("weights") == true))
            {
                Assert.True(weighting != PromptWeightingMode.None, $"{family} claims weights but its pipeline strips them.");
            }
        }
    }

    [Fact]
    public void ABoxFormIsDeclaredOnlyWithBoxSupport()
    {
        foreach ((string family, _, _, IReadOnlyList<PromptProfile?> profiles) in EveryRecipe())
        {
            foreach (PromptProfile? profile in profiles.Where(p => p?.Boxes is not null))
            {
                Assert.True(profile!.Supports!.Contains("boxes"), family);
            }
        }
    }

    [Fact]
    public void IdeogramDeclaresJsonWithYxyxBoxesOnAThousandGrid()
    {
        PromptProfile prompt = Describe(Modality.Image, "ideogram4").Prompt!;

        Assert.Equal("json", prompt.Format);
        Assert.Equal("yxyx-1000", prompt.Boxes);
        Assert.Equal(["boxes", "text-rendering"], prompt.Supports);
        Assert.Equal("docs/prompting/ideogram-4.md", prompt.Guide);
    }

    [Fact]
    public void AFieldTheGuideDoesNotStateStaysUnset()
    {
        PromptProfile ideogram = Describe(Modality.Image, "ideogram4").Prompt!;
        PromptProfile lance = Describe(Modality.Image, "lance-image").Prompt!;
        PromptProfile fibo = Describe(Modality.Image, "fibo2").Prompt!;
        PromptProfile lumina = Describe(Modality.Image, "lumina2").Prompt!;

        Assert.Null(ideogram.Negative);
        Assert.Null(ideogram.Encoders);
        Assert.Null(lance.Format);
        Assert.Null(lance.Supports);
        Assert.Null(lance.Guide);
        Assert.Null(lumina.Negative);
        Assert.Null(fibo.Boxes);
        Assert.DoesNotContain("boxes", fibo.Supports!);
    }

    [Fact]
    public void AVariantThatTakesNoNegativeSaysSoWhileItsBaseDoesNot()
    {
        PromptProfile family = RecipePrompts.Find("zimage")!;
        PromptProfile turbo = Catalog().DescribeFamily(Modality.Image, "zimage", "turbo").Prompt!;
        PromptProfile baseBuild = Catalog().DescribeFamily(Modality.Image, "zimage", "base").Prompt!;

        Assert.Null(family.Negative);
        Assert.Equal("separate", Describe(Modality.Image, "zimage").Prompt!.Negative);
        Assert.Equal("none", turbo.Negative);
        Assert.Equal("separate", baseBuild.Negative);
        Assert.Equal(family.Supports, turbo.Supports);
    }

    [Fact]
    public void LtxDistilledLeavesTheNegativeUnsetWhileDevDeclaresIt()
    {
        PromptProfile dev = Describe(Modality.Video, "ltx-video-2").Prompt!;
        PromptProfile distilled = Describe(Modality.Video, LtxVideo2VariantsId()).Prompt!;

        Assert.Equal("separate", dev.Negative);
        Assert.Null(distilled.Negative);
        Assert.Null(RecipePrompts.Find("ltx-2")!.Negative);
        Assert.Null(RecipePrompts.Find("ltx-2.5-distilled")!.Negative);
    }

    [Fact]
    public void WanClaimsOnScreenTextOnlyForTheIdThatIsWan21Alone()
    {
        Assert.Equal(["text-rendering"], Describe(Modality.Video, "wan-21-1_3b").Prompt!.Supports);
        Assert.Empty(Describe(Modality.Video, "wan-21-14b").Prompt!.Supports!);
        Assert.Empty(Describe(Modality.Video, "wan-22-5b").Prompt!.Supports!);
        Assert.Empty(Describe(Modality.Video, "wan").Prompt!.Supports!);
    }

    [Theory]
    [InlineData("animate", "wan-animate")]
    [InlineData("animate2", "wan-animate-2")]
    [InlineData("s2v", "wan-s2v")]
    public void AWanVariantTheGuideDoesNotCoverAnswersTheSameWhicheverRouteNamesIt(string variant, string standalone)
    {
        PromptProfile viaVariant = Catalog().DescribeFamily(Modality.Video, "wan", variant).Prompt!;
        PromptProfile viaRecipe = Describe(Modality.Video, standalone).Prompt!;

        Assert.Equal(new PromptProfile { Kind = "video" }.ToModelSpecKeys(), viaVariant.ToModelSpecKeys());
        Assert.Equal(viaRecipe.Kind, viaVariant.Kind);
        Assert.Null(viaVariant.Supports);
        Assert.Null(viaVariant.Guide);
        Assert.Null(viaRecipe.Supports);
        Assert.Null(viaRecipe.Guide);
    }

    [Theory]
    [InlineData(Modality.Image, "flux2", "iris-z-small")]
    [InlineData(Modality.Image, "flux2", "iris-z-big")]
    [InlineData(Modality.Image, "flux1", "onereward")]
    [InlineData(Modality.Image, "zimage", "nitrofusion2")]
    [InlineData(Modality.Image, "mage-flow", "edit")]
    [InlineData(Modality.Image, "mage-flow", "edit-turbo")]
    [InlineData(Modality.Image, "boogu", "edit")]
    [InlineData(Modality.Image, "boogu", "edit-turbo")]
    public void AVariantNoGuideCoversKeepsOnlyWhatItsArchitectureFixes(Modality modality, string family, string variant)
    {
        PromptProfile prompt = Catalog().DescribeFamily(modality, family, variant).Prompt!;

        Assert.NotNull(prompt.Encoders);
        Assert.Null(prompt.Format);
        Assert.Null(prompt.Negative);
        Assert.Null(prompt.Supports);
        Assert.Null(prompt.Boxes);
        Assert.Null(prompt.Guide);
    }

    [Fact]
    public void EveryImageAndVideoIdentityStampsOnlyWhatItsRecipeEstablishes()
    {
        List<string> missing = [];
        foreach ((string id, ArtifactIdentity identity) in ModelIdentityCatalog.All)
        {
            if (identity.Tags.Count == 0 || identity.Tags[0] is not ("image" or "video"))
            {
                continue;
            }
            PromptProfile? prompt = RecipePrompts.Find(id);
            if (prompt is null)
            {
                missing.Add(id);
                continue;
            }
            Assert.Equal(identity.Tags[0], prompt.Kind);
            Assert.Empty(prompt.Problems());
        }
        Assert.Empty(missing);
    }

    [Fact]
    public void AFineTuneIdentityDoesNotInheritItsBaseModelsPromptFacts()
    {
        foreach (string id in new[] { "iris-z-small", "iris-z-big" })
        {
            PromptProfile prompt = RecipePrompts.Find(id)!;

            Assert.Equal([PromptProfile.EncoderKey], prompt.ToModelSpecKeys().Keys);
        }
        Assert.Null(RecipePrompts.Find("flux2")!.Negative);
    }

    [Fact]
    public void FluxSaysNoNegativeOnlyForTheBuildsTheGuideCovers()
    {
        Assert.Equal("none", NegativeOf("flux2", "klein-4b"));
        Assert.Equal("none", NegativeOf("flux2", "klein-9b"));
        Assert.Null(NegativeOf("flux2", "dev"));
        Assert.Null(RecipePrompts.Find("flux2")!.Negative);
        Assert.Equal("none", NegativeOf("flux3", "distilled"));
        Assert.Null(NegativeOf("flux3", "base"));
        Assert.Null(RecipePrompts.Find("flux3")!.Negative);
    }

    [Fact]
    public void FluxKleinClaimsNoBoxesBecauseTheGuideAppliesThemToKleinOnlyByInference()
    {
        foreach (string variant in new[] { "base", "distilled" })
        {
            PromptProfile prompt = Catalog().DescribeFamily(Modality.Image, "flux3", variant).Prompt!;

            Assert.Null(prompt.Boxes);
            Assert.DoesNotContain("boxes", prompt.Supports!);
        }
    }

    [Fact]
    public void ACheckpointThatOnlyDefaultedToAVariantAnswersWithTheFamilyLevelProfile()
    {
        ModelSpec unnamed = new() { Requested = "zimage", Modality = Modality.Image };
        ModelSpec named = new() { Requested = "zimage", Modality = Modality.Image, Variant = "turbo" };

        PromptProfile fromDefault = Catalog().Describe(unnamed).Prompt!;
        PromptProfile fromEvidence = Catalog().Describe(named).Prompt!;

        Assert.Null(fromDefault.Negative);
        Assert.Equal("text-rendering", Assert.Single(fromDefault.Supports!));
        Assert.Equal("none", fromEvidence.Negative);
        Assert.Equal("separate", Describe(Modality.Image, "zimage").Prompt!.Negative);
    }

    [Fact]
    public void EveryDescriptorCarriesItsRecipesProfile()
    {
        foreach (ModelDescriptor descriptor in Catalog().All())
        {
            Assert.NotNull(descriptor.Prompt);
            Assert.Equal(descriptor.Modality == Modality.Image ? "image" : "video", descriptor.Prompt.Kind);
        }
    }

    [Fact]
    public void TheDescriptorWritesThePromptBlockWithOnlyTheFieldsThatAreSet()
    {
        string json = JsonSerializer.Serialize(Describe(Modality.Image, "ideogram4"), _wire);
        using JsonDocument document = JsonDocument.Parse(json);
        JsonElement prompt = document.RootElement.GetProperty("prompt");

        Assert.Equal("image", prompt.GetProperty("kind").GetString());
        Assert.Equal("json", prompt.GetProperty("format").GetString());
        Assert.Equal("yxyx-1000", prompt.GetProperty("boxes").GetString());
        Assert.Equal(["boxes", "text-rendering"], prompt.GetProperty("supports").EnumerateArray().Select(e => e.GetString()));
        Assert.False(prompt.TryGetProperty("negative", out _));
        Assert.False(prompt.TryGetProperty("encoders", out _));
        Assert.False(prompt.TryGetProperty("maxTokens", out _));
    }

    [Fact]
    public void AProfileRoundTripsThroughJson()
    {
        PromptProfile original = new()
        {
            Kind = "image", Encoders = ["vlm", "clip"], Format = "text", Negative = "none",
            Supports = ["boxes", "text-rendering"], Boxes = "yxyx-1000", MaxTokens = 4096, Guide = "docs/prompting/x.md",
        };

        PromptProfile? back = JsonSerializer.Deserialize<PromptProfile>(JsonSerializer.Serialize(original, _wire), _wire);

        Assert.NotNull(back);
        Assert.Equal(original.Kind, back.Kind);
        Assert.Equal(original.Encoders, back.Encoders);
        Assert.Equal(original.Format, back.Format);
        Assert.Equal(original.Negative, back.Negative);
        Assert.Equal(original.Supports, back.Supports);
        Assert.Equal(original.Boxes, back.Boxes);
        Assert.Equal(original.MaxTokens, back.MaxTokens);
        Assert.Equal(original.Guide, back.Guide);
        Assert.Empty(back.Problems());
    }

    [Fact]
    public void AMinimalProfileOmitsEveryOptionalField()
    {
        Assert.Equal("""{"kind":"video"}""", JsonSerializer.Serialize(new PromptProfile { Kind = "video" }, _wire));
    }

    [Fact]
    public void AFamilyFoundByEngineIdSuppliesItsFamilyLevelProfile()
    {
        Assert.Equal("yxyx-1000", RecipePrompts.Find("ideogram4")!.Boxes);
        Assert.Equal("video", RecipePrompts.Find("ltx-2")!.Kind);
        Assert.Null(RecipePrompts.Find("kokoro"));
        Assert.Null(RecipePrompts.Find(" "));
    }

    private static string LtxVideo2VariantsId() => Recipes.Video.LtxVideo2Variants.DistilledFamilyId;

    private static string? NegativeOf(string family, string variant) =>
        Catalog().DescribeFamily(Modality.Image, family, variant).Prompt!.Negative;

    private static RecipeCatalog Catalog() => new();

    private static ModelDescriptor Describe(Modality modality, string familyId) => Catalog().DescribeFamily(modality, familyId);
}
