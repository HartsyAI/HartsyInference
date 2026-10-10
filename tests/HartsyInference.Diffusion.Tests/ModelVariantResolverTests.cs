using HartsyInference.Engine.Variants;
using HartsyInference.ModelAssets.Checkpoints;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The evidence order every family's variant rides on. A tier applied out of order picks a plausible variant
/// with no error — a file name overriding the caller, a guess overriding the weights — so each boundary is pinned.</summary>
public sealed class ModelVariantResolverTests
{
    private static readonly ModelVariant Plain = new ModelVariant { Id = "plain", DisplayName = "Plain" };

    private static readonly ModelVariant Marked = new ModelVariant
    {
        Id = "marked",
        DisplayName = "Marked",
        HintAliases = ["marked-class"],
        MetadataClassIds = ["fam-marked"],
        StructuralMarkers = ["marker"],
        FilenameTokenSets = [["marked"]],
    };

    private static readonly ModelVariant Named = new ModelVariant
    {
        Id = "named",
        DisplayName = "Named",
        HintAliases = ["named-class"],
        MetadataClassIds = ["fam-named"],
        FilenameTokenSets = [["edit"], ["alt", "!plain"]],
    };

    private static readonly ModelVariant Modules = new ModelVariant
    {
        Id = "modules",
        DisplayName = "Modules",
        StructureRequired = true,
        StructuralMatch = probe => probe.AnyKeyContains("module_"),
    };

    private static readonly ModelVariantCatalog Catalog = new ModelVariantCatalog("fam", Plain, [Modules, Marked, Named, Plain]);

    [Fact]
    public void Structure_BeatsAContradictingHint_AndRecordsIt()
    {
        ResolvedModelVariant resolved = ModelVariantResolver.Classify(Catalog, Probe(keys: ["marker"]), ["named"]);
        Assert.True(resolved.Is(Marked));
        Assert.Equal(ModelVariantSource.Structure, resolved.Source);
        Assert.Equal("named", resolved.OverriddenHint);
    }

    [Fact]
    public void Hint_BeatsMetadataAndFilename()
    {
        ResolvedModelVariant resolved = ModelVariantResolver.Classify(Catalog,
            Probe(metadata: ("modelspec.architecture", "fam-marked"), fileName: "thing_marked"), [null, "", "named-class"]);
        Assert.True(resolved.Is(Named));
        Assert.Equal(ModelVariantSource.CallerHint, resolved.Source);
        Assert.True(resolved.IsDefinitive);
    }

    [Fact]
    public void Metadata_BeatsFilename()
    {
        ResolvedModelVariant resolved = ModelVariantResolver.Classify(Catalog,
            Probe(metadata: ("modelspec.architecture", "fam-marked"), fileName: "model_edit"), []);
        Assert.True(resolved.Is(Marked));
    }

    [Theory]
    [InlineData("model_edit_fp8", "named", ModelVariantSource.Filename)]
    // Whole tokens only: a substring match would read "credit" as "edit".
    [InlineData("credit_model", "plain", ModelVariantSource.Default)]
    // An excluded token vetoes its set.
    [InlineData("model_alt_plain", "plain", ModelVariantSource.Default)]
    public void Filename_MatchesWholeTokensOnly(string fileName, string expected, ModelVariantSource source)
    {
        ResolvedModelVariant resolved = ModelVariantResolver.Classify(Catalog, Probe(fileName: fileName), []);
        Assert.Equal(expected, resolved.Id);
        Assert.Equal(source, resolved.Source);
        Assert.False(resolved.IsDefinitive);
    }

    /// <summary>A variant whose modules must be in the file cannot be conjured by a hint: that would construct it on
    /// weights that lack them.</summary>
    [Fact]
    public void StructureRequiredVariant_IgnoresHints()
    {
        Assert.True(ModelVariantResolver.Classify(Catalog, Probe(), ["modules"]).Is(Plain));
        Assert.True(ModelVariantResolver.Classify(Catalog, Probe(keys: ["blocks.0.module_x"]), []).Is(Modules));
    }

    [Fact]
    public void Catalog_RejectsAnAliasNamingTwoVariants()
    {
        ModelVariant clash = new ModelVariant { Id = "clash", DisplayName = "Clash", HintAliases = ["named-class"] };
        Assert.Throws<ArgumentException>(() => new ModelVariantCatalog("fam", Plain, [Named, clash, Plain]));
    }

    [Fact]
    public void Catalog_RejectsADefaultOutsideTheList()
    {
        Assert.Throws<ArgumentException>(() => new ModelVariantCatalog("fam", Plain, [Named]));
    }

    [Fact]
    public void CacheToken_DiffersPerVariant()
    {
        ResolvedModelVariant plain = ModelVariantResolver.Classify(Catalog, Probe(), []);
        ResolvedModelVariant named = ModelVariantResolver.Classify(Catalog, Probe(), ["named"]);
        Assert.NotEqual(plain.CacheToken, named.CacheToken);
    }

    private static CheckpointProbe Probe(string[]? keys = null, (string Key, string Value)? metadata = null, string? fileName = null)
    {
        Dictionary<string, string> meta = new Dictionary<string, string>(StringComparer.Ordinal);
        if (metadata is (string key, string value))
        {
            meta[key] = value;
        }
        return CheckpointProbe.Empty with
        {
            Keys = new HashSet<string>(keys ?? [], StringComparer.Ordinal),
            Metadata = meta,
            FileNames = fileName is null ? [] : [fileName],
        };
    }
}
