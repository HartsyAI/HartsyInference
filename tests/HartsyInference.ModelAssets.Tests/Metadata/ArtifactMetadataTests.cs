using HartsyInference.ModelAssets.Metadata;
using Xunit;

namespace HartsyInference.ModelAssets.Tests.Metadata;

/// <summary>The metadata builder's contract: the keys SwarmUI classifies on, and the resolution rule that decides
/// whether a stamped model keeps its class or loses its parameters.</summary>
public sealed class ArtifactMetadataTests
{
    private static readonly ArtifactProvenance Repack =
        new() { Converter = "test", Component = ArtifactProvenance.MainComponent };

    [Fact]
    public void Build_EmitsTheKeysSwarmClassifiesOn()
    {
        ArtifactIdentity identity = ModelIdentityCatalog.All["krea2"];
        Dictionary<string, string> metadata = ArtifactMetadata.WithoutHash(identity, Repack);

        Assert.Equal("krea-2", metadata["modelspec.architecture"]);
        Assert.Equal(ArtifactMetadata.SpecVersion, metadata["modelspec.sai_model_spec"]);
        Assert.Equal("Krea 2", metadata["modelspec.title"]);
        Assert.Equal("Krea AI", metadata["modelspec.author"]);
        Assert.Equal("other", metadata["modelspec.license"]);
        Assert.Equal(ArtifactMetadata.Implementation, metadata["modelspec.implementation"]);
        Assert.Equal("krea2", metadata["hartsy.engine_id"]);
    }

    /// <summary>An image model carries the class's own standard. Anything else — including a value copied out of the
    /// source file — makes SwarmUI clone the class with its matcher disabled.</summary>
    [Fact]
    public void Build_ImageCarriesTheClassStandardResolution()
    {
        Dictionary<string, string> metadata = ArtifactMetadata.WithoutHash(ModelIdentityCatalog.All["zimage"], Repack);
        Assert.Equal("1024x1024", metadata["modelspec.resolution"]);
    }

    /// <summary>Qwen-Image 2.1 declares 1024, not v1's 1328. Stamping the family's "obvious" resolution would match
    /// the architecture id and disagree on size, which is the silent-breakage case.</summary>
    [Fact]
    public void Build_QwenImage21UsesItsOwnStandardNotV1s()
    {
        Assert.Equal("1328x1328",
            ArtifactMetadata.WithoutHash(ModelIdentityCatalog.All["qwen-image"], Repack)["modelspec.resolution"]);
        Assert.Equal("1024x1024",
            ArtifactMetadata.WithoutHash(ModelIdentityCatalog.All["qwen-image-2.1"], Repack)["modelspec.resolution"]);
    }

    /// <summary>The rule that motivated the whole field being nullable: audio classes declare no standard size.</summary>
    [Fact]
    public void Build_AudioEmitsNoResolutionAtAll()
    {
        Dictionary<string, string> metadata = ArtifactMetadata.WithoutHash(ModelIdentityCatalog.All["kokoro"], Repack);
        Assert.False(metadata.ContainsKey("modelspec.resolution"));
    }

    /// <summary>Negative control for the assertion above: the same call on an image row does produce the key, so a
    /// builder that silently stopped emitting resolutions could not pass both tests.</summary>
    [Fact]
    public void Build_ResolutionAssertionIsNotVacuous()
    {
        Assert.True(ArtifactMetadata.WithoutHash(ModelIdentityCatalog.All["krea2"], Repack)
            .ContainsKey("modelspec.resolution"));
    }

    [Fact]
    public void Build_PrecisionAppearsInTheTitleAndItsOwnKey()
    {
        ArtifactProvenance provenance = new()
        {
            Converter = "test", Component = ArtifactProvenance.MainComponent, Precision = "Q4_K_M",
        };
        Dictionary<string, string> metadata =
            ArtifactMetadata.WithoutHash(ModelIdentityCatalog.All["krea2"], provenance);

        Assert.Equal("Krea 2 (Q4_K_M)", metadata["modelspec.title"]);
        Assert.Equal("Q4_K_M", metadata["hartsy.precision"]);
    }

    [Fact]
    public void Build_ProvenanceSourceOverridesTheCatalogUpstreamRepo()
    {
        ArtifactProvenance provenance = new()
        {
            Converter = "test", Component = ArtifactProvenance.MainComponent,
            SourceRepo = "someone/their-repack", SourceFile = "kokoro-v1_0.pth", SourceSha256 = "abc123",
        };
        Dictionary<string, string> metadata =
            ArtifactMetadata.WithoutHash(ModelIdentityCatalog.All["kokoro"], provenance);

        Assert.Equal("someone/their-repack", metadata["hartsy.source_repo"]);
        Assert.Equal("kokoro-v1_0.pth", metadata["hartsy.source_file"]);
        Assert.Equal("abc123", metadata["hartsy.source_sha256"]);
    }

    /// <summary>With no source repo given, the catalog's upstream stands in, so provenance is never simply absent.</summary>
    [Fact]
    public void Build_FallsBackToTheCatalogUpstreamRepo()
    {
        Assert.Equal("hexgrad/Kokoro-82M",
            ArtifactMetadata.WithoutHash(ModelIdentityCatalog.All["kokoro"], Repack)["hartsy.source_repo"]);
    }

    /// <summary>The repacker fills an empty hash slot from the payload it is already streaming; the empty string is
    /// the signal, so the builder must emit it rather than omit the key.</summary>
    [Fact]
    public void ForRepack_LeavesTheHashSlotEmptyForTheRepackerToFill()
    {
        Dictionary<string, string> metadata = ArtifactMetadata.ForRepack(ModelIdentityCatalog.All["kokoro"], Repack);
        Assert.True(metadata.ContainsKey(ArtifactMetadata.HashKey));
        Assert.Equal("", metadata[ArtifactMetadata.HashKey]);
    }

    /// <summary>A container the caller hashes itself gets no key, because an absent hash reads as unknown while an
    /// empty one reads as wrong.</summary>
    [Fact]
    public void WithoutHash_OmitsTheHashKeyEntirely()
    {
        Assert.False(ArtifactMetadata.WithoutHash(ModelIdentityCatalog.All["kokoro"], Repack)
            .ContainsKey(ArtifactMetadata.HashKey));
    }

    [Fact]
    public void Build_RefusesAnIdentityWithNoArchitecture()
    {
        ArtifactIdentity broken = new()
        {
            EngineId = "nameless", SwarmClassId = "  ", DisplayName = "Nameless", Author = "n/a", License = "other",
        };
        Assert.Throws<ArgumentException>(() => ArtifactMetadata.WithoutHash(broken, Repack));
    }

    /// <summary>A codec or vocoder in its own file is part of a model, not a model. Without this it would be
    /// admitted to the model list as something selectable that generates nothing.</summary>
    [Fact]
    public void Build_ComponentCarriesNoArchitecture()
    {
        ArtifactProvenance component = new() { Converter = "test", Component = "codec" };
        Dictionary<string, string> metadata = ArtifactMetadata.WithoutHash(ModelIdentityCatalog.All["yue"], component);

        Assert.False(metadata.ContainsKey("modelspec.architecture"));
        Assert.False(metadata.ContainsKey("modelspec.resolution"));
        Assert.Equal("codec", metadata["hartsy.component"]);
        Assert.Equal("yue", metadata["hartsy.engine_id"]);
        Assert.Equal("YuE (codec)", metadata["modelspec.title"]);
    }

    /// <summary>Negative control: the same identity stamped as primary does carry an architecture, so the test
    /// above cannot pass on a builder that stopped emitting the key entirely.</summary>
    [Fact]
    public void Build_PrimaryDoesCarryArchitecture()
    {
        Assert.True(ArtifactMetadata.WithoutHash(ModelIdentityCatalog.All["yue"], Repack)
            .ContainsKey("modelspec.architecture"));
    }

    /// <summary>A component makes no claim about who wrote it: ContentVec and RMVPE ship inside RVC but are other
    /// people's work under other terms, and inheriting the family's author would state something false.</summary>
    [Fact]
    public void Build_ComponentClaimsNoAuthorOrLicense()
    {
        ArtifactProvenance component = new() { Converter = "test", Component = "pitch-estimator" };
        Dictionary<string, string> metadata = ArtifactMetadata.WithoutHash(ModelIdentityCatalog.All["rvc"], component);

        Assert.False(metadata.ContainsKey("modelspec.author"));
        Assert.False(metadata.ContainsKey("modelspec.license"));
    }

    [Fact]
    public void Build_RefusesProvenanceWithNoComponent()
    {
        ArtifactProvenance nameless = new() { Converter = "test", Component = " " };
        Assert.Throws<ArgumentException>(
            () => ArtifactMetadata.WithoutHash(ModelIdentityCatalog.All["kokoro"], nameless));
    }

    [Fact]
    public void Build_AudioCarriesTheIdsAudioLabAdmitsOn()
    {
        ArtifactIdentity identity = ModelIdentityCatalog.All["qwen3tts"].ForVariant("1.7B-Base");
        Dictionary<string, string> metadata = ArtifactMetadata.WithoutHash(identity, Repack with { ModelId = "1.7B-Base" });
        Assert.Equal("qwen3_tts", metadata["hartsy.provider_id"]);
        Assert.Equal("1.7B-Base", metadata["hartsy.model_id"]);
        Assert.Equal("qwen3_tts_clone", metadata["modelspec.architecture"]);
        Assert.False(ArtifactMetadata.WithoutHash(ModelIdentityCatalog.All["krea2"], Repack).ContainsKey("hartsy.provider_id"));
    }
}
