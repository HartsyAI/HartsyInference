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

    /// <summary>Audio classes declare no standard size. The image row next to it still emits one, so a builder that
    /// silently stopped emitting resolutions cannot pass.</summary>
    [Fact]
    public void Build_AudioEmitsNoResolutionAtAll()
    {
        Assert.False(ArtifactMetadata.WithoutHash(ModelIdentityCatalog.All["kokoro"], Repack).ContainsKey("modelspec.resolution"));
        Assert.True(ArtifactMetadata.WithoutHash(ModelIdentityCatalog.All["krea2"], Repack).ContainsKey("modelspec.resolution"));
    }

    /// <summary>With no source repo given, the catalog's upstream stands in, so provenance is never simply absent.</summary>
    [Fact]
    public void Build_FallsBackToTheCatalogUpstreamRepo()
    {
        Assert.Equal("hexgrad/Kokoro-82M",
            ArtifactMetadata.WithoutHash(ModelIdentityCatalog.All["kokoro"], Repack)["hartsy.source_repo"]);
    }

    /// <summary>The repacker fills an empty hash slot from the payload it is already streaming; the empty string is
    /// the signal, so the builder must emit it rather than omit the key. A container the caller hashes itself gets no
    /// key, because an absent hash reads as unknown while an empty one reads as wrong.</summary>
    [Fact]
    public void Hash_IsEmptyForRepackAndAbsentWhenCallerHashes()
    {
        Dictionary<string, string> forRepack = ArtifactMetadata.ForRepack(ModelIdentityCatalog.All["kokoro"], Repack);
        Assert.Equal("", forRepack[ArtifactMetadata.HashKey]);

        Assert.False(ArtifactMetadata.WithoutHash(ModelIdentityCatalog.All["kokoro"], Repack).ContainsKey(ArtifactMetadata.HashKey));
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

    /// <summary>A codec or vocoder in its own file is part of a model, not a model, so it carries no architecture.
    /// A component also makes no claim about who wrote it: ContentVec and RMVPE ship inside RVC but are other people's
    /// work under other terms, so inheriting the family's author or license would state something false.</summary>
    [Fact]
    public void Build_ComponentCarriesNoArchitectureOrAuthor()
    {
        ArtifactProvenance codec = new() { Converter = "test", Component = "codec" };
        Dictionary<string, string> metadata = ArtifactMetadata.WithoutHash(ModelIdentityCatalog.All["yue"], codec);

        Assert.False(metadata.ContainsKey("modelspec.architecture"));
        Assert.False(metadata.ContainsKey("modelspec.resolution"));
        Assert.Equal("codec", metadata["hartsy.component"]);
        Assert.Equal("yue", metadata["hartsy.engine_id"]);
        Assert.Equal("YuE (codec)", metadata["modelspec.title"]);

        ArtifactProvenance estimator = new() { Converter = "test", Component = "pitch-estimator" };
        Dictionary<string, string> rvc = ArtifactMetadata.WithoutHash(ModelIdentityCatalog.All["rvc"], estimator);
        Assert.False(rvc.ContainsKey("modelspec.author"));
        Assert.False(rvc.ContainsKey("modelspec.license"));
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
