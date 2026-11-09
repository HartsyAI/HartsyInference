using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Worlds;

namespace HartsyInference.Engine.Tests.Worlds;

public sealed class WorldComponentResolverTests
{
    [Fact]
    public void Oasis_ResolvesItsBackboneAndVae()
    {
        ModelSpec spec = WorldTestFixtures.Spec(ModelCatalog.Find("oasis"), "oasis") with
        {
            Aux = new Dictionary<string, string> { [WorldComponentResolver.VaeAuxKey] = "/models/vit-l-20.safetensors" },
        };

        WorldComponents components = WorldComponentResolver.Resolve(spec);

        Assert.Equal(WorldArchitecture.Oasis, components.Architecture);
        Assert.Equal(spec.LocalPath, components.Backbone);
        Assert.Equal("/models/vit-l-20.safetensors", components.Vae);
        Assert.Null(components.TextEncoder);
        Assert.Null(components.ClipEncoder);
    }

    [Fact]
    public void Oasis_MissingVaePath_NamesTheAuxKey()
    {
        ModelSpec spec = WorldTestFixtures.Spec(ModelCatalog.Find("oasis"), "oasis");

        ArgumentException ex = Assert.Throws<ArgumentException>(() => WorldComponentResolver.Resolve(spec));

        Assert.Contains("ViT-VAE", ex.Message);
        Assert.Contains("'vae-path'", ex.Message);
    }

    [Fact]
    public void GameCraft_NamesEachMissingCompanionInItsOwnOrder()
    {
        ModelSpec bare = WorldTestFixtures.Spec(ModelCatalog.Find("hunyuan-gamecraft"), "hunyuan-gamecraft");
        Assert.Contains("3D-VAE", Assert.Throws<ArgumentException>(() => WorldComponentResolver.Resolve(bare)).Message);

        ModelSpec withVae = bare with { Aux = new Dictionary<string, string> { [WorldComponentResolver.VaeAuxKey] = "vae" } };
        Assert.Contains("Llava-Llama-3-8B", Assert.Throws<ArgumentException>(() => WorldComponentResolver.Resolve(withVae)).Message);

        ModelSpec withLlava = withVae with
        {
            Aux = new Dictionary<string, string>
            {
                [WorldComponentResolver.VaeAuxKey] = "vae",
                [WorldComponentResolver.LlavaAuxKey] = "llava",
            },
        };
        Assert.Contains("CLIP-L", Assert.Throws<ArgumentException>(() => WorldComponentResolver.Resolve(withLlava)).Message);

        ModelSpec full = withLlava with
        {
            Aux = new Dictionary<string, string>
            {
                [WorldComponentResolver.VaeAuxKey] = "vae",
                [WorldComponentResolver.LlavaAuxKey] = "llava",
                [WorldComponentResolver.ClipAuxKey] = "clip",
            },
        };
        WorldComponents components = WorldComponentResolver.Resolve(full);
        Assert.Equal(WorldArchitecture.HunyuanGameCraft, components.Architecture);
        Assert.Equal("vae", components.Vae);
        Assert.Equal("llava", components.TextEncoder);
        Assert.Equal("clip", components.ClipEncoder);
    }

    [Fact]
    public void Diamond_NeedsOnlyItsBackbone()
    {
        ModelSpec diamond = WorldTestFixtures.Spec(ModelCatalog.Find("diamond"), "diamond");

        WorldComponents components = WorldComponentResolver.Resolve(diamond);

        Assert.Equal(WorldArchitecture.Diamond, components.Architecture);
        Assert.Null(components.Vae);
    }

    [Theory]
    [InlineData("oasis", WorldArchitecture.Oasis)]
    [InlineData("DIAMOND", WorldArchitecture.Diamond)]
    [InlineData("Hunyuan-GameCraft", WorldArchitecture.HunyuanGameCraft)]
    [InlineData("sana-wm", WorldArchitecture.SanaWm)]
    public void ArchitectureOf_MapsEachKnownId(string catalogId, WorldArchitecture expected)
    {
        Assert.Equal(expected, WorldComponentResolver.ArchitectureOf(catalogId));
    }

    [Theory]
    [InlineData("some-future-world")]
    [InlineData(null)]
    public void ArchitectureOf_RefusesIdsItDoesNotKnow_InsteadOfFallingBackToOasis(string? catalogId)
    {
        NotSupportedException ex = Assert.Throws<NotSupportedException>(() => WorldComponentResolver.ArchitectureOf(catalogId));

        Assert.Contains("is not a world model this engine can load", ex.Message);
        Assert.Contains("sana-wm", ex.Message);
    }

    [Fact]
    public void NoLocalPath_IsFileNotFound()
    {
        ModelSpec spec = WorldTestFixtures.Spec(ModelCatalog.Find("diamond"), "diamond") with { LocalPath = null };

        Assert.Throws<FileNotFoundException>(() => WorldComponentResolver.Resolve(spec));
    }
}
