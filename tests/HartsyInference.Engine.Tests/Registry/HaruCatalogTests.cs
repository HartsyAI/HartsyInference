using HartsyInference.Audio.Models.Haru;
using HartsyInference.Engine.Audio;
using HartsyInference.Engine.Registry;

namespace HartsyInference.Engine.Tests.Registry;

public sealed class HaruCatalogTests
{
    [Theory]
    [InlineData("haru")]
    [InlineData("HARU-4B")]
    public void TheDescriptorIsRegisteredUnderBothIds(string id) => Assert.Same(HaruModel.Descriptor, TtsCatalog.Resolve(id));

    [Theory]
    [InlineData("haru", HaruPrecision.Native, "")]
    [InlineData("haru-4b", HaruPrecision.Native, "")]
    [InlineData("q8", HaruPrecision.Q8_0, "q8_0")]
    [InlineData("q4", HaruPrecision.Q4_K, "q4_k")]
    [InlineData("native", HaruPrecision.Native, "")]
    public void TheVariantPicksThePrecisionAndTheCacheTag(string variant, HaruPrecision precision, string tag)
    {
        Assert.Equal(precision, HaruModel.ParsePrecision(variant));
        Assert.Equal(tag, HaruModel.Descriptor.ResolveCacheTag!(variant));
        Assert.Equal(HaruModel.Repo, HaruModel.Descriptor.ResolveRepo(variant));
    }

    [Fact]
    public void AnUnknownPrecisionIsRefusedByName() =>
        Assert.Contains("q3", Assert.Throws<ArgumentException>(() => HaruModel.ParsePrecision("q3")).Message);

    [Fact]
    public async Task TheDescriptorNamesTheThreeReleaseFilesWithTheirHashesLastTheWeights()
    {
        IReadOnlyList<HartsyInference.Audio.Cache.AudioModelFile> files = await HaruModel.Descriptor.ResolveFiles!("haru", CancellationToken.None);
        Assert.Equal(["config.json", "tokenizer.json", "model.safetensors"], files.Select(static f => f.Name));
        Assert.All(files, static f => Assert.Equal(64, f.Sha256!.Length));
    }

    [Fact]
    public void TheCatalogEntryCarriesTheSameHashesAsTheLoader()
    {
        CatalogEntry entry = ModelCatalog.All.Single(static e => e.Id == "haru");
        Dictionary<string, string?> catalog = entry.Assets.ToDictionary(static a => a.RepoPath, static a => a.Sha256);
        foreach (HartsyInference.Audio.Cache.AudioModelFile file in HaruModel.Files())
        {
            Assert.Equal(file.Sha256, catalog[file.Name]);
        }
    }

    [Fact]
    public void ReadingTextAloudIsRefusedWithAPointerToTheDuplexApi()
    {
        // The runner is private to the descriptor; the refusal is checked through the exception a TTS call would raise.
        ITtsRunner runner = (ITtsRunner)Activator.CreateInstance(
            typeof(HaruModel).GetNestedType("HaruRunner", System.Reflection.BindingFlags.NonPublic)!, [null!])!;
        NotSupportedException error = Assert.Throws<NotSupportedException>(() => runner.Synthesize(null!, null!));
        Assert.Contains("OpenDuplexAsync", error.Message);
    }
}
