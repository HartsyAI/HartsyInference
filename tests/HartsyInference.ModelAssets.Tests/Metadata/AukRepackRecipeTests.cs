using System.Text.Json;
using HartsyInference.ModelAssets.Metadata;
using Xunit;

namespace HartsyInference.ModelAssets.Tests.Metadata;

/// <summary>Pins the AuK repack recipes and identity rows: the recipe fields the repacker reads, the keep-fp32 patterns
/// that protect precision-sensitive tensors, and the catalog and AudioLab identity entries a repacked file is stamped with.</summary>
public sealed class AukRepackRecipeTests
{
    private static readonly JsonDocumentOptions Lenient = new() { CommentHandling = JsonCommentHandling.Skip, AllowTrailingCommas = true };

    private static string RepackDir()
    {
        string? dir = AppContext.BaseDirectory;
        while (dir is not null && !Directory.Exists(Path.Combine(dir, "tools", "repack")))
        {
            dir = Path.GetDirectoryName(dir);
        }
        return dir is null ? throw new DirectoryNotFoundException("tools/repack not found above the test binaries.")
            : Path.Combine(dir, "tools", "repack");
    }

    private static JsonElement LoadRecipe(string name) =>
        JsonDocument.Parse(File.ReadAllText(Path.Combine(RepackDir(), "recipes", name + ".json")), Lenient).RootElement.Clone();

    private static List<string> Strings(JsonElement recipe, string field) =>
        recipe.TryGetProperty(field, out JsonElement list) ? [.. list.EnumerateArray().Select(e => e.GetString()!)] : [];

    [Theory]
    [InlineData("auk-base", "base", "tencent/AuK", "tts/tencent--AuK/auk_base.safetensors", "tencent--AuK/auk_base.safetensors")]
    [InlineData("auk-flash", "flash", "tencent/AuK-Flash", "tts/tencent--AuK-Flash/auk_flash.safetensors", "tencent--AuK-Flash/auk_flash.safetensors")]
    public void TransformerRecipesCastToBf16AndKeepSensitiveTensorsInFp32(string name, string variant, string repo, string standsIn, string source)
    {
        JsonElement recipe = LoadRecipe(name);
        Assert.Equal(name, recipe.GetProperty("name").GetString());
        Assert.Equal("auk", recipe.GetProperty("model").GetString());
        Assert.Equal(variant, recipe.GetProperty("variant").GetString());
        Assert.Equal(repo, recipe.GetProperty("source_repo").GetString());
        Assert.Equal("bf16", recipe.GetProperty("dtype").GetString());
        Assert.Equal([standsIn], Strings(recipe, "stands_in_for"));
        Assert.Equal(source, recipe.GetProperty("components")[0].GetProperty("source_file").GetString());
        Assert.Empty(Strings(recipe, "drop"));
        List<string> keep = Strings(recipe, "keep_dtype");
        foreach (string pattern in new[] { "layer_weights", "layer_scale", "*norm*", "*.bias", "*inv_freq*" })
        {
            Assert.Contains(pattern, keep);
        }
    }

    [Fact]
    public void VaeRecipeStaysFp32AndDropsOnlyTheTrainingFlow()
    {
        JsonElement recipe = LoadRecipe("auk-vae");
        Assert.Equal("auk", recipe.GetProperty("model").GetString());
        Assert.Equal("vae", recipe.GetProperty("component").GetString());
        Assert.Equal("fp32", recipe.GetProperty("dtype").GetString());
        Assert.Equal("tencent/AuK", recipe.GetProperty("source_repo").GetString());
        Assert.Equal(["tts/tencent--AuK/vae.safetensors"], Strings(recipe, "stands_in_for"));
        Assert.Equal(["flow.*"], Strings(recipe, "drop"));
        List<string> keep = Strings(recipe, "keep_dtype");
        foreach (string pattern in new[] { "global_*", "*alpha", "*beta", "*filter", "*.bias" })
        {
            Assert.Contains(pattern, keep);
        }
    }

    [Fact]
    public void NoRecipeRehostsQwenOmni()
    {
        foreach (string file in Directory.GetFiles(Path.Combine(RepackDir(), "recipes"), "*.json"))
        {
            Assert.DoesNotContain("omni", Path.GetFileName(file), StringComparison.OrdinalIgnoreCase);
        }
    }

    [Fact]
    public void CatalogRowIdentifiesAukAsATtsModelUnderMit()
    {
        ArtifactIdentity? identity = ModelIdentityCatalog.Find("auk");
        Assert.NotNull(identity);
        Assert.Equal("auk_tts", identity.SwarmClassId);
        Assert.Equal("auk_tts", identity.ProviderId);
        Assert.Equal("Tencent", identity.Author);
        Assert.Equal("mit", identity.License);
        Assert.Equal("tts", identity.Tags[0]);
        Assert.Empty(identity.VariantClassIds);
        Assert.Equal("auk_tts", identity.ForVariant("flash").SwarmClassId);
    }

    [Fact]
    public void AudioLabIdentityMirrorsTheCatalogRow()
    {
        JsonElement families = JsonDocument.Parse(File.ReadAllText(Path.Combine(RepackDir(), "audiolab_identity.json"))).RootElement.GetProperty("families");
        JsonElement family = families.GetProperty("tts/auk");
        ArtifactIdentity identity = ModelIdentityCatalog.Find("auk")!;
        Assert.Equal(identity.EngineId, family.GetProperty("engine_id").GetString());
        Assert.Equal(identity.ProviderId, family.GetProperty("provider_id").GetString());
        Assert.Equal(identity.SwarmClassId, family.GetProperty("architecture").GetString());
        Assert.Equal(identity.Author, family.GetProperty("author").GetString());
        Assert.Equal(identity.License, family.GetProperty("license").GetString());
    }
}
