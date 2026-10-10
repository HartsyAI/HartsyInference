using System.Text.Json;
using Xunit;

namespace HartsyInference.ModelAssets.Tests.Metadata;

/// <summary>Pins the AuK repack recipe fields the repacker reads, the keep-fp32 patterns that protect precision-sensitive
/// tensors, and the policy that no recipe rehosts an omni model.</summary>
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

    [Fact]
    public void TransformerRecipe_CastsToBf16AndKeepsSensitiveTensorsInFp32()
    {
        JsonElement recipe = LoadRecipe("auk-base");
        Assert.Equal("bf16", recipe.GetProperty("dtype").GetString());
        Assert.Equal(["tts/tencent--AuK/auk_base.safetensors"], Strings(recipe, "stands_in_for"));
        Assert.Equal("tencent--AuK/auk_base.safetensors", recipe.GetProperty("components")[0].GetProperty("source_file").GetString());
        Assert.Empty(Strings(recipe, "drop"));
        List<string> keep = Strings(recipe, "keep_dtype");
        foreach (string pattern in new[] { "layer_weights", "layer_scale", "*norm*", "*.bias", "*inv_freq*" })
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
}
