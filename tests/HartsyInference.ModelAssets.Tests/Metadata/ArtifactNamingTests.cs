using HartsyInference.ModelAssets.Metadata;
using Xunit;

namespace HartsyInference.ModelAssets.Tests.Metadata;

/// <summary>The canonical file name: dashes inside the name, one underscore before the precision token.</summary>
public sealed class ArtifactNamingTests
{
    [Theory]
    [InlineData("krea2", "turbo", "fp8-scaled", ".safetensors", "krea2-turbo_fp8-scaled.safetensors")]
    [InlineData("zimage", null, "bf16", ".safetensors", "zimage_bf16.safetensors")]
    [InlineData("flux2", "dev", "Q4_K_M", ".gguf", "flux2-dev_Q4_K_M.gguf")]
    [InlineData("kokoro", null, "repack", "safetensors", "kokoro_repack.safetensors")]
    public void FileName_FollowsTheConvention(string engineId, string? variant, string precision, string extension,
        string expected)
    {
        Assert.Equal(expected, ArtifactNaming.FileName(engineId, variant, precision, extension));
    }

    /// <summary>A version number is part of the id, so its dot survives while separators do not.</summary>
    [Fact]
    public void FileName_KeepsVersionDotsAndCollapsesSeparators()
    {
        Assert.Equal("qwen-image-2.1_bf16.safetensors",
            ArtifactNaming.FileName("qwen-image-2.1", null, "bf16", ".safetensors"));
        Assert.Equal("ltx-2.5-distilled_fp8-scaled.safetensors",
            ArtifactNaming.FileName("LTX 2.5", "Distilled", "fp8_scaled", ".safetensors"));
    }

    /// <summary>GGUF presets keep their upstream casing and underscores; every other token is slugged, so the two
    /// spellings of fp8-scaled cannot produce two different file names for one build.</summary>
    [Theory]
    [InlineData("Q4_K_M", "Q4_K_M")]
    [InlineData("q8_0", "Q8_0")]
    [InlineData("fp8_scaled", "fp8-scaled")]
    [InlineData("fp8 scaled", "fp8-scaled")]
    [InlineData("int8-convrot", "int8-convrot")]
    [InlineData("BF16", "bf16")]
    public void PrecisionToken_NormalizesConsistently(string input, string expected)
    {
        Assert.Equal(expected, ArtifactNaming.PrecisionToken(input));
    }

    [Fact]
    public void FileName_RefusesTheInputsThatWouldProduceANamelessFile()
    {
        Assert.Throws<ArgumentException>(() => ArtifactNaming.FileName("", null, "bf16", ".safetensors"));
        Assert.Throws<ArgumentException>(() => ArtifactNaming.FileName("krea2", null, "  ", ".safetensors"));
    }

    [Fact]
    public void Slug_DropsLeadingAndTrailingSeparators()
    {
        Assert.Equal("krea-2", ArtifactNaming.Slug("  Krea 2 "));
        Assert.Equal("f-lite", ArtifactNaming.Slug("F-Lite"));
    }
}
