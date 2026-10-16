using HartsyInference.Audio.Models.FishAudio;
using HartsyInference.Engine;
using HartsyInference.Engine.Audio;
using HartsyInference.Engine.Registry;
using HartsyInference.ModelAssets.Metadata;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Fish Audio Drama 3's engine wiring: the variant picks the build and the precision, a precision keys its own runner, and
/// the model is reachable from every catalog it belongs in.</summary>
public sealed class FishAudioDrama3ModelTests
{
    [Theory]
    [InlineData("fishaudio-drama3", FishAudioDrama3Model.Repo, FishAudioDrama3Precision.Native)]
    [InlineData("drama3", FishAudioDrama3Model.Repo, FishAudioDrama3Precision.Native)]
    [InlineData("", FishAudioDrama3Model.Repo, FishAudioDrama3Precision.Native)]
    [InlineData("fp8", FishAudioDrama3Model.Repo, FishAudioDrama3Precision.Native)]
    [InlineData("int4", FishAudioDrama3Model.Int4Repo, FishAudioDrama3Precision.Native)]
    [InlineData("GPTQ", FishAudioDrama3Model.Int4Repo, FishAudioDrama3Precision.Native)]
    [InlineData("q4", FishAudioDrama3Model.Repo, FishAudioDrama3Precision.Q4_K)]
    [InlineData("int4+q8", FishAudioDrama3Model.Int4Repo, FishAudioDrama3Precision.Q8_0)]
    [InlineData("bf16, fp8", FishAudioDrama3Model.Repo, FishAudioDrama3Precision.Bf16)]
    [InlineData("someone/drama-3-bf16+bf16", "someone/drama-3-bf16", FishAudioDrama3Precision.Bf16)]
    public void Parse_PicksTheBuildAndThePrecision(string variant, string repo, FishAudioDrama3Precision precision)
    {
        FishAudioDrama3Variant parsed = FishAudioDrama3Variant.Parse(variant);
        Assert.Equal(repo, parsed.Repo);
        Assert.Equal(precision, parsed.Precision);
    }

    [Theory]
    [InlineData("fp8+int4")]
    [InlineData("q4+q8")]
    [InlineData("q3")]
    [InlineData("large")]
    public void Parse_RefusesUnknownOrConflictingTokens(string variant) =>
        Assert.Throws<ArgumentException>(() => FishAudioDrama3Variant.Parse(variant));

    [Fact]
    public void APrecision_KeysItsOwnRunner_AndTheBuildsOwnKeysNone()
    {
        TtsModelDescriptor descriptor = FishAudioDrama3Model.Descriptor;
        Assert.Equal("", descriptor.ResolveCacheTag!("drama3"));
        Assert.Equal("", descriptor.ResolveCacheTag("int4"));
        Assert.Equal("q4_k", descriptor.ResolveCacheTag("int4+q4"));
        Assert.Equal(FishAudioDrama3Model.Int4Repo, descriptor.ResolveRepo("int4+q4"));
    }

    [Fact]
    public void EveryCatalogId_ResolvesToTheDescriptor()
    {
        foreach (string id in FishAudioDrama3Model.CatalogIds)
        {
            Assert.Same(FishAudioDrama3Model.Descriptor, TtsCatalog.Resolve(id));
        }
        CatalogEntry entry = ModelCatalog.Find("fishaudio-drama3")!;
        Assert.Equal(Modality.Speech, entry.Modality);
        Assert.True(entry.CliDrivable);
        Assert.Contains(entry.Assets, static a => a.Repo == FishAudioDrama3Model.CodecRepo && a.RepoPath == "codec.pth");
    }

    [Fact]
    public void BothFishAudioModels_CarryTheirAudioLabClassIds()
    {
        Assert.Equal("fishaudio_tts", ModelIdentityCatalog.Find("fishaudio")!.SwarmClassId);
        Assert.Equal("fishaudio_drama3_tts", ModelIdentityCatalog.Find("fishaudio-drama3")!.SwarmClassId);
    }
}
