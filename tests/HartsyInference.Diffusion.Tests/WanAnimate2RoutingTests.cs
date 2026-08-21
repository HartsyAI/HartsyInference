using System.Buffers.Binary;
using System.Text;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Recipes.Video;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Wan-Animate-2 checkpoint routing and the defaults that hang off it. Two silent failures live here.
/// First, the checkpoint is <b>key-for-key a Wan2.1 I2V-14B one</b>, so a key-based sniff classifies it as the plain
/// backbone and it loads as text/image-to-video with the driving video quietly discarded. Second,
/// <see cref="InferenceEngine"/> resolves video defaults per FAMILY, and every Wan variant shares the
/// <c>wan-21-14b</c> compat class — a variant missing from <see cref="WanVideoRecipe.DefaultsFor"/> silently gets
/// plain Wan's 50 steps at guidance 5.0 instead of its own, which is exactly what happened to Animate V1 before
/// <c>9605d195</c>.</summary>
public sealed class WanAnimate2RoutingTests
{
    /// <summary>The real checkpoint's <c>__metadata__["config"]</c>, trimmed to the key that carries the claim.</summary>
    private const string Animate2ConfigJson = """{"transformer":{"model_type":"animate2"}}""";

    [Fact]
    public void DefaultsFor_Animate2Checkpoint_ResolvesAnimate2Numbers_NotPlainWans()
    {
        string path = WriteHeaderOnlySafeTensors(Animate2ConfigJson);
        try
        {
            WanVideoRecipe family = new WanVideoRecipe(WanVideoRecipe.Wan21_14BCompatClassId);
            VideoDefaults resolved = family.DefaultsFor(path);
            VideoDefaults expected = new WanAnimate2Recipe().Defaults;

            Assert.Equal(expected.Steps, resolved.Steps);
            Assert.Equal(expected.CfgScale, resolved.CfgScale);
            Assert.Equal(expected.Frames, resolved.Frames);
            Assert.Equal(expected.Fps, resolved.Fps);
            // The trap, stated positively: these must NOT be the family's own numbers.
            Assert.NotEqual(family.Defaults.Steps, resolved.Steps);
            Assert.NotEqual(family.Defaults.CfgScale, resolved.CfgScale);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void DefaultsFor_Animate2Checkpoint_CarriesTheReferencesOperativeBaseSettings()
    {
        VideoDefaults defaults = new WanAnimate2Recipe().Defaults;

        Assert.Equal(40, defaults.Steps);
        Assert.Equal(3.0f, defaults.CfgScale);
        Assert.Equal(81, defaults.Frames);
    }

    [Fact]
    public void SupportsFor_Animate2Checkpoint_ClaimsDrivingVideo_AndNothingItCannotConsume()
    {
        string path = WriteHeaderOnlySafeTensors(Animate2ConfigJson);
        try
        {
            VideoFeatures supports = new WanVideoRecipe(WanVideoRecipe.Wan21_14BCompatClassId).SupportsFor(path);

            Assert.Equal(VideoFeatures.DrivingVideo, supports & VideoFeatures.DrivingVideo);
            Assert.Equal(VideoFeatures.InitImage, supports & VideoFeatures.InitImage);
            // Animate-2 has no end-frame or reference-image/video/audio pathway; claiming one would let a transport
            // populate it and get a plausible clip with the input dropped.
            Assert.Equal(VideoFeatures.None, supports & VideoFeatures.EndFrame);
            Assert.Equal(VideoFeatures.None, supports & VideoFeatures.ReferenceImages);
            Assert.Equal(VideoFeatures.None, supports & VideoFeatures.ReferenceVideos);
            Assert.Equal(VideoFeatures.None, supports & VideoFeatures.ReferenceAudios);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void DetectVariant_PlainI2vHeaderWithNoMetadata_IsNotMistakenForAnimate2()
    {
        // The negative case matters as much as the positive: Animate-2 shares its entire key set with plain I2V-14B,
        // so a detector that guessed from keys would claim every Wan i2v checkpoint.
        string path = WriteHeaderOnlySafeTensors(configJson: null);
        try
        {
            Assert.Equal(WanVideoRecipe.WanVariant.Base, WanVideoRecipe.DetectVariant(path));
            WanVideoRecipe family = new WanVideoRecipe(WanVideoRecipe.Wan21_14BCompatClassId);
            Assert.Equal(family.Defaults.Steps, family.DefaultsFor(path).Steps);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void DetectVariant_Animate2Metadata_WinsOverTheKeySniff()
    {
        string path = WriteHeaderOnlySafeTensors(Animate2ConfigJson);
        try
        {
            Assert.Equal(WanVideoRecipe.WanVariant.Animate2, WanVideoRecipe.DetectVariant(path));
        }
        finally
        {
            File.Delete(path);
        }
    }

    /// <summary>Writes a valid safetensors file whose header carries the I2V key set (and optionally the Animate-2
    /// metadata) but no tensor bytes — the detector reads the header only.</summary>
    private static string WriteHeaderOnlySafeTensors(string? configJson)
    {
        StringBuilder header = new StringBuilder("{");
        if (configJson is not null)
        {
            header.Append($"\"__metadata__\":{{\"config\":{JsonEscape(configJson)}}},");
        }
        // A couple of real I2V-14B key names, so the key-based arms of DetectVariant have something to reject.
        header.Append("\"patch_embedding.weight\":{\"dtype\":\"F32\",\"shape\":[5120,36,1,2,2],\"data_offsets\":[0,0]},");
        header.Append("\"blocks.0.self_attn.q.weight\":{\"dtype\":\"F32\",\"shape\":[5120,5120],\"data_offsets\":[0,0]}}");
        byte[] json = Encoding.UTF8.GetBytes(header.ToString());
        string path = Path.Combine(Path.GetTempPath(), $"hartsy-animate2-{Guid.NewGuid():N}.safetensors");
        using (FileStream fs = File.Create(path))
        {
            Span<byte> len = stackalloc byte[8];
            BinaryPrimitives.WriteInt64LittleEndian(len, json.Length);
            fs.Write(len);
            fs.Write(json);
        }
        return path;
    }

    private static string JsonEscape(string value) => System.Text.Json.JsonSerializer.Serialize(value);
}
