using HartsyInference.Core.Configuration;
using HartsyInference.Engine.Requests;
using HartsyInference.Tests.Shared;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The avatar conditioning dump and replacement: a generation that dumps its conditioning and then replaces it with its own dumped files
/// gives the same frames; a replacement of the wrong shape is refused.</summary>
[Trait("Category", "SyntheticSmoke")]
[Collection(GlobalKnobStore.Name)]
public sealed class LongCatConditioningRoundTripTests
{
    [Fact]
    public void ReplacingWithTheOwnDumpReproducesTheFrames()
    {
        string dump = Path.Combine(Path.GetTempPath(), "longcat-cond-" + Guid.NewGuid().ToString("N"));
        KnobStore.Set(EngineKnobs.LongCatConditioningDump, dump);
        try
        {
            VideoGenerationResult first = LongCatAvatarRecipeEndToEndTests.Run(onePointSix: false, audioSeed: 1).Result;
            Assert.True(File.Exists(Path.Combine(dump, "call00_text_embeds.npy")));
            Assert.True(File.Exists(Path.Combine(dump, "call00_audio_tokens.npy")));
            Assert.True(File.Exists(Path.Combine(dump, "call00_first_frame_latent.npy")));
            KnobStore.Clear(EngineKnobs.LongCatConditioningDump);
            KnobStore.Set(EngineKnobs.LongCatTextEmbedNpy, Path.Combine(dump, "call00_text_embeds.npy"));
            KnobStore.Set(EngineKnobs.LongCatAudioTokensNpy, Path.Combine(dump, "call00_audio_tokens.npy"));
            KnobStore.Set(EngineKnobs.LongCatFirstFrameLatentNpy, Path.Combine(dump, "call00_first_frame_latent.npy"));
            VideoGenerationResult second = LongCatAvatarRecipeEndToEndTests.Run(onePointSix: false, audioSeed: 1).Result;
            Assert.Equal(first.Frames.Count, second.Frames.Count);
            for (int i = 0; i < first.Frames.Count; i++)
                Assert.Equal(first.Frames[i].Rgb, second.Frames[i].Rgb);
        }
        finally
        {
            KnobStore.Clear(EngineKnobs.LongCatConditioningDump);
            KnobStore.Clear(EngineKnobs.LongCatTextEmbedNpy);
            KnobStore.Clear(EngineKnobs.LongCatAudioTokensNpy);
            KnobStore.Clear(EngineKnobs.LongCatFirstFrameLatentNpy);
            Directory.Delete(dump, recursive: true);
        }
    }

    [Fact]
    public void AReplacementOfTheWrongShapeIsRefused()
    {
        string dump = Path.Combine(Path.GetTempPath(), "longcat-cond-shape-" + Guid.NewGuid().ToString("N"));
        KnobStore.Set(EngineKnobs.LongCatConditioningDump, dump);
        try
        {
            LongCatAvatarRecipeEndToEndTests.Run(onePointSix: false, audioSeed: 1);
            KnobStore.Clear(EngineKnobs.LongCatConditioningDump);
            KnobStore.Set(EngineKnobs.LongCatAudioTokensNpy, Path.Combine(dump, "call00_text_embeds.npy"));
            Assert.ThrowsAny<Exception>(() => LongCatAvatarRecipeEndToEndTests.Run(onePointSix: false, audioSeed: 1));
        }
        finally
        {
            KnobStore.Clear(EngineKnobs.LongCatConditioningDump);
            KnobStore.Clear(EngineKnobs.LongCatAudioTokensNpy);
            Directory.Delete(dump, recursive: true);
        }
    }
}
