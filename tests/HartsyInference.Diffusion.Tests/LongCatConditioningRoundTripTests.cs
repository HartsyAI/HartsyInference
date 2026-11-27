using HartsyInference.Core.Configuration;
using HartsyInference.Core.Numerics;
using HartsyInference.Engine.Requests;
using HartsyInference.Tests.Shared;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The avatar step-0 conditioning dump and substitution under the reference's file names: a generation that dumps and then substitutes its own
/// files reproduces its frames; a substitute file of the wrong element count is refused.</summary>
[Trait("Category", "SyntheticSmoke")]
[Collection(GlobalKnobStore.Name)]
public sealed class LongCatConditioningRoundTripTests
{
    private static readonly string[] Names =
    [
        "text_embeddings_step00", "text_mask_step00", "audio_embedding_step00", "audio_tokens_step00", "input_latent_step00", "speech_16k_mono_step00",
    ];

    [Fact]
    public void SubstitutingTheOwnDumpReproducesTheFrames()
    {
        string dump = Path.Combine(Path.GetTempPath(), "longcat-cond-" + Guid.NewGuid().ToString("N"));
        try
        {
            KnobStore.Set(EngineKnobs.LongCatConditioningDump, dump);
            VideoGenerationResult first = LongCatAvatarRecipeEndToEndTests.Run(onePointSix: false, audioSeed: 1).Result;
            KnobStore.Clear(EngineKnobs.LongCatConditioningDump);
            foreach (string name in Names)
                Assert.True(File.Exists(Path.Combine(dump, name + ".npy")), name);

            KnobStore.Set(EngineKnobs.LongCatConditioningSubstitute, dump);
            VideoGenerationResult second = LongCatAvatarRecipeEndToEndTests.Run(onePointSix: false, audioSeed: 1).Result;
            Assert.Equal(first.Frames.Count, second.Frames.Count);
            for (int i = 0; i < first.Frames.Count; i++)
                Assert.Equal(first.Frames[i].Rgb, second.Frames[i].Rgb);
        }
        finally
        {
            KnobStore.Clear(EngineKnobs.LongCatConditioningDump);
            KnobStore.Clear(EngineKnobs.LongCatConditioningSubstitute);
            if (Directory.Exists(dump)) Directory.Delete(dump, recursive: true);
        }
    }

    [Fact]
    public void ASubstituteOfTheWrongElementCountIsRefused()
    {
        string dump = Path.Combine(Path.GetTempPath(), "longcat-cond-bad-" + Guid.NewGuid().ToString("N"));
        try
        {
            KnobStore.Set(EngineKnobs.LongCatConditioningDump, dump);
            LongCatAvatarRecipeEndToEndTests.Run(onePointSix: false, audioSeed: 1);
            KnobStore.Clear(EngineKnobs.LongCatConditioningDump);
            // The audio file holds three values, not the clip's windows.
            File.WriteAllBytes(Path.Combine(dump, "audio_embedding_step00.npy"), NpyFile.Float32([1f, 2f, 3f], [3]));
            KnobStore.Set(EngineKnobs.LongCatConditioningSubstitute, dump);
            Assert.ThrowsAny<Exception>(() => LongCatAvatarRecipeEndToEndTests.Run(onePointSix: false, audioSeed: 1));
        }
        finally
        {
            KnobStore.Clear(EngineKnobs.LongCatConditioningDump);
            KnobStore.Clear(EngineKnobs.LongCatConditioningSubstitute);
            if (Directory.Exists(dump)) Directory.Delete(dump, recursive: true);
        }
    }
}
