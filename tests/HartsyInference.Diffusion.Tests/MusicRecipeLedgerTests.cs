using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The music recipes are the single source of music defaults. Each expected value below is the literal its
/// loader used before the recipes existed, cited to its line on origin/main, so a passing run proves no default moved.</summary>
public sealed class MusicRecipeLedgerTests
{
    public static TheoryData<string> MusicFamilies => [.. MusicRecipeRegistry.DefaultNames];

    public static TheoryData<string, string> MusicVariants
    {
        get
        {
            TheoryData<string, string> data = [];
            foreach (string family in MusicRecipeRegistry.DefaultNames)
            {
                foreach (string variant in MusicRecipeRegistry.Resolve(family)!.Variants)
                {
                    data.Add(family, variant);
                }
            }
            return data;
        }
    }

    [Fact]
    public void MusicRegistry_ShipsTheNineFamiliesOfTheCatalog() =>
        Assert.Equal(
            ["musicgen", "audiogen", "acestep", "yue", "yue2", "heartmula", "minimaxmusic3", "stableaudio", "controlfoley"],
            MusicRecipeRegistry.DefaultNames);

    [Theory]
    [MemberData(nameof(MusicFamilies))]
    public void MusicRecipe_DeclarationsAreValid(string family) =>
        Assert.Empty(RecipeRegistryValidator.Problems(MusicRecipeRegistry.Resolve(family)!));

    [Theory]
    [MemberData(nameof(MusicVariants))]
    public void MusicVariant_DeclaresPositiveSampling(string family, string variant)
    {
        MusicDefaults defaults = MusicRecipeRegistry.Resolve(family)!.DefaultsFor(variant);
        Assert.True(defaults.Steps is null or >= 1);
        Assert.True(defaults.CfgScale is null or > 0);
        Assert.True(defaults.Shift is null or > 0);
    }

    [Theory]
    [MemberData(nameof(MusicFamilies))]
    public void MusicDefaults_FillOnlyUnsetValues(string family)
    {
        IMusicRecipe recipe = MusicRecipeRegistry.Resolve(family)!;
        foreach (MusicDefaults defaults in DefaultsOf(recipe))
        {
            Assert.Equal(Expected(defaults), defaults.Apply(new MusicRequest { Prompt = "p" }));

            MusicRequest explicitRequest = new MusicRequest
            {
                Prompt = "p", Duration = 37, InferSteps = 9, Shift = 2.5, CfgScale = 3.5, CfgIntervalStart = 0.25,
                CfgIntervalEnd = 0.75, CoverStrength = 0.25, ReferenceStartSeconds = 1.5, ReferenceEndSeconds = 12,
                InferMethod = "sde", GuidanceType = "cfg", Thinking = false, LmTemperature = 1.1, LmCfgScale = 3.5,
                LmTopK = 40, LmTopP = 0.8,
            };
            Assert.Equal(explicitRequest, defaults.Apply(explicitRequest));
        }
    }

    [Theory]
    [InlineData("musicgen")]
    [InlineData("acestep")]
    public void MusicDefaults_KeepsCallerDurationAndFillsTheRest(string family)
    {
        MusicDefaults defaults = MusicRecipeRegistry.Resolve(family)!.Defaults;
        MusicRequest filled = defaults.Apply(new MusicRequest { Prompt = "p", Duration = 37 });
        Assert.Equal(37, filled.Duration);
        Assert.Equal(defaults.Steps, filled.InferSteps);
    }

    [Fact]
    public void MusicRecipe_MusicGenAndAudioGenDeclareTheSharedWindow()
    {
        foreach (string family in new[] { "musicgen", "audiogen" })
        {
            IMusicRecipe recipe = MusicRecipeRegistry.Resolve(family)!;
            // MusicRequest.cs:16 (= 10); MusicCatalog.cs:211 clamps the seconds to 30.
            Assert.Equal(new MusicDefaults { Duration = 10 }, recipe.Defaults);
            Assert.Equal(new MusicConstraints { MaxDurationSeconds = 30 }, recipe.Constraints);
        }
    }

    [Fact]
    public void MusicRecipe_AceStepFamilyDefaultsMatchTheLoaders()
    {
        IMusicRecipe recipe = MusicRecipeRegistry.Resolve("acestep")!;
        // AceStepMusicModel.cs:189 clamps to 600 s on 1.5; MusicRequest.cs:16/76/82/85/88/91/94/155 and
        // AceStepMusicModel.cs:256 (InferMethod "ode") and :545 (GuidanceType "apg" when UseAdg was false).
        Assert.Equal(
            new MusicDefaults
            {
                Duration = 10,
                InferMethod = "ode",
                GuidanceType = "apg",
                CfgIntervalStart = 0,
                CfgIntervalEnd = 1,
                CoverStrength = 0.5,
                Thinking = true,
                LmTemperature = 0.85,
                LmCfgScale = 2,
                LmTopK = 0,
                LmTopP = 0.9,
            },
            recipe.Defaults);
        Assert.Equal(new MusicConstraints { MaxDurationSeconds = 600 }, recipe.Constraints);
    }

    [Theory]
    [InlineData("turbo", 8, 3)]
    [InlineData("turbo-shift1", 8, 1)]
    [InlineData("turbo-shift3", 8, 3)]
    [InlineData("turbo-continuous", 8, 3)]
    [InlineData("sft", 50, 1)]
    [InlineData("base", 32, 1)]
    [InlineData("xl-turbo", 8, 3)]
    [InlineData("xl-sft", 50, 1)]
    [InlineData("xl-base", 32, 1)]
    [InlineData("acestep", 8, 3)]
    public void MusicRecipe_AceStepVariantsMatchTheLoaderTable(string variant, int steps, double shift)
    {
        // AceStepMusicModel.cs:627-641 (VariantDefaults) and :251 (CfgScale 7f on the 1.5 path).
        MusicDefaults defaults = MusicRecipeRegistry.Resolve("acestep")!.DefaultsFor(variant);
        Assert.Equal((steps, shift, 7d), (defaults.Steps, defaults.Shift, defaults.CfgScale));
    }

    [Fact]
    public void MusicRecipe_YueDeclaresItsReferenceWindow()
    {
        IMusicRecipe recipe = MusicRecipeRegistry.Resolve("yue")!;
        // MusicRequest.cs:170 (ReferenceEndSeconds = 30); the reference start was implicit 0.
        Assert.Equal(new MusicDefaults { Duration = 10, ReferenceStartSeconds = 0, ReferenceEndSeconds = 30 }, recipe.Defaults);
        // YueMusicModel.cs:206 clamps to 300 s.
        Assert.Equal(new MusicConstraints { MaxDurationSeconds = 300 }, recipe.Constraints);
    }

    [Fact]
    public void MusicRecipe_Yue2DeclaresItsOdeStepsAndCap()
    {
        IMusicRecipe recipe = MusicRecipeRegistry.Resolve("yue2")!;
        // Yue2Config.cs:51 (OdeSteps = 32, read through Yue2Config.V1 at Yue2MusicModel.cs:264); Yue2Protocol.cs:83 caps at 900 s.
        Assert.Equal(new MusicDefaults { Duration = 10, Steps = 32 }, recipe.Defaults);
        Assert.Equal(new MusicConstraints { MaxDurationSeconds = 900 }, recipe.Constraints);
    }

    [Fact]
    public void MusicRecipe_HeartMulaDeclaresItsCap()
    {
        IMusicRecipe recipe = MusicRecipeRegistry.Resolve("heartmula")!;
        // HeartMulaMusicModel.cs:141 clamps to 300 s; the duration default is MusicRequest.cs:16.
        Assert.Equal(new MusicDefaults { Duration = 10 }, recipe.Defaults);
        Assert.Equal(new MusicConstraints { MaxDurationSeconds = 300 }, recipe.Constraints);
    }

    [Fact]
    public void MusicRecipe_MiniMaxMusic3DeclaresItsFlowDefaults()
    {
        IMusicRecipe recipe = MusicRecipeRegistry.Resolve("minimaxmusic3")!;
        // MiniMaxMusic3FlowPipeline.cs:37 (30 steps) and :34 (1.7 CFG), applied by MiniMaxMusic3MusicModel.cs:158-160;
        // the frame grid caps the clip at MaxAudioFrames / FrameRate = 9000 / 25 s (MiniMaxMusic3MusicModel.cs:156-157).
        Assert.Equal(new MusicDefaults { Duration = 10, Steps = 30, CfgScale = 1.7 }, recipe.Defaults);
        Assert.Equal(new MusicConstraints { MaxDurationSeconds = 360 }, recipe.Constraints);
    }

    [Fact]
    public void MusicRecipe_StableAudioDeclaresItsEightStepsAndTrainedWindow()
    {
        IMusicRecipe recipe = MusicRecipeRegistry.Resolve("stableaudio")!;
        // StableAudioPipeline.cs:44 (steps ?? 8); the trained window is 256 latent tokens x 2048 / 44100 Hz (StableAudioDitConfig.cs:85,88).
        Assert.Equal(new MusicDefaults { Duration = 10, Steps = 8 }, recipe.Defaults);
        Assert.InRange(recipe.Constraints.MaxDurationSeconds!.Value, 11.88, 11.89);
    }

    [Fact]
    public void MusicRecipe_ControlFoleyDeclaresItsSamplingAndReferenceWindow()
    {
        IMusicRecipe recipe = MusicRecipeRegistry.Resolve("controlfoley")!;
        // ControlFoleyMusicModel.cs:121-122 (25 steps, 4.5 CFG); MusicRequest.cs:170 (reference end 30 s).
        Assert.Equal(
            new MusicDefaults { Duration = 10, Steps = 25, CfgScale = 4.5, ReferenceStartSeconds = 0, ReferenceEndSeconds = 30 },
            recipe.Defaults);
        Assert.Equal(new MusicConstraints(), recipe.Constraints);
    }

    private static IEnumerable<MusicDefaults> DefaultsOf(IMusicRecipe recipe)
    {
        yield return recipe.Defaults;
        foreach (string variant in recipe.Variants)
        {
            yield return recipe.DefaultsFor(variant);
        }
    }

    private static MusicRequest Expected(MusicDefaults defaults) => new MusicRequest
    {
        Prompt = "p",
        Duration = defaults.Duration,
        InferSteps = defaults.Steps,
        Shift = defaults.Shift,
        CfgScale = defaults.CfgScale,
        CfgIntervalStart = defaults.CfgIntervalStart,
        CfgIntervalEnd = defaults.CfgIntervalEnd,
        CoverStrength = defaults.CoverStrength,
        ReferenceStartSeconds = defaults.ReferenceStartSeconds,
        ReferenceEndSeconds = defaults.ReferenceEndSeconds,
        InferMethod = defaults.InferMethod,
        GuidanceType = defaults.GuidanceType,
        Thinking = defaults.Thinking,
        LmTemperature = defaults.LmTemperature,
        LmCfgScale = defaults.LmCfgScale,
        LmTopK = defaults.LmTopK,
        LmTopP = defaults.LmTopP,
    };

    [Theory]
    [InlineData(0, null, null)]
    [InlineData(-3, null, null)]
    [InlineData(null, -1.0, null)]
    [InlineData(null, 0.0, null)]
    [InlineData(null, double.NaN, null)]
    [InlineData(null, null, 0.0)]
    public void ExplicitTunablesNoSamplerCanRun_AreRefusedByName(int? steps, double? cfg, double? duration)
    {
        MusicRequest request = new MusicRequest { Prompt = "p", InferSteps = steps, CfgScale = cfg, Duration = duration };

        ArgumentOutOfRangeException error = Assert.Throws<ArgumentOutOfRangeException>(
            () => MusicService.RejectInvalidTunables(request));
        Assert.Contains(error.ParamName, new[] { "InferSteps", "CfgScale", "Duration" });
    }

    [Fact]
    public void UnsetAndPositiveTunables_PassValidation() =>
        MusicService.RejectInvalidTunables(new MusicRequest { Prompt = "p", InferSteps = 8, CfgScale = 1, Duration = 10 });
}
