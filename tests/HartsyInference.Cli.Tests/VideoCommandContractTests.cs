using System.Reflection;
using HartsyInference.Cli.Commands;
using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Planning;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Requests;
using Spectre.Console.Cli;
using Xunit;

namespace HartsyInference.Cli.Tests;

public sealed class VideoCommandContractTests
{
    [Fact]
    public void EngineProfileAndModelProfileRemainDistinctOptions()
    {
        PropertyInfo engineProfile = typeof(VideoCommand.Settings).GetProperty(nameof(VideoCommand.Settings.Profile))!;
        PropertyInfo modelProfile = typeof(VideoCommand.Settings).GetProperty(nameof(VideoCommand.Settings.ModelProfile))!;

        CommandOptionAttribute engineOption = engineProfile.GetCustomAttribute<CommandOptionAttribute>()!;
        CommandOptionAttribute modelOption = modelProfile.GetCustomAttribute<CommandOptionAttribute>()!;
        Assert.Contains("profile", engineOption.LongNames);
        Assert.Contains("model-profile", modelOption.LongNames);
        Assert.DoesNotContain("model-profile", engineOption.LongNames);
    }

    [Fact]
    public void InspectAcceptsRepeatableLoraCompositionInputs()
    {
        PropertyInfo loras = typeof(InspectCommand.Settings).GetProperty(nameof(InspectCommand.Settings.Loras))!;
        PropertyInfo weights = typeof(InspectCommand.Settings).GetProperty(nameof(InspectCommand.Settings.LoraWeights))!;

        Assert.Contains("lora", loras.GetCustomAttribute<CommandOptionAttribute>()!.LongNames);
        Assert.Contains("lora-weight", weights.GetCustomAttribute<CommandOptionAttribute>()!.LongNames);
        Assert.Equal(typeof(string[]), loras.PropertyType);
        Assert.Equal(typeof(double[]), weights.PropertyType);
    }

    [Fact]
    public void InspectDowngradesMissingRuntimeMediaButPreservesReleaseAndStructuralFailures()
    {
        VideoPlan plan = new VideoPlan
        {
            Model = new ModelSpec { Requested = "model", Modality = Modality.Video },
            Profile = new VideoModelProfile
            {
                Id = "ref",
                DisplayName = "Ref",
                FamilyId = "minimax-h3",
                Task = VideoTaskFamily.Ref2Va,
                Acceleration = VideoAccelerationKind.None,
                Attention = VideoAttentionKind.Dense,
                Defaults = VideoDefaults.Standard,
                Features = VideoFeatures.ReferenceImages,
            },
            EffectiveSettings = new VideoEffectiveSettings
            {
                Width = 512,
                Height = 288,
                Frames = 39,
                Fps = 24,
                Steps = 30,
                CfgScale = 1,
                Seed = 1,
                ReferenceSizing = VideoReferenceSizing.Native,
                LockedFields = VideoLockedFields.None,
            },
            Issues =
            [
                Issue("video.profile.reference_required"),
                Issue("video.h3_expansion.release_blocked"),
                Issue("video.profile.tensor_invalid"),
            ],
            CacheIdentity = "test",
        };

        VideoPlan inspected = InspectCommand.AsInspectionResult(plan);

        Assert.Equal(VideoPlanIssueSeverity.Info, inspected.Issues[0].Severity);
        Assert.Equal(VideoPlanIssueSeverity.Error, inspected.Issues[1].Severity);
        Assert.Equal(VideoPlanIssueSeverity.Error, inspected.Issues[2].Severity);
        Assert.False(inspected.IsValid);
    }

    private static VideoPlanIssue Issue(string code) => new VideoPlanIssue
    {
        Code = code,
        Severity = VideoPlanIssueSeverity.Error,
        Message = code,
    };

    [Fact]
    public void ValidationPendingH3ConvertersAreNotRegisteredByTheCli()
    {
        MethodInfo? dispatch = typeof(Program).GetMethod("RunH3Conversion",
            BindingFlags.Static | BindingFlags.NonPublic);

        Assert.Null(dispatch);
    }
}
