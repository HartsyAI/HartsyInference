using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tests.MemoryManagement;
using Xunit;

namespace HartsyInference.Core.Tests.Configuration;

/// <summary>Covers the scoped-profile layer: that it beats the environment, that it unwinds, and that the <c>reference</c> profile is neither empty nor silently pinning the wrong direction.</summary>
[Collection(EnvironmentSensitiveCollection.Name)]
public sealed class KnobProfileTests
{
    private static Knob<bool> Declare(string id, bool defaultValue = false)
        => new(id, legacyEnv: null, defaultValue, KnobScope.Runtime, KnobDomain.Numerics, "test knob");

    /// <summary>A scoped profile beats the machine's configured value — the property that lets one request run at reference numerics while others keep the machine's settings.</summary>
    [Fact]
    public void ScopedProfile_BeatsTheConfiguredValue()
    {
        Knob<bool> knob = Declare("test.profile.configured");
        KnobStore.Set(knob, true);
        try
        {
            Assert.True(knob.Value);
            using (KnobProfile.Create("t").With(knob, false).Push())
            {
                Assert.False(knob.Value);
            }
            Assert.True(knob.Value);
        }
        finally
        {
            KnobStore.Clear(knob);
        }
    }

    /// <summary>Leaving the scope restores the previous profile, not merely "no profile", so nesting cannot leak.</summary>
    [Fact]
    public void Scopes_Nest_AndUnwindToTheEnclosingProfile()
    {
        Knob<bool> knob = Declare("test.profile.nest");
        using (KnobProfile.Create("outer").With(knob, true).Push())
        {
            Assert.True(knob.Value);
            using (KnobProfile.Create("inner").With(knob, false).Push())
            {
                Assert.False(knob.Value);
            }
            Assert.True(knob.Value);
        }
        Assert.False(knob.Value);
    }

    /// <summary>A knob the profile does not mention keeps resolving normally.</summary>
    [Fact]
    public void UnmentionedKnobs_AreUnaffected()
    {
        Knob<bool> pinned = Declare("test.profile.pinned");
        Knob<bool> other = Declare("test.profile.other", defaultValue: true);
        using (KnobProfile.Create("t").With(pinned, true).Push())
        {
            Assert.True(pinned.Value);
            Assert.True(other.Value);
        }
    }

    /// <summary>Pushing null is a no-op, so an unscoped request costs no behavior change.</summary>
    [Fact]
    public void PushingNull_ChangesNothing()
    {
        Knob<bool> knob = Declare("test.profile.null", defaultValue: true);
        using (KnobProfileScope.Push(null))
        {
            Assert.True(knob.Value);
        }
    }

    [Theory]
    [InlineData("numerics.sageAttn", "0", true)]
    [InlineData("numerics.sageAttn", "false", true)]
    [InlineData("numerics.sageAttn", "banana", false)]
    [InlineData("numerics.gemvWpb", "8", true)]
    [InlineData("numerics.gemvWpb", "eight", false)]
    [InlineData("nosuch.knob", "1", false)]
    public void TrySet_ParsesOrReportsWithoutThrowing(string id, string value, bool expected)
    {
        bool ok = KnobProfile.Create("t").TrySet(id, value, out _, out string? error);
        Assert.Equal(expected, ok);
        Assert.Equal(expected, error is null);
    }

    /// <summary>The default profile is empty on purpose: selecting it must be a no-op that merely names the baseline.</summary>
    [Fact]
    public void DefaultProfile_IsEmpty() => Assert.Equal(0, KnobProfiles.Default.Count);

    /// <summary>The reference profile must actually pin things — an empty one would silently make every parity run meaningless.</summary>
    [Fact]
    public void ReferenceProfile_PinsASubstantialSurface()
        => Assert.True(KnobProfiles.Reference.Count >= 40,
            $"reference pins only {KnobProfiles.Reference.Count} knobs; it is supposed to disable every approximation.");

    /// <summary>Spot-checks the entries whose faithful direction reads BACKWARDS from the name, since those are what a well-meaning edit would invert.</summary>
    /// <remarks><c>noTf32</c> and <c>sdpaNoF16</c> are kill-switches, so faithful is true. <c>bf16Gemv</c> is pinned ON
    /// because the fused GEMV keeps activations F32 while the cuBLAS fallback casts them to BF16. Both coopmat knobs
    /// are required: coopmat2 dispatches first under its own flag and <c>vkDisableCoopmat</c> does not gate it.</remarks>
    [Fact]
    public void ReferenceProfile_PinsTheCounterintuitiveOnesCorrectly()
    {
        using (KnobProfiles.Reference.Push())
        {
            Assert.True(EngineKnobs.NoTf32.Value);
            Assert.True(EngineKnobs.SdpaNoF16.Value);
            Assert.True(EngineKnobs.Bf16Gemv.Value);
            Assert.True(EngineKnobs.VkDisableCoopmat.Value);
            Assert.False(EngineKnobs.VkCoopmat2.Value);
            Assert.False(EngineKnobs.SageAttn.Value);
            Assert.False(EngineKnobs.W8a8.Value);
        }
    }

    /// <summary>The reference profile has to beat a configured value, or it would be useless on exactly the machines that need it.</summary>
    [Fact]
    public void ReferenceProfile_BeatsAConfiguredValue()
    {
        KnobStore.Set(EngineKnobs.SageAttn, true);
        try
        {
            Assert.True(EngineKnobs.SageAttn.Value);
            using (KnobProfiles.Reference.Push())
            {
                Assert.False(EngineKnobs.SageAttn.Value);
            }
        }
        finally
        {
            KnobStore.Clear(EngineKnobs.SageAttn);
        }
    }

    [Theory]
    [InlineData("default")]
    [InlineData("reference")]
    [InlineData("REFERENCE")]
    [InlineData(" reference ")]
    public void ByName_ResolvesKnownProfiles(string name) => Assert.NotNull(KnobProfiles.ByName(name));

    [Fact]
    public void ByName_ReturnsNullForUnknown() => Assert.Null(KnobProfiles.ByName("nope"));

    /// <summary>A per-request override of a load-time setting is REJECTED, because accepting it would report success and change nothing.</summary>
    /// <remarks>Measured, not theorised: an API generation with the reference profile returned a byte-identical
    /// image to one without it, because the backend's TF32/F16-GEMM/FP8 decisions are assigned in the CudaBackend
    /// constructor and are fixed before any request arrives on a long-lived server.</remarks>
    [Fact]
    public void RequestSettings_RejectConstructionScopedOverrides()
    {
        RequestSettings settings = new() { Set = new Dictionary<string, string> { ["numerics.ltx2TwoStage"] = "1" } };
        ArgumentException ex = Assert.Throws<ArgumentException>(() => settings.Resolve());
        Assert.Contains("cannot be changed per request", ex.Message, StringComparison.Ordinal);
        Assert.Contains("numerics.ltx2TwoStage", ex.Message, StringComparison.Ordinal);
    }

    /// <summary>A per-call setting is accepted, so the rejection above is specific rather than blanket.</summary>
    [Fact]
    public void RequestSettings_AcceptRuntimeScopedOverrides()
    {
        RequestSettings settings = new() { Set = new Dictionary<string, string> { ["numerics.sageAttn"] = "0" } };
        KnobProfile? profile = settings.Resolve();
        Assert.NotNull(profile);
        Assert.Equal(1, profile!.Count);
    }

    /// <summary>An empty settings block resolves to null so an ordinary request pushes no scope.</summary>
    [Fact]
    public void RequestSettings_EmptyResolvesToNull() => Assert.Null(new RequestSettings().Resolve());

    /// <summary>The reference profile names what a request could not move, which is what lets a caller refuse instead of silently ignoring.</summary>
    [Fact]
    public void UnreachablePerRequest_NamesTheLoadTimeKnobs()
    {
        IReadOnlyList<string> unreachable = RequestSettings.UnreachablePerRequest(KnobProfiles.Reference);
        Assert.NotEmpty(unreachable);
        Assert.All(unreachable, id => Assert.Equal(KnobScope.Construction,
            KnobRegistry.Describe(KnobRegistry.Find(id)!).Scope));
    }
}
