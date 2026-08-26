using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tests.MemoryManagement;
using Xunit;

namespace HartsyInference.Core.Tests.Configuration;

/// <summary>Covers the scoped-profile layer: that it beats the environment, that it unwinds, and that the <c>reference</c> profile is neither empty nor silently pinning the wrong direction.</summary>
[Collection(EnvironmentSensitiveCollection.Name)]
public sealed class KnobProfileTests
{
    private const string Var = "HARTSY_PROFILE_TEST";

    private static void With(string? value, Action body)
    {
        string? previous = Environment.GetEnvironmentVariable(Var);
        try
        {
            Environment.SetEnvironmentVariable(Var, value);
            body();
        }
        finally
        {
            Environment.SetEnvironmentVariable(Var, previous);
        }
    }

    private static Knob<bool> Declare(string id, bool defaultValue = false)
        => new(id, Var, defaultValue, KnobScope.Runtime, KnobDomain.Numerics, "test knob");

    /// <summary>A scoped profile beats the machine's exported value — the property that makes one request able to run at reference numerics.</summary>
    [Fact]
    public void ScopedProfile_BeatsTheEnvironment()
    {
        Knob<bool> knob = Declare("test.profile.env");
        With("1", () =>
        {
            Assert.True(knob.Value);
            using (KnobProfile.Create("t").With(knob, false).Push())
            {
                Assert.False(knob.Value);
            }
            Assert.True(knob.Value);
        });
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

    /// <summary>The reference profile has to beat an exported value, or it would be useless on exactly the machines that need it.</summary>
    [Fact]
    public void ReferenceProfile_BeatsAnExportedValue()
    {
        string? previous = Environment.GetEnvironmentVariable("HARTSY_SAGE_ATTN");
        try
        {
            Environment.SetEnvironmentVariable("HARTSY_SAGE_ATTN", "1");
            Assert.True(EngineKnobs.SageAttn.Value);
            using (KnobProfiles.Reference.Push())
            {
                Assert.False(EngineKnobs.SageAttn.Value);
            }
        }
        finally
        {
            Environment.SetEnvironmentVariable("HARTSY_SAGE_ATTN", previous);
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
}
