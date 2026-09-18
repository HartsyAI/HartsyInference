using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Selector grammar for <see cref="BackendFactory"/>: the <c>cuda:1</c> device-ordinal suffix that lets a host pick a
/// GPU without relaunching the process. Parsing is deliberately driver-free, so these run on a machine with no GPU.</summary>
public sealed class BackendSelectorTests
{
    [Theory]
    [InlineData("auto", "auto")]
    [InlineData("cpu", "cpu")]
    [InlineData("cuda", "cuda")]
    [InlineData("vulkan", "vulkan")]
    [InlineData("cuda:1", "cuda")]
    [InlineData("CUDA:1", "cuda")]
    [InlineData("  cuda:2  ", "cuda")]
    [InlineData("auto:1", "auto")]
    [InlineData(null, "auto")]
    public void Kind_Strips_The_Ordinal_Suffix(string? selector, string expected)
    {
        Assert.Equal(expected, BackendFactory.Kind(selector));
    }

    [Theory]
    [InlineData("cuda", 0)]
    [InlineData("cuda:0", 0)]
    [InlineData("cuda:1", 1)]
    [InlineData("CUDA:1", 1)]
    [InlineData(" cuda:3 ", 3)]
    [InlineData("vulkan:2", 2)]
    [InlineData("auto:1", 1)]
    [InlineData("cpu", 0)]
    [InlineData("auto", 0)]
    [InlineData(null, 0)]
    public void ParseOrdinal_Reads_The_Suffix(string? selector, int expected)
    {
        Assert.Equal(expected, BackendFactory.ParseOrdinal(selector));
    }

    [Theory]
    [InlineData("cuda:")]
    [InlineData("cuda:abc")]
    [InlineData("cuda:-1")]
    [InlineData("cuda:1.5")]
    [InlineData("cpu:0")]
    [InlineData("cpu:1")]
    public void ParseOrdinal_Rejects_Malformed_And_Non_Device_Suffixes(string selector)
    {
        Assert.Throws<ArgumentException>(() => BackendFactory.ParseOrdinal(selector));
    }

    [Theory]
    [InlineData("auto")]
    [InlineData("cpu")]
    [InlineData("cuda")]
    [InlineData("vulkan")]
    [InlineData("cuda:0")]
    [InlineData("cuda:1")]
    [InlineData("CUDA:1")]
    [InlineData("vulkan:1")]
    [InlineData("auto:1")]
    public void IsValidSelector_Accepts_Every_Documented_Form(string selector)
    {
        Assert.True(BackendFactory.IsValidSelector(selector));
    }

    [Theory]
    [InlineData("")]
    [InlineData("   ")]
    [InlineData(null)]
    [InlineData("opencl")]
    [InlineData("cuda1")]
    [InlineData("cuda:abc")]
    [InlineData("cuda:-1")]
    [InlineData("cpu:1")]
    public void IsValidSelector_Rejects_Unknown_Kinds_And_Bad_Ordinals(string? selector)
    {
        Assert.False(BackendFactory.IsValidSelector(selector));
    }

    /// <summary>Ordinal 0 must compose back to the bare selector, so every pre-existing caller's string is unchanged.</summary>
    [Theory]
    [InlineData("auto", 0, "auto")]
    [InlineData("cuda", 0, "cuda")]
    [InlineData("cpu", 0, "cpu")]
    [InlineData("cuda", 1, "cuda:1")]
    [InlineData("CUDA", 1, "cuda:1")]
    [InlineData("vulkan", 2, "vulkan:2")]
    [InlineData("auto", 1, "auto:1")]
    [InlineData("cuda:1", 2, "cuda:2")]
    public void WithOrdinal_Composes_A_Selector(string selector, int ordinal, string expected)
    {
        Assert.Equal(expected, BackendFactory.WithOrdinal(selector, ordinal));
    }

    [Fact]
    public void WithOrdinal_Rejects_A_Negative_Ordinal()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => BackendFactory.WithOrdinal("cuda", -1));
    }

    /// <summary>The composing door and the parsing door must agree: <c>cpu:1</c> throws, so composing it must too.</summary>
    [Fact]
    public void WithOrdinal_Rejects_A_Device_Ordinal_On_A_Device_Less_Backend()
    {
        Assert.Throws<ArgumentException>(() => BackendFactory.WithOrdinal("cpu", 1));
        Assert.Throws<ArgumentException>(() => BackendFactory.ParseOrdinal("cpu:1"));
        Assert.False(BackendFactory.IsValidSelector("cpu:1"));
    }

    /// <summary>A selector the factory would reject must not be laundered into a plausible-looking banner string.</summary>
    [Theory]
    [InlineData("cpu:1")]
    [InlineData("cuda:abc")]
    [InlineData("opencl")]
    public void Describe_Echoes_An_Invalid_Selector_Verbatim(string selector)
    {
        Assert.Equal(selector, BackendFactory.Describe(selector));
    }

    [Theory]
    [InlineData("cuda", 0)]
    [InlineData("cuda:1", 1)]
    [InlineData("auto:2", 2)]
    [InlineData("cpu", 0)]
    public void Engine_Exposes_The_Selectors_Ordinal(string selector, int expected)
    {
        // Backend construction is lazy, so this asserts the plumbing without touching a device.
        using InferenceEngine engine = new InferenceEngine(selector);
        Assert.Equal(expected, engine.DeviceOrdinal);
    }

    [Fact]
    public void Engine_Ordinal_Constructor_Composes_The_Selector()
    {
        using InferenceEngine engine = new InferenceEngine("cuda", 1);
        Assert.Equal("cuda:1", engine.BackendSelector);
        Assert.Equal(1, engine.DeviceOrdinal);
    }

    /// <summary>The default two-argument form must produce exactly the string the single-argument callers already pass.</summary>
    [Fact]
    public void Engine_Ordinal_Zero_Is_Indistinguishable_From_The_Legacy_Constructor()
    {
        using InferenceEngine explicitZero = new InferenceEngine("cuda", 0);
        using InferenceEngine legacy = new InferenceEngine("cuda");
        Assert.Equal(legacy.BackendSelector, explicitZero.BackendSelector);
    }

    /// <summary>A device kind is one that takes an ordinal — the test callers must use instead of <c>StartsWith("cuda")</c>,
    /// which reads "not CPU, therefore CUDA" and routed Vulkan requests onto CUDA.</summary>
    [Theory]
    [InlineData("cuda", true)]
    [InlineData("vulkan", true)]
    [InlineData("cpu", false)]
    [InlineData("auto", false)]
    public void IsDeviceKind_Names_The_Kinds_That_Take_An_Ordinal(string kind, bool expected)
    {
        Assert.Equal(expected, BackendFactory.IsDeviceKind(kind));
    }

    /// <summary>A key naming a device is reduced to one concrete spelling of it. Two spellings of one device must
    /// not read as two devices to a caller that uses the key to identify a slot or a serialization gate.</summary>
    [Theory]
    [InlineData("cuda", "cuda:0")]
    [InlineData("cuda:0", "cuda:0")]
    [InlineData("cuda:2", "cuda:2")]
    [InlineData("vulkan", "vulkan:0")]
    [InlineData("vulkan:1", "vulkan:1")]
    [InlineData("cpu", "cpu")]
    public void CanonicalDeviceKey_Gives_One_Spelling_Per_Device(string selector, string expected)
    {
        Assert.Equal(expected, BackendFactory.CanonicalDeviceKey(selector));
    }

    /// <summary>'auto' must not survive canonicalization. It resolves to a real backend, so a caller that then asks
    /// "is this a device kind?" would be told no and skip the gate that serializes access to the very GPU the
    /// request is about to use — while a second slot sat beside the concrete key naming that same device.</summary>
    [Theory]
    [InlineData("auto")]
    [InlineData("auto:1")]
    public void CanonicalDeviceKey_Resolves_Auto_To_A_Concrete_Device(string selector)
    {
        string canonical = BackendFactory.CanonicalDeviceKey(selector);
        Assert.NotEqual("auto", BackendFactory.Kind(canonical));
        // Driver-dependent by nature, so this asserts the invariant rather than a fixed string: whatever 'auto'
        // resolves to on this machine, the key names it concretely and carries the requested ordinal.
        string resolved = BackendFactory.Resolve(selector);
        Assert.Equal(
            BackendFactory.IsDeviceKind(resolved) ? $"{resolved}:{BackendFactory.ParseOrdinal(selector)}" : resolved,
            canonical);
    }

    /// <summary>A layer-split composite is a LIST of selectors, not one, and comes back untouched — reading it as a
    /// single selector would parse "0+cuda:1" as an ordinal.</summary>
    [Theory]
    [InlineData("cuda:0+cuda:1")]
    [InlineData("not-a-backend")]
    public void CanonicalDeviceKey_Leaves_What_It_Cannot_Name_Alone(string selector)
    {
        Assert.Equal(selector, BackendFactory.CanonicalDeviceKey(selector));
    }

    /// <summary>A registered recipe name resolves as its own family id. The video error text advertises these names as
    /// "currently drivable", and the Wan compat classes exist ONLY there — no catalog entry carries them — so before
    /// this they were advertised, accepted on the command line, and then rejected as family 'unknown'.</summary>
    [Theory]
    [InlineData("wan-22-5b")]
    [InlineData("wan-21-14b")]
    [InlineData("wan")]
    public void A_Registered_Recipe_Name_Resolves_As_Its_Own_Family(string familyId)
    {
        ModelSpec spec = new() { Requested = familyId, Modality = Modality.Video, LocalPath = "/nonexistent.safetensors" };
        Assert.Equal(familyId, InferenceEngine.ResolveVideoFamilyId(spec));
    }

    /// <summary>A name no registry knows still falls through to header detection rather than being invented. Resolution
    /// therefore reaches the checkpoint — and reports the path it could not read — instead of returning a family id no
    /// recipe can satisfy, which would surface much later as a construction failure naming a recipe the user never chose.</summary>
    [Fact]
    public void An_Unregistered_Name_Falls_Through_To_Detection()
    {
        ModelSpec spec = new() { Requested = "not-a-real-family", Modality = Modality.Video, LocalPath = "/nonexistent.safetensors" };
        FileNotFoundException ex = Assert.Throws<FileNotFoundException>(() => InferenceEngine.ResolveVideoFamilyId(spec));
        Assert.Contains("/nonexistent.safetensors", ex.Message);
    }
}
