using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tests.MemoryManagement;
using Xunit;

namespace HartsyInference.Core.Tests.Configuration;

/// <summary>Covers the settings file as a place a user CHANGES something, not just one the engine reads: a value
/// written here has to survive a restart, and writing one must not destroy the rest of the document.</summary>
/// <remarks>Serialised with the other configuration tests because <see cref="KnobFile.ExplicitPath"/> and the
/// override store are process-wide.</remarks>
[Collection(EnvironmentSensitiveCollection.Name)]
public sealed class KnobFileTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "hartsy-knobfile-" + Guid.NewGuid().ToString("N"));
    private readonly string? _previous = KnobFile.ExplicitPath;

    public KnobFileTests()
    {
        Directory.CreateDirectory(_dir);
        KnobFile.ExplicitPath = Path.Combine(_dir, "settings.json");
    }

    public void Dispose()
    {
        KnobFile.ExplicitPath = _previous;
        KnobStore.ResetOverrides();
        try
        {
            Directory.Delete(_dir, recursive: true);
        }
        catch (IOException)
        {
            // A leftover temp directory is not worth failing a test over.
        }
    }

    /// <summary>The whole point of the file: a value set once is still there for the next process.</summary>
    [Fact]
    public void Save_PersistsAcrossAReload()
    {
        KnobFile.Save("paths.modelsRoot", "/mnt/models");

        KnobStore.ResetOverrides();
        KnobFile.Reload();

        Assert.Equal("/mnt/models", EngineKnobs.ModelsRoot.Value);
        Assert.Equal("settings file", KnobStore.SourceOf("paths.modelsRoot"));
    }

    /// <summary>Writing one setting must not drop the others, or editing a value silently resets the machine.</summary>
    [Fact]
    public void Save_KeepsTheRestOfTheDocument()
    {
        File.WriteAllText(KnobFile.ExplicitPath!,
            """{"profile":"default","settings":{"vram.keepModels":false}}""");
        KnobFile.Reload();

        KnobFile.Save("paths.modelsRoot", "/mnt/models");

        string written = File.ReadAllText(KnobFile.ExplicitPath!);
        Assert.Contains("\"profile\": \"default\"", written, StringComparison.Ordinal);
        Assert.Contains("vram.keepModels", written, StringComparison.Ordinal);
        Assert.Contains("paths.modelsRoot", written, StringComparison.Ordinal);
    }

    /// <summary>Re-setting a value replaces it rather than writing the id twice, which would be invalid JSON to read back.</summary>
    [Fact]
    public void Save_ReplacesAnExistingValue()
    {
        KnobFile.Save("paths.modelsRoot", "/first");
        KnobFile.Save("paths.modelsRoot", "/second");

        string written = File.ReadAllText(KnobFile.ExplicitPath!);
        Assert.Equal(1, written.Split("paths.modelsRoot").Length - 1);

        KnobStore.ResetOverrides();
        KnobFile.Reload();
        Assert.Equal("/second", EngineKnobs.ModelsRoot.Value);
    }

    /// <summary>A typo is rejected when it is made, not at the next startup.</summary>
    [Fact]
    public void Save_RejectsAnUnknownSetting()
    {
        InvalidOperationException ex = Assert.Throws<InvalidOperationException>(
            () => KnobFile.Save("paths.noSuchSetting", "x"));
        Assert.Contains("paths.noSuchSetting", ex.Message, StringComparison.Ordinal);
        Assert.False(File.Exists(KnobFile.ExplicitPath!));
    }

    /// <summary>A value the knob's type cannot hold is rejected too, through the same parse the file load uses.</summary>
    [Fact]
    public void Save_RejectsAWrongTypedValue()
        => Assert.Throws<InvalidOperationException>(() => KnobFile.Save("vram.keepModels", "banana"));

    /// <summary>A clamped knob stores the clamped value, so the file cannot hold something the engine would not honour.</summary>
    [Fact]
    public void Save_StoresTheCoercedValue()
    {
        object? stored = KnobFile.Save("numerics.gemvWpb", "999");

        Assert.Equal(16, stored);
        Assert.Contains("16", File.ReadAllText(KnobFile.ExplicitPath!), StringComparison.Ordinal);
    }

    /// <summary>A host's explicit Set beats the file, and the reported source says so — this is how SwarmUI drives the models root.</summary>
    [Fact]
    public void HostOverrideBeatsTheFileAndIsReportedAsSuch()
    {
        KnobFile.Save("paths.modelsRoot", "/from-file");
        KnobStore.Set(EngineKnobs.ModelsRoot, "/from-host");

        Assert.Equal("/from-host", EngineKnobs.ModelsRoot.Value);
        Assert.Equal("host", KnobStore.SourceOf("paths.modelsRoot"));
    }

    /// <summary>A settings file that does not exist yet is the normal state, not an error — Save has to be able
    /// to create it. Discover used to throw for an ExplicitPath that was absent, which made writing the first
    /// setting impossible for any host that names its own file, because writing one reads one first.</summary>
    [Fact]
    public void Save_CreatesTheFileWhenItDoesNotExistYet()
    {
        Assert.False(File.Exists(KnobFile.ExplicitPath!));

        KnobFile.Save("paths.modelsRoot", "/mnt/created");

        Assert.True(File.Exists(KnobFile.ExplicitPath!));
        Assert.Equal("/mnt/created", EngineKnobs.ModelsRoot.Value);
    }

    /// <summary>Reading a setting before any file exists yields the declared defaults rather than throwing.</summary>
    [Fact]
    public void Reading_WithNoFileYet_UsesDefaults()
    {
        Assert.False(File.Exists(KnobFile.ExplicitPath!));
        Assert.Equal(4, EngineKnobs.GemvWpb.Value);
    }

    /// <summary>An unset setting reports the declared default as its source, so "where did this come from" always has an answer.</summary>
    [Fact]
    public void UnsetSettingReportsTheDefault()
    {
        KnobStore.ResetOverrides();
        Assert.Equal("default", KnobStore.SourceOf("numerics.gemvWpb"));
    }
}
