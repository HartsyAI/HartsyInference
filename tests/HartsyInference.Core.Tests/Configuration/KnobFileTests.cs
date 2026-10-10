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
        // Loads whatever settings file this process uses BEFORE pointing at one that does not exist yet. A test that
        // happened to make the process's first knob read would otherwise go looking for the empty temp path, which
        // throws, so whether the class passed depended on what had run before it.
        KnobFile.EnsureLoaded();
        KnobFile.ExplicitPath = Path.Combine(_dir, "settings.json");
    }

    public void Dispose()
    {
        KnobFile.ExplicitPath = _previous;
        // Put back the settings the rest of the process was running with, not just an empty override store.
        KnobFile.Reload();
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

    /// <summary>A typo is rejected when it is made, not at the next startup.</summary>
    [Fact]
    public void Save_RejectsAnUnknownSetting()
    {
        InvalidOperationException ex = Assert.Throws<InvalidOperationException>(
            () => KnobFile.Save("paths.noSuchSetting", "x"));
        Assert.Contains("paths.noSuchSetting", ex.Message, StringComparison.Ordinal);
        Assert.False(File.Exists(KnobFile.ExplicitPath!));
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
}
