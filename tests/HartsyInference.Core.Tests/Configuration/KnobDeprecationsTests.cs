using System.Xml.Linq;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tests.MemoryManagement;
using Xunit;

namespace HartsyInference.Core.Tests.Configuration;

/// <summary>A retired setting id keeps working, with a warning, until its window passes: the settings file is shared with
/// the engine inside SwarmUI, and an unknown id there stops every load.</summary>
[Collection(EnvironmentSensitiveCollection.Name)]
public sealed class KnobDeprecationsTests : IDisposable
{
    private const string Renamed = "test.retiredRenamed", Removed = "test.retiredRemoved";
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "hartsy-knobdep-" + Guid.NewGuid().ToString("N"));
    private readonly string? _previous = KnobFile.ExplicitPath;

    public KnobDeprecationsTests()
    {
        Directory.CreateDirectory(_dir);
        KnobFile.EnsureLoaded();
        KnobFile.ExplicitPath = Path.Combine(_dir, "settings.json");
        KnobDeprecations.Add(new(Renamed, "vram.keepModels", v => v is bool b ? !b : v, int.MaxValue, "inverted"));
        KnobDeprecations.Add(new(Removed, null, null, int.MaxValue, "no longer has an effect"));
    }

    public void Dispose()
    {
        KnobDeprecations.Remove(Renamed);
        KnobDeprecations.Remove(Removed);
        KnobFile.ExplicitPath = _previous;
        KnobFile.Reload();
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    [Fact]
    public void ASettingsFileNamingARetiredId_LoadsAndMapsIt()
    {
        File.WriteAllText(KnobFile.ExplicitPath!, $$$"""{"settings":{"{{{Renamed}}}":true,"{{{Removed}}}":5}}""");
        KnobStore.ResetOverrides();
        KnobFile.Reload();
        Assert.False(EngineKnobs.KeepModels.Value);
        Assert.Equal("settings file", KnobStore.SourceOf("vram.keepModels"));
    }

    [Fact]
    public void TheReplacementId_WinsOverItsRetiredAlias()
    {
        File.WriteAllText(KnobFile.ExplicitPath!, $$$"""{"settings":{"vram.keepModels":true,"{{{Renamed}}}":true}}""");
        KnobStore.ResetOverrides();
        KnobFile.Reload();
        Assert.True(EngineKnobs.KeepModels.Value);
    }

    [Fact]
    public void SetByARetiredId_MapsToTheReplacement()
    {
        KnobProfile p = KnobProfiles.ByName("default")!;
        Assert.True(p.TrySet(Renamed, "false", out KnobProfile updated, out string? error), error);
        Assert.Equal(true, updated.Values["vram.keepModels"]);
        Assert.True(p.TrySet(Removed, "1", out KnobProfile same, out _));
        Assert.Equal(p.Values.Count, same.Values.Count);
    }

    [Fact]
    public void Save_RewritesRetiredKeysUnderTheirReplacement()
    {
        File.WriteAllText(KnobFile.ExplicitPath!, $$$"""{"settings":{"{{{Renamed}}}":true,"{{{Removed}}}":5,"paths.modelsRoot":"/m"}}""");
        KnobStore.ResetOverrides();
        KnobFile.Reload();
        KnobFile.Save("paths.modelsRoot", "/n");
        string text = File.ReadAllText(KnobFile.ExplicitPath!);
        Assert.DoesNotContain(Renamed, text);
        Assert.DoesNotContain(Removed, text);
        Assert.Contains("\"vram.keepModels\": false", text);
    }

    /// <summary>The ratchet: once a release passes an entry's window, this fails until the entry and its shim are deleted.</summary>
    [Fact]
    public void NoRetiredIdIsAcceptedPastItsWindow()
    {
        string props = File.ReadAllText(Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "Directory.Build.props"));
        string suffix = XDocument.Parse(props).Descendants("VersionSuffix").First().Value;
        int current = int.Parse(suffix.Replace("alpha.", "", StringComparison.Ordinal));
        foreach (KnobDeprecations.Entry e in KnobDeprecations.All.Where(e => e.OldId is not (Renamed or Removed)))
            Assert.True(current <= e.AcceptedThrough,
                $"'{e.OldId}' was accepted through alpha.{e.AcceptedThrough}; this is alpha.{current}. Delete the entry and its mapping.");
    }
}
