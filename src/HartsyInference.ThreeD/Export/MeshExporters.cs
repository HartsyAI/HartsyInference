namespace HartsyInference.ThreeD.Export;

/// <summary>The exporters the engine ships. To add a format, implement <see cref="IMeshExporter"/> and add one instance to <see cref="All"/>; the conversion service and its HTTP route pick it up by id, and no model profile changes.</summary>
public static class MeshExporters
{
    /// <summary>Every exporter, by id.</summary>
    public static IReadOnlyList<IMeshExporter> All { get; } = [new ObjExporter(), new PlyExporter(), new StlExporter(), new GlbExporter()];

    /// <summary>The exporter for <paramref name="id"/> (case-insensitive).</summary>
    /// <exception cref="NotSupportedException">No exporter has that id.</exception>
    public static IMeshExporter Get(string id) =>
        All.FirstOrDefault(e => string.Equals(e.Id, id, StringComparison.OrdinalIgnoreCase))
        ?? throw new NotSupportedException($"Export format '{id}' is not supported; expected one of {string.Join(", ", All.Select(e => e.Id))}.");

    /// <summary>The notes every export shares: what a static format drops from a source that moved.</summary>
    internal static List<string> CommonNotes(Document.MeshDocument document, bool carriesMotion)
    {
        List<string> notes = [.. document.Notes];
        if (!carriesMotion && document.HadMotion)
        {
            string skin = document.HasSkin ? "the skin (joints and weights; the geometry is the bind pose)" : "";
            string animations = document.Animations.Count > 0 ? $"{document.Animations.Count} animation(s)" : "";
            notes.Add($"Static geometry only: {string.Join(" and ", new[] { skin, animations }.Where(s => s.Length > 0))} dropped.");
        }
        return notes;
    }
}
