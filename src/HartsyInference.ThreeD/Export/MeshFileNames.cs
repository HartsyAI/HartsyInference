namespace HartsyInference.ThreeD.Export;

/// <summary>Turns a caller's name into a file stem that is safe to put in a path, an archive entry and an MTL reference.</summary>
internal static class MeshFileNames
{
    /// <summary>Letters, digits, dash, underscore and dot kept; everything else becomes an underscore; an empty result is <c>model</c>.</summary>
    public static string Stem(string name)
    {
        string stem = new([.. Path.GetFileNameWithoutExtension(name ?? "").Select(c => char.IsAsciiLetterOrDigit(c) || c is '-' or '_' or '.' ? c : '_')]);
        return stem.Trim('.').Length == 0 ? "model" : stem;
    }
}
