using System.Text.Json;

namespace HartsyInference.Audio.Tests;

/// <summary>Loads a text-frontend parity fixture from <c>Fixtures/Frontend/&lt;name&gt;.json</c> (copied next to the test
/// assembly). Fixtures are produced by the scripts in <c>tools/parity/</c> from the upstream Python frontends.</summary>
internal static class FrontendParityFixture
{
    public static JsonDocument Load(string name)
    {
        string path = Path.Combine(AppContext.BaseDirectory, "Fixtures", "Frontend", name + ".json");
        return JsonDocument.Parse(File.ReadAllText(path));
    }

    public static List<string> Strings(JsonElement array) => [.. array.EnumerateArray().Select(static e => e.GetString()!)];

    public static List<int[]> IdLists(JsonElement array) =>
        [.. array.EnumerateArray().Select(static e => e.EnumerateArray().Select(static x => x.GetInt32()).ToArray())];
}
