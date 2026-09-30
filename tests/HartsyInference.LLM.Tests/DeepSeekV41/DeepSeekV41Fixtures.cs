using System.Text.Json.Nodes;

namespace HartsyInference.LLM.Tests.DeepSeekV41;

/// <summary>Locates and edits the committed DeepSeek-V4.1 config fixtures.</summary>
internal static class DeepSeekV41Fixtures
{
    /// <summary>Full text of a fixture copied next to the test assembly.</summary>
    public static string Read(string fileName) =>
        File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "DeepSeekV41", "Fixtures", fileName));

    /// <summary>The pinned official config with <paramref name="edit"/> applied to its <c>text_config</c> object.</summary>
    public static string OfficialConfig(Action<JsonObject> edit)
    {
        JsonObject root = JsonNode.Parse(Read("official_config.json"))!.AsObject();
        edit(root["text_config"]!.AsObject());
        return root.ToJsonString();
    }
}
