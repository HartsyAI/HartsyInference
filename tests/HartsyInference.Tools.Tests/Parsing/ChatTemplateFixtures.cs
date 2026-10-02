using HartsyInference.Tests.Common;

namespace HartsyInference.Tools.Tests.Parsing;

/// <summary>Loads the committed <c>tokenizer.chat_template</c> fixtures (see <c>Fixtures/ChatTemplates/README.md</c>
/// for provenance) and the matching local GGUF path each one was extracted from, for the guarded
/// re-extraction check in <c>ToolCallTemplateDetectionTests</c>.</summary>
internal static class ChatTemplateFixtures
{
    public static string Qwen3_4B => Read("qwen3-4b.jinja");
    public static string Qwen25_1_5B => Read("qwen2.5-1.5b.jinja");
    public static string Qwen35_0_8B => Read("qwen3.5-0.8b.jinja");
    public static string DeepSeekR1DistillQwen1_5B => Read("deepseek-r1-distill-qwen-1.5b.jinja");
    public static string Glm4_9B_0414 => Read("glm-4-9b-0414.jinja");
    public static string Llama32_1B_Instruct => Read("llama-3.2-1b-instruct.jinja");
    public static string Mistral7B_Instruct_v0_3 => Read("mistral-7b-instruct-v0.3.jinja");
    public static string Gemma4_E2B_It => Read("gemma-4-e2b-it.jinja");

    /// <summary>Fixture file name → the local GGUF it was extracted from (<see cref="TestPaths.Llm"/> — the
    /// repo's single source of truth for these, each overridable via its own env var), for the guarded live
    /// re-extraction check. Not every dev machine or CI runner has the models directory mounted, so that check
    /// skips cleanly when a path is missing via <see cref="RealWeightGate"/> — the committed fixtures above are
    /// what the Unit-lane tests actually run against. A computed property, not a static field: the plain fixture
    /// properties above share this class's type initializer, and a Unit-lane test must never be able to fail
    /// from a <see cref="TestPaths"/> resolution problem it has nothing to do with.</summary>
    public static IReadOnlyDictionary<string, string> SourceGgufPaths => new Dictionary<string, string>(StringComparer.Ordinal)
    {
        ["qwen3-4b.jinja"] = TestPaths.Llm.Qwen3_4BQ4KM,
        ["qwen2.5-1.5b.jinja"] = TestPaths.Llm.Qwen25_15B("Q8_0"),
        ["qwen3.5-0.8b.jinja"] = TestPaths.Llm.Qwen35_08BQ4KM,
        ["deepseek-r1-distill-qwen-1.5b.jinja"] = TestPaths.Llm.DeepSeekR1DistillQwen15BQ4KM,
        ["glm-4-9b-0414.jinja"] = TestPaths.Llm.Glm4_9BQ4KM,
        ["llama-3.2-1b-instruct.jinja"] = TestPaths.Llm.Llama32_1BQ8,
        ["mistral-7b-instruct-v0.3.jinja"] = TestPaths.Llm.Mistral7BInstructV0_3Q4KM,
        ["gemma-4-e2b-it.jinja"] = TestPaths.Llm.Gemma4E2BItQ4KM,
    };

    private static string Read(string name) => File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Fixtures", "ChatTemplates", name));
}
