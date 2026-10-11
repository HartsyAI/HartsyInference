using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using HartsyInference.LLM.Generation;

namespace HartsyInference.LLM.ChatTemplates;

/// <summary>The Hermes/Qwen2.5 tool prompt: the <c># Tools</c> system block, the calls and results it teaches, and the rewrite that carries them into a template with no tool slot.</summary>
public static partial class HermesToolPrompt
{
    /// <summary>The system text a conversation gets when it brings none.</summary>
    public const string DefaultSystem = "You are a helpful assistant.";

    private const string ToolsHeader = "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
        + "You are provided with function signatures within <tools></tools> XML tags:\n<tools>";
    private const string ToolsFooter = "\n</tools>\n\nFor each function call, return a json object with function name and arguments "
        + "within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>";

    [GeneratedRegex(@"\btools\b")]
    private static partial Regex ToolsReference();

    /// <summary>True when a chat template's own source refers to the <c>tools</c> variable, so it renders tool definitions itself.</summary>
    public static bool TemplateOffersTools(string templateSource) => ToolsReference().IsMatch(templateSource);

    /// <summary>The <c># Tools</c> block with each schema re-serialized the way Jinja's <c>tojson</c> would print it, so the fallback matches the real template byte for byte.</summary>
    public static string ToolsBlock(IReadOnlyList<ToolSpec> tools)
    {
        StringBuilder block = new(ToolsHeader);
        foreach (ToolSpec tool in tools) block.Append('\n').Append(Values.ToJson(Values.ParseJson(tool.Json)));
        return block.Append(ToolsFooter).ToString();
    }

    /// <summary>One sentence telling the model which tool this turn must call.</summary>
    public static string ForceDirective(string toolName) => $"You must answer this turn by calling the tool `{toolName}` using the tool-call format above, with no other text.";

    /// <summary>Rewrites a conversation for a template with no tool slot: the system turn carries the tools block (and the force directive when a tool is forced), an assistant call is written as its <c>&lt;tool_call&gt;</c> text, and consecutive tool results become one user turn of <c>&lt;tool_response&gt;</c> blocks.</summary>
    /// <param name="messages">The conversation, system turn optional.</param>
    /// <param name="tools">The offered tools.</param>
    /// <param name="systemPrompt">The request's system prompt, merged into the system turn the same way the engine merges it elsewhere.</param>
    /// <param name="forcedTool">The one tool this turn must call, or null.</param>
    public static IReadOnlyList<ChatMessage> RewriteForToolLessTemplate(IReadOnlyList<ChatMessage> messages, IReadOnlyList<ToolSpec> tools, string? systemPrompt, string? forcedTool)
    {
        ArgumentNullException.ThrowIfNull(messages);
        ArgumentNullException.ThrowIfNull(tools);
        IReadOnlyList<ChatMessage> merged = PromptBuilder.WithSystemPrompt(messages, systemPrompt);
        int start = 0;
        string system = "";
        if (merged.Count > 0 && string.Equals(merged[0].Role, "system", StringComparison.OrdinalIgnoreCase))
        {
            system = merged[0].Content;
            start = 1;
        }
        StringBuilder text = new(system.Length > 0 ? system : DefaultSystem);
        text.Append(ToolsBlock(tools));
        if (forcedTool is { Length: > 0 }) text.Append("\n\n").Append(ForceDirective(forcedTool));
        List<ChatMessage> rewritten = [new ChatMessage("system", text.ToString())];
        for (int i = start; i < merged.Count; i++)
        {
            ChatMessage message = merged[i];
            if (IsRole(message, "tool"))
            {
                StringBuilder responses = new();
                for (; i < merged.Count && IsRole(merged[i], "tool"); i++)
                {
                    if (responses.Length > 0) responses.Append('\n');
                    responses.Append("<tool_response>\n").Append(merged[i].Content).Append("\n</tool_response>");
                }
                i--;
                rewritten.Add(new ChatMessage("user", responses.ToString()));
                continue;
            }
            if (IsRole(message, "assistant") && message.ToolCalls is { Count: > 0 } calls)
            {
                StringBuilder body = new(message.Content);
                foreach (ChatToolCall call in calls) body.Append("\n<tool_call>\n").Append(CallJson(call)).Append("\n</tool_call>");
                rewritten.Add(new ChatMessage("assistant", body.ToString()));
                continue;
            }
            rewritten.Add(message);
        }
        return rewritten;
    }

    /// <summary>A call as the Hermes JSON object the tools block asks for: <c>{"name": …, "arguments": …}</c>.</summary>
    internal static string CallJson(ChatToolCall call) => "{\"name\": " + Values.ToJson(call.Name) + ", \"arguments\": " + ArgumentsJson(call.ArgumentsJson) + "}";

    /// <summary>Arguments re-serialized as JSON; a value that is not JSON is written as a string.</summary>
    internal static string ArgumentsJson(string argumentsJson)
    {
        try
        {
            return Values.ToJson(Values.ParseJson(argumentsJson));
        }
        catch (JsonException)
        {
            return Values.ToJson(argumentsJson);
        }
    }

    private static bool IsRole(ChatMessage message, string role) => string.Equals(message.Role, role, StringComparison.OrdinalIgnoreCase);
}
