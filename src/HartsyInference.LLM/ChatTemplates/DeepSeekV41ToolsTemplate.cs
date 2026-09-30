namespace HartsyInference.LLM.ChatTemplates;

/// <summary>The V4.1 system-prompt tool preamble, verbatim from the reference encoder; <c>{name}</c> markers are filled by the renderer.</summary>
internal static class DeepSeekV41ToolsTemplate
{
    private static readonly string[] Lines =
    [
        "## Tools",
        "",
        "You have access to a set of tools to help answer the user's question. You can invoke tools "
        + "by writing a \"<{dsml_token}{tc_block_name}>\" block like the following:",
        "",
        "<{dsml_token}{tc_block_name}>",
        "<{dsml_token}{tool_call_tag_name} name=\"$TOOL_NAME\">",
        "<{dsml_token}{tool_parameter_tag_name} name=\"$PARAMETER_NAME\" "
        + "string=\"true|false\">$PARAMETER_VALUE</{dsml_token}{tool_parameter_tag_name}>",
        "...",
        "</{dsml_token}{tool_call_tag_name}>",
        "<{dsml_token}{tool_call_tag_name} name=\"$TOOL_NAME2\">",
        "...",
        "</{dsml_token}{tool_call_tag_name}>",
        "</{dsml_token}{tc_block_name}>",
        "",
        "String parameters should be specified as is and set `string=\"true\"`. For all other types "
        + "(numbers, booleans, arrays, objects), pass the value in JSON format and set "
        + "`string=\"false\"`.",
        "",
        "If thinking_mode is enabled (triggered by {thinking_start_token}), you MUST output your "
        + "complete reasoning inside {thinking_start_token}...{thinking_end_token} BEFORE any tool "
        + "calls or final response.",
        "",
        "Otherwise, output directly after {thinking_end_token} with tool calls or final response.",
        "",
        "### Available Tool Schemas",
        "",
        "{tool_schemas}",
        "",
        "You MUST strictly follow the above defined tool name and parameter schemas to invoke tool "
        + "calls.",
    ];

    public static readonly string Text = string.Join("\n", Lines) + "\n";
}
