using System.Text;

namespace HartsyInference.LLM.ChatTemplates;

/// <summary>Renders messages to the V4.1 prompt string, a one-to-one port of the reference encoder's render half (encoding.py).</summary>
internal static class DeepSeekV41PromptRenderer
{
    public const string Bos = "<｜begin▁of▁sentence｜>";
    public const string Eos = "<｜end▁of▁sentence｜>";
    public const string ImagePlaceholder = "<｜deepseek_image｜>";
    public const string SystemToken = "<｜System｜>";
    public const string UserToken = "<｜User｜>";
    public const string AssistantToken = "<｜Assistant｜>";
    public const string LatestReminderToken = "<｜latest_reminder｜>";
    public const int DefaultReasoningEffort = 75;

    private const string ActionTask = "action";

    private static readonly Dictionary<string, string> TaskTokens = new(StringComparer.Ordinal)
    {
        ["action"] = "<｜action｜>",
        ["query"] = "<｜query｜>",
        ["authority"] = "<｜authority｜>",
        ["domain"] = "<｜domain｜>",
        ["title"] = "<｜title｜>",
        ["read_url"] = "<｜read_url｜>",
    };

    /// <summary>Renders the full prompt. <paramref name="imageOrder"/> receives the image index of every placeholder, in message order.</summary>
    public static string Render(IReadOnlyList<ChatMessage> messages, EncodeOptions options, List<int> imageOrder)
    {
        int effort = ResolveEffort(options.ReasoningEffort);
        List<PromptMessage> merged = MergeToolMessages(Preprocess(messages, options, imageOrder));
        SortToolResults(merged);

        bool drop = options.DropThinking ?? true;
        if (merged.Exists(m => m.Tools is { Count: > 0 })) drop = false;
        List<PromptMessage> full = options.Thinking && drop ? DropThinking(merged) : merged;

        StringBuilder prompt = new(Bos);
        for (int i = 0; i < full.Count; i++)
            prompt.Append(RenderMessage(i, full, options.Thinking, drop, effort, options.AddGenerationPrompt));
        return prompt.ToString();
    }

    private static int ResolveEffort(int? effort)
    {
        int value = effort ?? DefaultReasoningEffort;
        if (value < 1 || value > EncodeOptions.MaxReasoningEffort)
            throw new ArgumentOutOfRangeException(nameof(effort), value, "Reasoning effort must be within [1,100].");
        return value;
    }

    private static List<PromptMessage> Preprocess(IReadOnlyList<ChatMessage> messages, EncodeOptions options, List<int> imageOrder)
    {
        if (options.Tools is { Count: > 0 } && messages.Count == 0)
            throw new ArgumentException("Tools require at least one message to attach to.", nameof(messages));
        int imageCount = options.Images?.Count ?? 0;
        List<PromptMessage> result = new(messages.Count);
        for (int i = 0; i < messages.Count; i++)
        {
            ChatMessage source = messages[i];
            ValidateNoPlaceholder(source);
            PromptMessage msg = new(source.Role)
            {
                Content = source.Content ?? string.Empty,
                Tools = i == 0 && options.Tools is { Count: > 0 } ? options.Tools : source.Tools,
                ResponseFormatJson = source.ResponseFormatJson,
                ToolCalls = source.ToolCalls,
                ReasoningContent = source.ReasoningContent,
                Task = source.Task,
                WithoutEos = source.WithoutEos,
            };
            if (source.Blocks is { Count: > 0 })
            {
                msg.Blocks = ConvertBlocks(source.Blocks, imageCount, imageOrder);
                if (msg.Content.Length == 0) msg.Content = JoinText(msg.Blocks);
            }
            if (source.Role == "tool") msg.ReasoningContent = source.ToolCallId;
            result.Add(msg);
        }
        return result;
    }

    private static void ValidateNoPlaceholder(ChatMessage message)
    {
        if (message.Content is not null && message.Content.Contains(ImagePlaceholder, StringComparison.Ordinal))
            throw new ArgumentException($"Message content contains image special token '{ImagePlaceholder}'; use image blocks.");
        if (message.ReasoningContent is not null && message.ReasoningContent.Contains(ImagePlaceholder, StringComparison.Ordinal))
            throw new ArgumentException($"reasoning_content contains image special token '{ImagePlaceholder}'.");
        if (message.ToolCalls is null) return;
        foreach (ChatToolCall call in message.ToolCalls)
        {
            if (call.ArgumentsJson.Contains(ImagePlaceholder, StringComparison.Ordinal)
                || call.Name.Contains(ImagePlaceholder, StringComparison.Ordinal))
                throw new ArgumentException($"Tool call '{call.Name}' contains image special token '{ImagePlaceholder}'.");
        }
    }

    private static List<PromptBlock> ConvertBlocks(IReadOnlyList<ContentBlock> blocks, int imageCount, List<int> imageOrder)
    {
        List<PromptBlock> result = new(blocks.Count);
        foreach (ContentBlock block in blocks)
        {
            switch (block)
            {
                case TextBlock text:
                    if (text.Text.Contains(ImagePlaceholder, StringComparison.Ordinal))
                        throw new ArgumentException($"Text block contains image placeholder '{ImagePlaceholder}'; use image blocks.");
                    result.Add(new PromptBlock(false, text.Text, string.Empty));
                    break;
                case ImageBlock image:
                    if (image.ImageIndex < 0 || image.ImageIndex >= imageCount)
                        throw new ArgumentException($"Image block index {image.ImageIndex} has no entry in EncodeOptions.Images.");
                    imageOrder.Add(image.ImageIndex);
                    result.Add(new PromptBlock(false, ImagePlaceholder, string.Empty));
                    break;
                default:
                    throw new NotSupportedException($"Unsupported content block {block.GetType().Name}.");
            }
        }
        return result;
    }

    private static string JoinText(List<PromptBlock> blocks)
    {
        StringBuilder sb = new();
        for (int i = 0; i < blocks.Count; i++)
        {
            if (i > 0) sb.Append("\n\n");
            sb.Append(blocks[i].Text);
        }
        return sb.ToString();
    }

    // ReasoningContent doubles as the tool-call id carrier for "tool" turns until they are merged.
    private static List<PromptMessage> MergeToolMessages(List<PromptMessage> messages)
    {
        List<PromptMessage> merged = new(messages.Count);
        foreach (PromptMessage msg in messages)
        {
            PromptMessage? last = merged.Count > 0 ? merged[^1] : null;
            if (msg.Role == "tool")
            {
                PromptBlock block = new(true, msg.Content, msg.ReasoningContent ?? string.Empty);
                if (last is { Role: "user", Blocks: not null }) last.Blocks.Add(block);
                else merged.Add(new PromptMessage("user") { Blocks = [block] });
            }
            else if (msg.Role == "user")
            {
                List<PromptBlock> blocks = msg.Blocks ?? [new PromptBlock(false, msg.Content, string.Empty)];
                if (last is { Role: "user", Blocks: not null, Task: null }) last.Blocks.AddRange(blocks);
                else
                {
                    msg.Blocks = blocks;
                    merged.Add(msg);
                }
            }
            else merged.Add(msg);
        }
        return merged;
    }

    private static void SortToolResults(List<PromptMessage> messages)
    {
        Dictionary<string, int> callOrder = new(StringComparer.Ordinal);
        foreach (PromptMessage msg in messages)
        {
            if (msg.Role == "assistant" && msg.ToolCalls is { Count: > 0 })
            {
                callOrder.Clear();
                for (int i = 0; i < msg.ToolCalls.Count; i++)
                    if (msg.ToolCalls[i].Id.Length > 0) callOrder[msg.ToolCalls[i].Id] = i;
            }
            else if (msg.Role == "user" && msg.Blocks is { Count: > 0 })
                SortBlocks(msg.Blocks, callOrder);
        }
    }

    private static void SortBlocks(List<PromptBlock> blocks, Dictionary<string, int> callOrder)
    {
        List<PromptBlock> results = blocks.FindAll(b => b.IsToolResult);
        if (results.Count <= 1 || callOrder.Count == 0) return;
        List<PromptBlock> sorted = [.. results.OrderBy(b => callOrder.TryGetValue(b.ToolUseId, out int rank) ? rank : 0)];
        int next = 0;
        for (int i = 0; i < blocks.Count; i++)
            if (blocks[i].IsToolResult) blocks[i] = sorted[next++];
    }

    // Mid-conversation system turns count as user turns for the generation header.
    private static int FindLastUserIndex(List<PromptMessage> messages)
    {
        for (int i = messages.Count - 1; i >= 0; i--)
            if (messages[i].Role == "user" || (messages[i].Role == "system" && i > 0)) return i;
        return -1;
    }

    private static List<PromptMessage> DropThinking(List<PromptMessage> messages)
    {
        int lastUser = FindLastUserIndex(messages);
        List<PromptMessage> result = new(messages.Count);
        for (int i = 0; i < messages.Count; i++)
        {
            string role = messages[i].Role;
            bool keep = role is "user" or "system" or "tool" or "latest_reminder" or "direct_search_results";
            if (keep || i >= lastUser) result.Add(messages[i]);
            else if (role == "assistant") result.Add(messages[i].WithoutReasoning());
        }
        return result;
    }

    private static string RenderMessage(int index, List<PromptMessage> messages, bool thinking, bool dropThinking,
        int effort, bool addGenerationPrompt)
    {
        PromptMessage msg = messages[index];
        int lastUser = FindLastUserIndex(messages);
        string effortPrompt = index == 0 && thinking
            ? $"Reasoning Effort: {effort} (range 1-100, the higher the value, the more thorough the reasoning)\n\n"
            : string.Empty;
        StringBuilder prompt = new();
        if (index == 0 && (effortPrompt.Length > 0 || msg.Role == "system")) prompt.Append(SystemToken);
        prompt.Append(effortPrompt);

        switch (msg.Role)
        {
            case "system":
                RenderSystem(prompt, msg, index);
                break;
            case "user":
                prompt.Append(UserToken).Append(RenderUserBody(msg));
                break;
            case "latest_reminder":
                prompt.Append(LatestReminderToken).Append(msg.Content);
                break;
            case "assistant":
                RenderAssistant(prompt, index, messages, thinking, dropThinking, lastUser);
                break;
            default:
                throw new NotSupportedException($"Unknown role: {msg.Role}");
        }

        bool followed = index + 1 < messages.Count;
        if (followed && messages[index + 1].Role is not ("assistant" or "latest_reminder")) return prompt.ToString();
        if (!followed && !addGenerationPrompt) return prompt.ToString();
        AppendTransition(prompt, msg, index, thinking, dropThinking, lastUser);
        return prompt.ToString();
    }

    private static void RenderSystem(StringBuilder prompt, PromptMessage msg, int index)
    {
        if (index > 0) prompt.Append(SystemToken);
        prompt.Append(msg.Content);
        if (msg.Tools is { Count: > 0 }) prompt.Append("\n\n").Append(DeepSeekV41Tools.RenderTools(msg.Tools));
        if (!string.IsNullOrEmpty(msg.ResponseFormatJson) && PyJson.Parse(msg.ResponseFormatJson) is { } format
            && format is not PyJsonObject { Count: 0 })
        {
            prompt.Append("\n\n## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n")
                .Append(PyJson.Dump(format));
        }
    }

    private static string RenderUserBody(PromptMessage msg)
    {
        if (msg.Blocks is not { Count: > 0 }) return msg.Content;
        StringBuilder sb = new();
        for (int i = 0; i < msg.Blocks.Count; i++)
        {
            if (i > 0) sb.Append("\n\n");
            PromptBlock block = msg.Blocks[i];
            if (block.IsToolResult) sb.Append("<tool_result>").Append(block.Text).Append("</tool_result>");
            else sb.Append(block.Text);
        }
        return sb.ToString();
    }

    private static void RenderAssistant(StringBuilder prompt, int index, List<PromptMessage> messages, bool thinking,
        bool dropThinking, int lastUser)
    {
        PromptMessage msg = messages[index];
        bool previousHasTask = index > 0 && messages[index - 1].Task is not null;
        if (thinking && !previousHasTask && (!dropThinking || index > lastUser))
            prompt.Append(msg.ReasoningContent ?? string.Empty).Append(DeepSeekV41Tools.ThinkEnd);
        prompt.Append(msg.Content);
        if (msg.ToolCalls is { Count: > 0 }) prompt.Append("\n\n").Append(DeepSeekV41Tools.RenderToolCalls(msg.ToolCalls));
        if (!msg.WithoutEos) prompt.Append(Eos);
    }

    private static void AppendTransition(StringBuilder prompt, PromptMessage msg, int index, bool thinking,
        bool dropThinking, int lastUser)
    {
        if (msg.Task is not null)
        {
            if (!TaskTokens.TryGetValue(msg.Task, out string? taskToken))
                throw new ArgumentException($"Invalid task: '{msg.Task}'. Valid tasks are: {string.Join(", ", TaskTokens.Keys)}");
            if (msg.Task != ActionTask) prompt.Append(taskToken);
            else prompt.Append(AssistantToken).Append(thinking ? DeepSeekV41Tools.ThinkStart : DeepSeekV41Tools.ThinkEnd).Append(taskToken);
        }
        else if (msg.Role == "user" || (msg.Role == "system" && index > 0))
        {
            prompt.Append(AssistantToken);
            bool open = thinking && (!dropThinking || index >= lastUser);
            prompt.Append(open ? DeepSeekV41Tools.ThinkStart : DeepSeekV41Tools.ThinkEnd);
        }
    }
}
