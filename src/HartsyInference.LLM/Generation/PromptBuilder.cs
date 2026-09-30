using HartsyInference.LLM.ChatTemplates;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.LLM.Generation;

/// <summary>Shared prompt-id construction for the token-in generation pipelines (transformer + SSM): raw ids pass through, messages go through the chat template (with the request's system prompt prepended when they lack a leading system turn), and a bare prompt is wrapped into a one-turn user (+ optional system) message before templating.</summary>
internal static class PromptBuilder
{
    public static int[] BuildPromptIds(GenerationRequest request, ILlmTokenizer tokenizer, IChatTemplate template)
    {
        if (request.RawTokenIds is not null) return [.. request.RawTokenIds];
        if (request.Messages is not null)
            return template.Encode(tokenizer, WithSystemPrompt(request.Messages, request.SystemPrompt), addGenerationPrompt: true, request.EnableThinking, request.Tools);
        if (request.Prompt is not null)
        {
            List<ChatMessage> messages = new(2);
            if (!string.IsNullOrEmpty(request.SystemPrompt)) messages.Add(ChatMessage.System(request.SystemPrompt));
            messages.Add(ChatMessage.User(request.Prompt));
            return template.Encode(tokenizer, messages, addGenerationPrompt: true, request.EnableThinking, request.Tools);
        }
        throw new ArgumentException("Request must set RawTokenIds, Messages, or Prompt.", nameof(request));
    }

    /// <summary>Prepends <paramref name="systemPrompt"/> as a system turn unless it is empty or <paramref name="messages"/> already opens with one; a host that folds its system text into the first message keeps that single copy.</summary>
    public static IReadOnlyList<ChatMessage> WithSystemPrompt(IReadOnlyList<ChatMessage> messages, string? systemPrompt)
    {
        if (string.IsNullOrEmpty(systemPrompt)) return messages;
        if (messages.Count > 0 && string.Equals(messages[0].Role, "system", StringComparison.OrdinalIgnoreCase)) return messages;
        List<ChatMessage> withSystem = new(messages.Count + 1) { ChatMessage.System(systemPrompt) };
        withSystem.AddRange(messages);
        return withSystem;
    }
}
