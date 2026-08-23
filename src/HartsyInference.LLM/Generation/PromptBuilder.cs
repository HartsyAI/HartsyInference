using HartsyInference.LLM.ChatTemplates;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.LLM.Generation;

/// <summary>Shared prompt-id construction for the token-in generation pipelines (transformer + SSM): raw ids pass through, messages go through the chat template, and a bare prompt is wrapped into a one-turn user (+ optional system) message before templating.</summary>
internal static class PromptBuilder
{
    public static int[] BuildPromptIds(GenerationRequest request, ILlmTokenizer tokenizer, IChatTemplate template)
    {
        if (request.RawTokenIds is not null) return [.. request.RawTokenIds];
        if (request.Messages is not null) return template.Encode(tokenizer, request.Messages, addGenerationPrompt: true, request.EnableThinking);
        if (request.Prompt is not null)
        {
            List<ChatMessage> messages = new(2);
            if (!string.IsNullOrEmpty(request.SystemPrompt)) messages.Add(ChatMessage.System(request.SystemPrompt));
            messages.Add(ChatMessage.User(request.Prompt));
            return template.Encode(tokenizer, messages, addGenerationPrompt: true, request.EnableThinking);
        }
        throw new ArgumentException("Request must set RawTokenIds, Messages, or Prompt.", nameof(request));
    }
}
