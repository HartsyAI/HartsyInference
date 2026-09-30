using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.LLM.ChatTemplates;

/// <summary>Renders a structured conversation (tools, images, reasoning, tasks) to model-ready ids. The completion-parser half of the protocol joins this interface in the parser change.</summary>
public interface IConversationEncoder
{
    /// <summary>Registry-style name of the prompt format.</summary>
    string Name { get; }

    /// <summary>Encodes <paramref name="messages"/> under <paramref name="options"/>.</summary>
    EncodedConversation Encode(ILlmTokenizer tokenizer, IReadOnlyList<ChatMessage> messages, EncodeOptions options);
}
