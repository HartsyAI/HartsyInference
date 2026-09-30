using HartsyInference.LLM.OutputParsing;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.LLM.ChatTemplates;

/// <summary>Renders a structured conversation (tools, images, reasoning, tasks) to model-ready ids. It also builds the parser for the completions of its own prompt format.</summary>
public interface IConversationEncoder
{
    /// <summary>Registry-style name of the prompt format.</summary>
    string Name { get; }

    /// <summary>Encodes <paramref name="messages"/> under <paramref name="options"/>.</summary>
    EncodedConversation Encode(ILlmTokenizer tokenizer, IReadOnlyList<ChatMessage> messages, EncodeOptions options);

    /// <summary>Creates the parser for completions of this format, starting in <paramref name="state"/>.</summary>
    IOutputParser CreateParser(ILlmTokenizer tokenizer, OutputParserState state);
}
