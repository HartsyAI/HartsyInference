using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>Typed text-generation surface: one-shot generation, token streaming, and token counting. The service
/// owns the tokenizer, chat-template selection, sampling, device slots, and the multimodal VLM path.</summary>
public interface ITextService
{
    /// <summary>Generates the full completion for <paramref name="request"/>.</summary>
    Task<TextResult> GenerateAsync(ModelSpec spec, TextRequest request, CancellationToken cancel = default);

    /// <summary>Streams the completion for <paramref name="request"/> as chunk/result/status/stop/tool-call events.</summary>
    IAsyncEnumerable<TextChunk> StreamAsync(ModelSpec spec, TextRequest request, CancellationToken cancel = default);

    /// <summary>Counts the tokens <paramref name="text"/> encodes to under the model's tokenizer.</summary>
    int CountTokens(ModelSpec spec, string text);

    /// <summary>Frees the model resident on <paramref name="device"/> (every device when null), releasing its device
    /// and host memory and the slot's backend. Waits for any in-flight generation on the slot rather than racing it.
    /// Safe when nothing is loaded; returns whether anything was actually freed.</summary>
    bool Unload(string? device = null);
}
