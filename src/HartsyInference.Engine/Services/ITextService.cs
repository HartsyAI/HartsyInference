using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>Typed text-generation surface: one-shot generation, token streaming, and token counting. The service owns the tokenizer, chat-template selection, sampling, device slots, and the multimodal VLM path.</summary>
public interface ITextService
{
    /// <summary>Generates the full completion for <paramref name="request"/>.</summary>
    Task<TextResult> GenerateAsync(ModelSpec spec, TextRequest request, CancellationToken cancel = default);

    /// <summary>Streams the completion for <paramref name="request"/> as chunk/result/status/stop/tool-call events.</summary>
    IAsyncEnumerable<TextChunk> StreamAsync(ModelSpec spec, TextRequest request, CancellationToken cancel = default);

    /// <summary>Counts the tokens <paramref name="text"/> encodes to under the model's tokenizer.</summary>
    /// <remarks>Never loads a model or touches a backend, so it takes no device. It counts with the tokenizer of an idle
    /// loaded slot on any device, preferring the one holding this spec's model and otherwise any other, and falls back
    /// to a four-characters-per-token estimate when no loaded slot is idle.</remarks>
    int CountTokens(ModelSpec spec, string text);

    /// <summary>Frees the model resident on <paramref name="device"/> (every device when null), releasing its device and host memory and the slot's backend. Waits for any in-flight generation on the slot rather than racing it. Safe when nothing is loaded; returns whether anything was actually freed.</summary>
    bool Unload(string? device = null);

    /// <summary>Returns cached-but-unused device memory (activation/workspace pool slack retained under
    /// <c>vram.mempoolKeep</c>) to the driver for <paramref name="device"/> (every loaded device when null),
    /// without unloading weights. Best-effort and non-blocking: a slot mid-generation is skipped rather than
    /// waited on. Safe when nothing is loaded. Default no-op so an existing <see cref="ITextService"/> (fakes
    /// in tests, any other implementation outside this repo) keeps compiling unchanged.</summary>
    Task TrimMemoryPool(string? device = null) => Task.CompletedTask;

    /// <summary>Loads <paramref name="request"/>'s model onto its device as a named deployment, without generating anything. Replaces whatever the device held. Default: not supported, so
    /// an existing <see cref="ITextService"/> (fakes in tests, other implementations) keeps compiling unchanged.</summary>
    Task<DeploymentStatus> DeployAsync(DeploymentRequest request, CancellationToken cancel = default)
        => throw new NotSupportedException("This text service does not support deployments.");

    /// <summary>Every deployment this service knows, in the order they were first deployed.</summary>
    IReadOnlyList<DeploymentStatus> Deployments => [];

    /// <summary>What deployment <paramref name="deploymentId"/>'s device is doing now, or null when there is no such deployment.</summary>
    DeploymentCapacity? Capacity(string deploymentId) => null;

}
