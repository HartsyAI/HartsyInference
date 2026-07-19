using System.Globalization;
using System.Text;
using System.Threading.Channels;
using HartsyInference.Core.Logging;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>Text-generation service. Phase 1 drives the existing GGUF/SSM path over a flattened prompt; the full lift
/// (device slots, chat-template selection, sampling knobs, the multimodal VLM path, native tool calls) lands with
/// E-LLM-2 behind these same signatures.</summary>
public sealed class TextService : ITextService
{
    private readonly InferenceEngine _engine;

    /// <summary>Creates the service bound to its owning engine.</summary>
    internal TextService(InferenceEngine engine) => _engine = engine;

    /// <inheritdoc/>
    public Task<TextResult> GenerateAsync(ModelSpec spec, TextRequest request, CancellationToken cancel = default)
    {
        ParamState parameters = BuildParams(request);
        string prompt = BuildPrompt(request);
        SinkBridge sink = new SinkBridge(null, null);
        return Task.Run(
            () =>
            {
                GeneratedArtifact artifact = _engine.RunHandler(spec, prompt, parameters, sink, cancel);
                return new TextResult
                {
                    Text = artifact.Text ?? "",
                    Stop = StopFrom(artifact),
                    PromptTokens = MetaInt(artifact, "prompt_tokens"),
                    CompletionTokens = MetaInt(artifact, "generated_tokens"),
                };
            },
            cancel);
    }

    /// <inheritdoc/>
    public async IAsyncEnumerable<TextChunk> StreamAsync(ModelSpec spec, TextRequest request, [EnumeratorCancellation] CancellationToken cancel = default)
    {
        ParamState parameters = BuildParams(request);
        string prompt = BuildPrompt(request);
        Channel<TextChunk> channel = Channel.CreateUnbounded<TextChunk>();
        SinkBridge sink = new SinkBridge(null, piece => channel.Writer.TryWrite(new TextChunk { Kind = TextChunkKind.Chunk, Text = piece }));

        Task worker = Task.Run(
            () =>
            {
                try
                {
                    GeneratedArtifact artifact = _engine.RunHandler(spec, prompt, parameters, sink, cancel);
                    channel.Writer.TryWrite(new TextChunk { Kind = TextChunkKind.Result, Text = artifact.Text ?? "" });
                    channel.Writer.TryWrite(new TextChunk { Kind = TextChunkKind.StopReason, Stop = StopFrom(artifact) });
                }
                catch (OperationCanceledException)
                {
                    channel.Writer.TryWrite(new TextChunk { Kind = TextChunkKind.StopReason, Stop = StopReason.Cancelled });
                }
                catch (Exception ex)
                {
                    Logs.Error($"Text stream failed: {ex}");
                    channel.Writer.TryWrite(new TextChunk { Kind = TextChunkKind.StopReason, Stop = StopReason.Error });
                }
                finally
                {
                    channel.Writer.Complete();
                }
            },
            cancel);

        await foreach (TextChunk chunk in channel.Reader.ReadAllAsync(cancel).ConfigureAwait(false))
            yield return chunk;
        await worker.ConfigureAwait(false);
    }

    /// <inheritdoc/>
    public int CountTokens(ModelSpec spec, string text) => Math.Max(1, ((text?.Length ?? 0) + 3) / 4);

    private static ParamState BuildParams(TextRequest request)
    {
        ParamState parameters = new ParamState(Modality.Text);
        parameters.Put("max-tokens", request.MaxTokens.ToString(CultureInfo.InvariantCulture));
        parameters.Put("temperature", (request.Greedy ? 0.0 : request.Temperature).ToString(CultureInfo.InvariantCulture));
        parameters.Put("top-p", request.TopP.ToString(CultureInfo.InvariantCulture));
        parameters.Put("seed", request.Seed.ToString(CultureInfo.InvariantCulture));
        if (request.GraphDecode == true)
            parameters.Put("graph-decode", "true");
        return parameters;
    }

    private static string BuildPrompt(TextRequest request)
    {
        // Phase 1 bridge: flatten to a single prompt for the existing raw-completion handler. E-LLM-2 replaces this
        // with proper chat-template application over the message list.
        StringBuilder builder = new StringBuilder();
        if (!string.IsNullOrEmpty(request.SystemPrompt))
            builder.AppendLine(request.SystemPrompt);
        foreach (TextMessage message in request.Messages)
            builder.AppendLine(message.Content);
        return builder.ToString().TrimEnd();
    }

    private static StopReason StopFrom(GeneratedArtifact artifact) =>
        artifact.Meta.TryGetValue("stopped_on", out string? reason) && reason == "max-tokens" ? StopReason.Length : StopReason.Stop;

    private static int MetaInt(GeneratedArtifact artifact, string key) =>
        artifact.Meta.TryGetValue(key, out string? value) && int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out int parsed) ? parsed : 0;
}
