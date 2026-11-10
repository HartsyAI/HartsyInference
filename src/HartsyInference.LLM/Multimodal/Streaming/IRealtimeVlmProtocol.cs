using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.Multimodal.Streaming;

/// <summary>One encoded frame or image for a vision-language model: its merged vision tokens, grid and presentation time.</summary>
/// <param name="MergedTokens">The merged vision tokens <c>[h·w, hidden]</c> F32 (borrowed).</param>
/// <param name="TimestampSeconds">The frame's time in its stream; 0 for a still image.</param>
public readonly record struct VisionFrameFeatures(Tensor MergedTokens, int MergedHeight, int MergedWidth, double TimestampSeconds);

/// <summary>The prompt text a streaming or video VLM is trained on: how a chat starts, how a user prompt and a frame are spliced into
/// a running stream, and how a whole video is laid out offline. The generic realtime loop and video paths build every segment
/// through it, so a model plugs in by supplying its own templates and control tokens; the math stays in the model's decoder.</summary>
public interface IRealtimeVlmProtocol
{
    /// <summary>The token each frame or image occupies in the text (one per frame).</summary>
    int ImagePlaceholderId { get; }

    /// <summary>The token the model emits when it has nothing to say, or null for a model without one (its loop never idles).</summary>
    int? SilenceTokenId { get; }

    /// <summary>Tokens a live session never emits (their logits are dropped before sampling), so the silence token stays the only idle
    /// signal; empty when the model needs none.</summary>
    IReadOnlyList<int> SuppressedTokenIds { get; }

    /// <summary>The system prompt a live session uses when the caller gives none.</summary>
    string DefaultSystemPrompt { get; }

    /// <summary>The text that opens a live session: the chat template over the system prompt and the first user prompt, ending where the
    /// assistant's turn begins.</summary>
    string InitialPrompt(string systemPrompt, string userPrompt);

    /// <summary>The text a user prompt adds mid-stream: it closes the assistant turn in progress and opens a new one.</summary>
    string PromptSegment(string prompt);

    /// <summary>The text one live frame adds at <paramref name="timestampSeconds"/>.</summary>
    string FrameSegment(double timestampSeconds);

    /// <summary>Text appended after the prompts of a cycle that also brings frames, so the turn starts the way training data does
    /// (MOSS-VL: a literal silence token); null when the model needs none.</summary>
    string? AfterPromptsWithFrames { get; }

    /// <summary>A one-turn offline request: the user turn holding <paramref name="images"/> still images, then a video of the given frame
    /// <paramref name="videoTimestamps"/> (none for no video), then <paramref name="question"/>, ending where the answer begins.</summary>
    string OfflinePrompt(int images, IReadOnlyList<double> videoTimestamps, string question);
}
