using System.Text;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Imaging;
using HartsyInference.LLM.Multimodal.Streaming;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.LLM.Models.MossVl;

/// <summary>MOSS-VL-Realtime's prompt layouts, as the released processor and realtime session build them.</summary>
/// <remarks>
/// <para>Offline (<c>processing_moss_vl.py</c> with the checkpoint's chat template): a still image is a bare <c>&lt;|image_pad|&gt;</c>; a
/// video is one <c>&lt;|vision_start|&gt;…&lt;|vision_end|&gt;</c> block with <c>&lt;|time_start|&gt;T seconds&lt;|time_end|&gt;&lt;|image_pad|&gt;</c> per
/// frame; images come before videos, and the question last.</para>
/// <para>Live (<c>MossVLRealtimeSession</c>): each frame is its own <c>&lt;|vision_start|&gt;&lt;|time_start|&gt;T seconds&lt;|time_end|&gt;&lt;|image_pad|&gt;&lt;|vision_end|&gt;</c>;
/// a prompt closes the open assistant turn and opens a new one; a cycle with both prompts and frames puts a literal
/// <c>&lt;|silence|&gt;</c> between them. Times are formatted as Python's <c>f"{t:.1f}"</c>.</para>
/// </remarks>
public sealed class MossVlRealtimeProtocol : IRealtimeVlmProtocol
{
    /// <summary>The live session's default system prompt (<c>MossVLRealtimeSession._run</c>).</summary>
    public const string SessionSystemPrompt =
        "You are a helpful AI assistant specializing in real-time video analysis. The video streams to you frame by frame. At every "
        + "frame, you decide independently whether to respond or stay silent — output `<|silence|>` when nothing relevant has happened, "
        + "and respond when the visual content warrants it.";

    private const string ImagePad = "<|image_pad|>";

    /// <param name="config">The model config (image token id).</param>
    /// <param name="tokenizer">The model tokenizer, which must know the <c>&lt;|silence|&gt;</c> and time tokens.</param>
    /// <exception cref="HartsyInferenceException">The tokenizer lacks a control token the protocol needs.</exception>
    public MossVlRealtimeProtocol(MossVlConfig config, ILlmTokenizer tokenizer)
    {
        ImagePlaceholderId = config.ImageTokenId;
        SilenceTokenId = tokenizer.SpecialId("<|silence|>")
            ?? throw new HartsyInferenceException("The MOSS-VL tokenizer has no <|silence|> token; it is not the realtime checkpoint's tokenizer.");
        foreach (string token in new[] { "<|time_start|>", "<|time_end|>", "<|vision_start|>", "<|vision_end|>", ImagePad })
            if (tokenizer.SpecialId(token) is null)
                throw new HartsyInferenceException($"The MOSS-VL tokenizer has no {token} token.");
        // The reference suppresses an ellipsis control token when the vocabulary has one; the released tokenizer does not.
        SuppressedTokenIds = tokenizer.SpecialId("<|...|>") is int ellipsis ? [ellipsis] : [];
    }

    /// <summary>The protocol with explicit token ids, for tests of its text layouts.</summary>
    internal MossVlRealtimeProtocol(int imagePlaceholderId, int silenceTokenId)
    {
        ImagePlaceholderId = imagePlaceholderId;
        SilenceTokenId = silenceTokenId;
    }

    public int ImagePlaceholderId { get; }

    public int? SilenceTokenId { get; }

    public IReadOnlyList<int> SuppressedTokenIds { get; } = [];

    public string DefaultSystemPrompt => SessionSystemPrompt;

    public string? AfterPromptsWithFrames => "<|silence|>";

    public string InitialPrompt(string systemPrompt, string userPrompt) =>
        $"<|im_start|>system\n{systemPrompt}<|im_end|>\n<|im_start|>user\n{userPrompt}<|im_end|>\n<|im_start|>assistant\n";

    public string PromptSegment(string prompt) => $"<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n";

    public string FrameSegment(double timestampSeconds) =>
        $"<|vision_start|><|time_start|>{VideoTimestampText.Seconds1(timestampSeconds)} seconds<|time_end|>{ImagePad}<|vision_end|>";

    public string OfflinePrompt(int images, IReadOnlyList<double> videoTimestamps, string question)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(images);
        StringBuilder user = new("<|im_start|>user\n");
        for (int i = 0; i < images; i++) user.Append(ImagePad);
        if (videoTimestamps.Count > 0)
        {
            user.Append("<|vision_start|>");
            foreach (double t in videoTimestamps)
                user.Append("<|time_start|>").Append(VideoTimestampText.Seconds1(t)).Append(" seconds<|time_end|>").Append(ImagePad);
            user.Append("<|vision_end|>");
        }
        // The processor strips leading newlines from the text part.
        user.Append(question.TrimStart('\n')).Append("<|im_end|>\n<|im_start|>assistant\n");
        return user.ToString();
    }
}
