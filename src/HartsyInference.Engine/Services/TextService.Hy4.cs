using HartsyInference.Core.Exceptions;
using HartsyInference.Engine.Requests;
using HartsyInference.LLM.ChatTemplates;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Hy4Flash;

namespace HartsyInference.Engine.Services;

public sealed partial class TextService
{
    /// <summary>The Hy4 Flash image route. The request's images belong to its last user turn; each is preprocessed to its patch rows, the template expands
    /// its pad to the image's feature rows, the vision tower encodes it, and the model prefills and samples the reply. A reply that opened a reasoning
    /// block is split at the Hy4 close marker; its reasoning is streamed separately, and its content after the marker. Images elsewhere, alpha, tool
    /// schemas and a prompt that does not fit are refused by name.</summary>
    private static GenOutcome RunHy4Vision(Hy4FlashTextModel hy4, TextRequest request, Action<TextChunk>? sink, CancellationToken cancel)
    {
        cancel.ThrowIfCancellationRequested();
        int lastUser = request.Messages.Count - 1;
        while (lastUser >= 0 && request.Messages[lastUser].Role != TextRole.User) lastUser--;
        for (int i = 0; i < request.Messages.Count; i++)
        {
            if (i != lastUser && request.Messages[i].Images is { Count: > 0 })
                throw new NotSupportedException("Hy4 Flash image input is supported on the last user turn only.");
        }
        IReadOnlyList<ImageData> images = lastUser >= 0 && request.Messages[lastUser].Images is { Count: > 0 } attached ? attached : [];
        if (images.Count == 0) throw new NotSupportedException("Hy4 Flash image input needs an image on the last user turn.");
        foreach (ImageData image in images)
        {
            if (image.HasAlpha) throw new NotSupportedException("Hy4 Flash image input with an alpha channel is not composited yet.");
        }

        GenerationRequest genRequest = BuildRequest(request, rawCompletion: false, hy4.Tokenizer);
        if (genRequest.Tools is { Count: > 0 }) throw new NotSupportedException("Hy4 Flash tool schemas are not rendered on the image route yet.");
        if (hy4.Template is not Hy4FlashChatTemplate template) throw new InvalidOperationException("The Hy4 Flash checkpoint must use its own chat template for images.");
        List<ChatMessage> messages = [.. genRequest.EffectiveMessages() ?? throw new HartsyInferenceException("The request has no messages.")];
        int userTurn = messages.FindLastIndex(m => m.Role == "user");
        if (userTurn < 0) throw new HartsyInferenceException("The request has no user turn to attach the image to.");

        // The patch and merge are the loaded checkpoint's own (the released 16 and 2; the tiny test checkpoint's 4 and 2).
        Hy4FlashVisionConfig vision = hy4.Model.Config.Vision;
        List<Hy4FlashPatchedImage> patched = [.. images.Select(image => Hy4FlashImagePreprocessor.Preprocess(image.Rgb, image.Width, image.Height,
            vision.PatchSize, vision.SpatialMergeSize))];
        List<ContentBlock> blocks = [];
        for (int k = 0; k < images.Count; k++) blocks.Add(new ImageBlock(k));
        blocks.Add(new TextBlock(VisionQuestion(request)));
        messages[userTurn] = messages[userTurn] with { Blocks = blocks };

        int[] promptIds = template.EncodeWithImages(hy4.Tokenizer, messages, addGenerationPrompt: true, request.EnableThinking, tools: null,
            request.ReasoningEffort, [.. patched.Select(p => (p.GridH, p.GridW))], vision.SpatialMergeSize);
        bool reasoningOpen = template.ResolveParserState(hy4.Tokenizer, messages, request.EnableThinking, tools: null, request.ReasoningEffort).ReasoningOpen;

        // A reply inside a reasoning block is streamed only after it is split, so its reasoning never reads as the answer.
        Action<string>? stream = sink is null || reasoningOpen ? null : delta => sink(new TextChunk { Kind = TextChunkKind.Chunk, Text = delta });
        int maxTokens = request.MaxTokens > 0 ? request.MaxTokens : 512;
        Hy4FlashVisionGenerator generator = new(hy4.Model, hy4.Vision, hy4.Tokenizer);
        Hy4FlashVisionReply reply = generator.Generate(promptIds, patched, BuildVisionSampling(request), maxTokens, stream, cancel);

        (string reasoning, string answer) = SplitVisionReasoning(reply.Text, reasoningOpen, !reply.HitTokenLimit,
            Hy4FlashChatTemplate.ReasoningMarkers[0], Hy4FlashChatTemplate.ReasoningMarkers[1]);
        if (reasoningOpen && sink is not null)
        {
            if (reasoning.Length > 0) sink(new TextChunk { Kind = TextChunkKind.Reasoning, Text = reasoning });
            if (answer.Length > 0) sink(new TextChunk { Kind = TextChunkKind.Chunk, Text = answer });
        }
        StopReason stop = reply.HitTokenLimit ? StopReason.Length : StopReason.Stop;
        return new GenOutcome(answer, stop, reply.PromptTokens, reply.Tokens, Reasoning: reasoning);
    }
}
