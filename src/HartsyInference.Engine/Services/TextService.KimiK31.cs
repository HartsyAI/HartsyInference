using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Imaging;
using HartsyInference.Engine.Requests;
using HartsyInference.LLM.ChatTemplates;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Kimi;

namespace HartsyInference.Engine.Services;

public sealed partial class TextService
{
    /// <summary>The Kimi K3.1 image route. The request's images belong to its last user turn; each is encoded by the vision tower and spliced at its
    /// media pad in the XTML prompt, and the decoder samples the reply. Images elsewhere, alpha, tool schemas and an image the checkpoint cannot take
    /// are refused by name.</summary>
    private static GenOutcome RunKimiVision(KimiK31TextModel kimi, TextRequest request, Action<TextChunk>? sink, CancellationToken cancel)
    {
        cancel.ThrowIfCancellationRequested();
        int lastUser = request.Messages.Count - 1;
        while (lastUser >= 0 && request.Messages[lastUser].Role != TextRole.User) lastUser--;
        for (int i = 0; i < request.Messages.Count; i++)
        {
            if (i != lastUser && request.Messages[i].Images is { Count: > 0 })
                throw new NotSupportedException("Kimi K3.1 image input is supported on the last user turn only.");
        }
        IReadOnlyList<ImageData> images = lastUser >= 0 && request.Messages[lastUser].Images is { Count: > 0 } attached ? attached : [];
        if (images.Count == 0) throw new NotSupportedException("Kimi K3.1 image input needs an image on the last user turn.");
        foreach (ImageData image in images)
        {
            if (image.HasAlpha) throw new NotSupportedException("Kimi K3.1 image input with an alpha channel is not composited yet.");
        }

        GenerationRequest genRequest = BuildRequest(request, rawCompletion: false, kimi.Tokenizer);
        if (genRequest.Tools is { Count: > 0 }) throw new NotSupportedException("Kimi K3.1 tool schemas are not rendered by the Text path yet.");
        List<ChatMessage> messages = [.. genRequest.EffectiveMessages() ?? throw new HartsyInferenceException("The request has no messages.")];
        int userTurn = messages.FindLastIndex(m => m.Role == "user");
        if (userTurn < 0) throw new HartsyInferenceException("The request has no user turn to attach the image to.");
        List<ContentBlock> blocks = [];
        // Writes the original size, not the resized one. Unverified against the reference processor;
        // the placeholder text it writes for a large image (encoding_k3.py) settles it.
        for (int k = 0; k < images.Count; k++) blocks.Add(new ImageBlock(k) { Width = images[k].Width, Height = images[k].Height });
        blocks.Add(new TextBlock(VisionQuestion(request)));
        messages[userTurn] = messages[userTurn] with { Blocks = blocks };

        int[] promptIds = kimi.Template.Encode(kimi.Tokenizer, messages, addGenerationPrompt: true, enableThinking: genRequest.EnableThinking);
        List<RawRgbFrame> frames = [.. images.Select(image => new RawRgbFrame(image.Rgb, image.Width, image.Height))];
        // The generator loads the tower here, after every refusal above, so a refused request never reads the tower.
        KimiK31VisionGenerator generator = new(kimi.Generation, kimi.Tokenizer, kimi.Vision);
        int maxTokens = request.MaxTokens > 0 ? request.MaxTokens : 512;
        KimiK31VisionReply reply = generator.Generate(promptIds, frames, BuildVisionSampling(request), maxTokens,
            sink is null ? null : delta => sink(new TextChunk { Kind = TextChunkKind.Chunk, Text = delta }), cancel);
        return new GenOutcome(reply.Text, reply.HitTokenLimit ? StopReason.Length : StopReason.Stop, reply.PromptTokens, reply.Tokens);
    }
}
