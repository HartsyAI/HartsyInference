using HartsyInference.Core.Exceptions;
using HartsyInference.LLM.Multimodal;
using HartsyInference.LLM.Sampling;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.LLM.Hy4Flash;

/// <summary>What an image reply returns: its text, the prompt's length in tokens, the reply's length, and whether it stopped at the token limit.</summary>
public sealed record Hy4FlashVisionReply(string Text, int PromptTokens, int Tokens, bool HitTokenLimit);

/// <summary>Answers a Hy4 Flash prompt whose turns carry images: the vision tower encodes each image, the model prefills the expanded prompt with the
/// features at its placeholder runs, and the decoder samples the reply.</summary>
/// <remarks>The prompt carries each image's expanded placeholder run (<see cref="Hy4FlashChatTemplate.EncodeWithImages"/>), in the order of the images.
/// Not thread-safe: run one call at a time per model. The model is used directly, not through the generation adapter, which refuses embeddings.</remarks>
public sealed class Hy4FlashVisionGenerator
{
    private readonly Hy4FlashModel _model;
    private readonly Hy4FlashVisionTower _tower;
    private readonly ILlmTokenizer _tokenizer;

    /// <param name="model">The loaded model; not disposed by this generator.</param>
    /// <param name="tower">The checkpoint's vision tower; not disposed by this generator.</param>
    /// <param name="tokenizer">The checkpoint's tokenizer, for the stop ids and the decode.</param>
    public Hy4FlashVisionGenerator(Hy4FlashModel model, Hy4FlashVisionTower tower, ILlmTokenizer tokenizer)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));
        _tower = tower ?? throw new ArgumentNullException(nameof(tower));
        _tokenizer = tokenizer ?? throw new ArgumentNullException(nameof(tokenizer));
    }

    /// <summary>Samples a reply of at most <paramref name="maxTokens"/> tokens to <paramref name="promptIds"/> and <paramref name="images"/>.
    /// Stops at a stop id, which is not returned. The reply is capped at the room the prompt leaves in the context.</summary>
    /// <exception cref="ArgumentException">The prompt's placeholder runs do not match the images.</exception>
    /// <exception cref="HartsyInferenceException">The prompt alone does not fit the loaded context.</exception>
    /// <exception cref="OperationCanceledException"><paramref name="cancel"/> is set; checked per image and per sampled token.</exception>
    public Hy4FlashVisionReply Generate(int[] promptIds, IReadOnlyList<Hy4FlashPatchedImage> images, SamplingOptions sampling, int maxTokens,
        Action<string>? onDelta, CancellationToken cancel)
    {
        ArgumentNullException.ThrowIfNull(promptIds);
        ArgumentNullException.ThrowIfNull(images);
        ArgumentNullException.ThrowIfNull(sampling);
        ArgumentOutOfRangeException.ThrowIfLessThan(maxTokens, 1);
        if (promptIds.Length >= _model.MaxPositions)
            throw new HartsyInferenceException($"The prompt holds {promptIds.Length} tokens, but the loaded context holds {_model.MaxPositions}.");

        List<Hy4FlashImage> features = EncodeImages(images, cancel);
        using Hy4FlashSequenceState state = _model.CreateState();
        float[] logits = _model.Forward(promptIds, state, allRows: false, images: features);
        SamplerChain sampler = SamplerChain.FromOptions(sampling, _tokenizer, _model.Config.VocabSize);
        HashSet<int> stops = [.. _tokenizer.StopIds];
        List<int> history = [];
        int first = sampler.Next(logits.AsSpan(), history);
        int budget = Math.Min(maxTokens, _model.MaxPositions - promptIds.Length);
        (string text, int tokens, bool hitLimit) = VlmDecodeOps.DecodeLoop(_tokenizer, first,
            token => sampler.Next(_model.Forward([token], state).AsSpan(), history), stops, budget, history, onDelta, cancel);
        return new Hy4FlashVisionReply(text, promptIds.Length, tokens, hitLimit);
    }

    // The logits of the last prompt position after the spliced prompt is prefilled, for a caller that checks the prefill itself.
    internal float[] PrefillLogits(int[] promptIds, IReadOnlyList<Hy4FlashPatchedImage> images)
    {
        using Hy4FlashSequenceState state = _model.CreateState();
        return _model.Forward(promptIds, state, allRows: false, images: EncodeImages(images, CancellationToken.None));
    }

    // Runs each image through the vision tower, in order; the features are what the model's placeholder runs take.
    private List<Hy4FlashImage> EncodeImages(IReadOnlyList<Hy4FlashPatchedImage> images, CancellationToken cancel)
    {
        List<Hy4FlashImage> features = new(images.Count);
        foreach (Hy4FlashPatchedImage image in images)
        {
            cancel.ThrowIfCancellationRequested();
            features.Add(new Hy4FlashImage(image.GridH, image.GridW, _tower.Forward(image.PixelValues, image.GridH, image.GridW)));
        }
        return features;
    }
}
