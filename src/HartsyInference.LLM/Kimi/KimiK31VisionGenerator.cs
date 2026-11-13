using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Imaging;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Kimi.Vision;
using HartsyInference.LLM.Sampling;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.LLM.Kimi;

/// <summary>Answers a Kimi K3.1 prompt with images: the MoonViT-V2 tower encodes each image, its outputs replace the prompt's embeddings at the media
/// pads, and the decoder prefills the spliced sequence and samples the reply.</summary>
/// <remarks>The prompt holds one media pad per image. The splice expands each pad to its image's vision-token count, so the decoder reads the same rows
/// the processor's placeholder stands for. Not thread-safe: run one call at a time per model.</remarks>
public sealed class KimiK31VisionGenerator
{
    private readonly KimiK31GenerationModel _generation;
    private readonly ILlmTokenizer _tokenizer;
    private readonly KimiVisionRoute _route;

    /// <param name="generation">The loaded decoder.</param>
    /// <param name="tokenizer">The checkpoint's tokenizer, for the stop ids and the decode.</param>
    /// <param name="vision">The checkpoint's image inputs. Loads the vision tower on the first call, if it has not been loaded yet.</param>
    /// <exception cref="NotSupportedException">The checkpoint refuses images; the message gives the reason.</exception>
    public KimiK31VisionGenerator(KimiK31GenerationModel generation, ILlmTokenizer tokenizer, KimiVisionInput vision)
    {
        ArgumentNullException.ThrowIfNull(generation);
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentNullException.ThrowIfNull(vision);
        _route = vision.Route();
        _generation = generation;
        _tokenizer = tokenizer;
    }

    /// <summary>Samples a reply of at most <paramref name="maxTokens"/> tokens to <paramref name="promptIds"/> and <paramref name="images"/>, in
    /// placeholder order. Stops at a stop id, which is not returned.</summary>
    /// <exception cref="ArgumentException">The prompt's media pads do not match the images, or an image is malformed.</exception>
    /// <exception cref="HartsyInferenceException">The spliced prompt and the reply do not fit the loaded context.</exception>
    /// <exception cref="OperationCanceledException"><paramref name="cancel"/> is set; checked per image, per prefill row and per sampled token.</exception>
    public KimiK31VisionReply Generate(int[] promptIds, IReadOnlyList<RawRgbFrame> images, SamplingOptions sampling, int maxTokens,
        CancellationToken cancel)
    {
        ArgumentNullException.ThrowIfNull(sampling);
        ArgumentOutOfRangeException.ThrowIfLessThan(maxTokens, 1);
        Prefilled prefilled = Prefill(promptIds, images, maxTokens, cancel);
        using KimiK31State state = prefilled.State;
        KimiK31Model decoder = _generation.Decoder;
        SamplerChain sampler = SamplerChain.FromOptions(sampling, _tokenizer, _generation.DecoderConfig.Vocab);
        HashSet<int> stops = new(_tokenizer.StopIds);
        List<int> history = [];
        List<int> reply = [];
        float[] logits = prefilled.Logits;
        bool stopped = false;
        for (int step = 0; step < maxTokens; step++)
        {
            cancel.ThrowIfCancellationRequested();
            int token = sampler.Next(logits.AsSpan(), history);
            if (stops.Contains(token))
            {
                stopped = true;
                break;
            }
            reply.Add(token);
            history.Add(token);
            if (step + 1 < maxTokens) logits = decoder.ForwardToken(token, state);
        }
        return new KimiK31VisionReply(_tokenizer.Decode(reply), reply.Count, HitTokenLimit: !stopped);
    }

    // The logits after the spliced prompt, for a caller that samples its own reply.
    internal float[] PrefillLogits(int[] promptIds, IReadOnlyList<RawRgbFrame> images)
    {
        Prefilled prefilled = Prefill(promptIds, images, extraTokens: 1, CancellationToken.None);
        prefilled.State.Dispose();
        return prefilled.Logits;
    }

    // Encodes the images, splices them into the prompt and prefills the sequence. The caller owns the state.
    private Prefilled Prefill(int[] promptIds, IReadOnlyList<RawRgbFrame> images, int extraTokens, CancellationToken cancel)
    {
        ArgumentNullException.ThrowIfNull(promptIds);
        ArgumentNullException.ThrowIfNull(images);
        KimiK31Config config = _generation.DecoderConfig;
        int hidden = config.Hidden, padId = _route.PadId;
        List<Tensor> features = [];
        try
        {
            foreach (RawRgbFrame image in images)
            {
                cancel.ThrowIfCancellationRequested();
                features.Add(Encode(image));
            }
            int[] ids = ExpandPads(promptIds, features, padId, hidden);
            int capacity = ids.Length + extraTokens;
            if (capacity > _generation.MaxSequenceTokens)
                throw new HartsyInferenceException($"The prompt is {ids.Length} tokens with its images and the reply {extraTokens}; the model holds {_generation.MaxSequenceTokens}.");

            // The splice overwrites the pad rows, so they look up token 0 rather than the pad's own embedding.
            int[] lookup = new int[ids.Length];
            for (int i = 0; i < ids.Length; i++) lookup[i] = ids[i] == padId ? 0 : ids[i];
            using Tensor embeds = new(new TensorShape(1, ids.Length, hidden), DType.F32);
            _generation.Decoder.EmbedTokens(lookup, embeds.AsSpan<float>());
            VisionTokenSplice.Apply(embeds, ids, padId, features);

            KimiK31State state = new(config, capacity, _generation.Decoder.Ranks);
            try
            {
                ReadOnlySpan<float> rows = embeds.AsReadOnlySpan<float>();
                float[] last = [];
                for (int r = 0; r < ids.Length; r++)
                {
                    cancel.ThrowIfCancellationRequested();
                    last = _generation.Decoder.ForwardEmbeddingHidden(rows.Slice(r * hidden, hidden), state);
                }
                return new Prefilled(_generation.Decoder.ProjectHidden(last), state);
            }
            catch
            {
                state.Dispose();
                throw;
            }
        }
        finally
        {
            foreach (Tensor f in features) f.Dispose();
        }
    }

    // The processor's pixels through the tower: [1, vision tokens, decoder hidden], which the caller disposes.
    private Tensor Encode(RawRgbFrame image)
    {
        image.Validate();
        using Tensor pixels = KimiImagePreprocessor.Preprocess(image.Rgb.Span, image.Width, image.Height, _route.Image);
        return _route.Encoder.Encode(_generation.OutputBackend, pixels);
    }

    // Each media pad becomes as many pads as its image has vision tokens, in order; the splice then writes one feature row per pad.
    private static int[] ExpandPads(int[] promptIds, IReadOnlyList<Tensor> features, int padId, int hidden)
    {
        List<int> ids = new(promptIds.Length);
        int image = 0;
        foreach (int id in promptIds)
        {
            if (id != padId)
            {
                ids.Add(id);
                continue;
            }
            if (image == features.Count)
                throw new ArgumentException($"The prompt holds more media pads than the {features.Count} image(s).", nameof(promptIds));
            int rows = (int)(features[image].ElementCount / hidden);
            for (int r = 0; r < rows; r++) ids.Add(padId);
            image++;
        }
        if (image != features.Count)
            throw new ArgumentException($"The prompt holds {image} media pad(s) for {features.Count} image(s).", nameof(promptIds));
        return [.. ids];
    }

    // The logits after the spliced prompt, and the state that holds it.
    private sealed record Prefilled(float[] Logits, KimiK31State State);
}
