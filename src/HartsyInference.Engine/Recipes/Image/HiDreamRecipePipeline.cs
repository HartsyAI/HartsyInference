using MergedLoraStack = HartsyInference.ModelAssets.Lora.LoraStack;
using System.Globalization;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.Diffusion.Prompting;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;

using HartsyInference.Engine.Features;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>A constructed HiDream-I1 pipeline driven against the native <see cref="ImageRequest"/>. <see cref="HiDreamPipeline"/> owns all four encoder forwards, so this only tokenizes: one CLIP BPE pass reused for both CLIP-L and CLIP-G (the reference feeds identical ids to both), a T5 pass with its attention mask, and a Llama-3.1 pass — then calls <see cref="HiDreamPipeline.GenerateFromTokens"/>. Mirrors the SwarmUI backend's <c>HiDreamLoader.Generate</c> drive path. Wraps the constructed HiDream pipeline plus its tokenizers and heavyweight components, taking ownership of every disposable.</summary>
public sealed class HiDreamRecipePipeline(HiDreamPipeline pipeline, ClipTokenizer clipTokenizer, T5Tokenizer t5Tokenizer,
    LlamaTokenizer llamaTokenizer, T5TextEncoder t5, LlamaStyleEncoder llama, HiDreamTransformer transformer,
    IReadOnlyList<IDisposable> componentSources, MergedLoraStack? loraStack = null) : IRecipePipeline
{
    private readonly HiDreamPipeline _pipeline = pipeline;
    private readonly ClipTokenizer _clipTokenizer = clipTokenizer;
    private readonly T5Tokenizer _t5Tokenizer = t5Tokenizer;
    private readonly LlamaTokenizer _llamaTokenizer = llamaTokenizer;
    private readonly T5TextEncoder _t5 = t5;
    private readonly LlamaStyleEncoder _llama = llama;
    private readonly HiDreamTransformer _transformer = transformer;
    private readonly IReadOnlyList<IDisposable> _componentSources = componentSources;

    private readonly MergedLoraStack? _loraStack = loraStack;

    /// <inheritdoc/>
    public ImageResult Generate(ImageRequest request, IProgress<StepPreview>? progress, CancellationToken cancel)
    {
        cancel.ThrowIfCancellationRequested();
        string prompt = request.Prompt;
        string negative = request.NegativePrompt ?? "";
        int steps = request.Steps ?? HiDreamRecipe.FamilyDefaults.Steps;
        float cfg = request.CfgScale ?? HiDreamRecipe.FamilyDefaults.CfgScale;

        // TODO(E-IMG-4/5): LoRA, ControlNet, IP-Adapter, refiner, regional prompting and
        // ImageRequest.Components overrides are deferred — text-to-image only.
        // Declaring ComfyBlend is what stops ImagesService collapsing `(word:N)`, so the recipe owns the grammar
        // now. The CLIP arms take no weights: HiDream keeps only their pooled vectors and discards the hidden
        // states, and ComfyUI's blend rewrites hidden states — wiring them would be a no-op.
        string baseText = PromptWeighting.Join(PromptWeighting.Parse(PromptTagFlattening.Flatten(prompt)));
        string negBaseText = PromptWeighting.Join(PromptWeighting.Parse(PromptTagFlattening.Flatten(negative)));
        int[] clipTokens = _clipTokenizer.Encode(baseText);
        int[] negClipTokens = _clipTokenizer.Encode(negBaseText);
        int eosPos = ClipTokenizer.FindEosPosition(clipTokens);
        int negEosPos = ClipTokenizer.FindEosPosition(negClipTokens);

        // Always tokenize the negative: the pipeline's parameters are non-optional and it decides internally
        // whether to run the negative pass (cfg > 1).
        (int[] t5Tokens, float[]? t5Weights) = T5WeightedConditioning.Tokenize(_t5Tokenizer, prompt);
        (int[] negT5Tokens, float[]? negT5Weights) = T5WeightedConditioning.Tokenize(_t5Tokenizer, negative);
        int[] t5Mask = T5Tokenizer.CreateAttentionMask(t5Tokens);
        int[] negT5Mask = T5Tokenizer.CreateAttentionMask(negT5Tokens);
        int[] emptyT5 = T5WeightedConditioning.EmptyTokens(_t5Tokenizer);

        (int[] llamaTokens, float[]? llamaWeights) = TokenizeLlamaWeighted(prompt);
        (int[] negLlamaTokens, float[]? negLlamaWeights) = TokenizeLlamaWeighted(negative);
        int[] emptyLlama = _llamaTokenizer.Encode("");

        (int reqWidth, int reqHeight) = RecipeRequestMapper.Size(request);
        using Img2ImgResolver.Img2ImgSpec? img2img = RecipeImg2ImgBinder.Resolve(request, reqWidth, reqHeight);
        TextToImageRequest inner = RecipeImg2ImgBinder.Apply(
            new TextToImageRequest
            {
                SeamlessTiling = request.SeamlessTiling,
                    VariationSeed = request.VariationSeed?.Seed ?? -1,
                    VariationSeedStrength = request.VariationSeed?.Strength ?? 0,
                Prompt = prompt,
                NegativePrompt = negative,
                Width = request.Width,
                Height = request.Height,
                Steps = steps,
                CfgScale = cfg,
                Seed = RecipeRequestMapper.MapSeed(request.Seed),
                // Routed through the resolver rather than read raw, so an unavailable sampler is refused by name here —
                // before the checkpoint loads — instead of deep inside the pipeline, or silently dropped.
                Scheduler = SamplingParamResolver.ResolveSchedulerName(request),
            },
            img2img);

        Action<GenerationProgress> bridge = RecipeProgressAdapter.Create(progress, cancel);

        (byte[] rgb, int outW, int outH, int usedSeed) = _pipeline.GenerateFromTokens(
            clipTokens, negClipTokens,
            clipTokens, negClipTokens,
            eosPos, negEosPos,
            eosPos, negEosPos,
            t5Tokens, negT5Tokens,
            t5Mask, negT5Mask,
            llamaTokens, negLlamaTokens,
            inner, bridge,
            t5Weights, negT5Weights, llamaWeights, negLlamaWeights,
            emptyT5, T5Tokenizer.CreateAttentionMask(emptyT5), emptyLlama);

        return new ImageResult
        {
            Rgb = rgb,
            Width = outW,
            Height = outH,
            Seed = usedSeed,
            Meta = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
            {
                ["arch"] = "hidream",
                ["size"] = $"{outW}x{outH}",
                ["seed"] = usedSeed.ToString(CultureInfo.InvariantCulture),
                ["steps"] = steps.ToString(CultureInfo.InvariantCulture),
                ["cfg"] = cfg.ToString(CultureInfo.InvariantCulture),
            },
        };
    }

    /// <summary>Mirrors <see cref="LlamaTokenizer.Encode"/>'s layout — BOS, then the prompt, then right-pad to
    /// the fixed window — while carrying one weight per row. BOS and pad rows weigh 1: they are not part of the
    /// prompt, and blending them would pull the padding toward the empty encode along with the words.</summary>
    private (int[] Tokens, float[]? Weights) TokenizeLlamaWeighted(string prompt)
    {
        IReadOnlyList<WeightedSpan> spans = PromptWeighting.Parse(PromptTagFlattening.Flatten(prompt));
        if (!PromptWeighting.HasWeights(spans))
        {
            return (_llamaTokenizer.Encode(PromptWeighting.Join(spans)), null);
        }
        WeightedTokenSequence built = WeightedTokenBuilder.Build(spans, _llamaTokenizer.EncodeRaw, [], []);
        int window = _llamaTokenizer.MaxLength;
        int[] tokens = new int[window];
        float[] weights = new float[window];
        Array.Fill(tokens, LlamaTokenizer.PadTokenId);
        Array.Fill(weights, 1f);
        tokens[0] = LlamaTokenizer.BosTokenId;
        int real = Math.Min(built.Tokens.Length, window - 1);
        Array.Copy(built.Tokens, 0, tokens, 1, real);
        Array.Copy(built.Weights, 0, weights, 1, real);
        return (tokens, weights);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _pipeline.Dispose();
        _clipTokenizer.Dispose();
        _t5Tokenizer.Dispose();
        _llamaTokenizer.Dispose();
        _t5.Dispose();
        _llama.Dispose();
        _transformer.Dispose();
        foreach (IDisposable source in _componentSources)
        {
            source.Dispose();
        }
        // Last: the stack owns the merged weight tensors the transformer was serving.
        _loraStack?.Dispose();
    }
}
