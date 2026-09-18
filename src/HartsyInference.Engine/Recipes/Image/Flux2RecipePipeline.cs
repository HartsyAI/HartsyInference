using MergedLoraStack = HartsyInference.ModelAssets.Lora.LoraStack;
using System.Globalization;
using System.Runtime.InteropServices;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Prompting;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;

using HartsyInference.Engine.Features;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>A constructed Flux.2 pipeline driven against the native <see cref="ImageRequest"/>. <see cref="Flux2Pipeline"/> owns the text encoder, so this only produces the token ids — the embedded Qwen3 chat template for Klein, or the spliced Mistral tekken conditioning ids for Dev — and calls <see cref="Flux2Pipeline.GenerateFromTokens"/>. Mirrors the SwarmUI backend's <c>Flux2Loader.Generate</c> text-to-image drive path. Wraps the constructed Flux.2 pipeline plus its tokenizer, taking ownership of every disposable. Exactly one of <paramref name="qwenTokenizer"/> (Klein) / <paramref name="mistralTokenizer"/> (Dev) is non-null. <paramref name="checkpoint"/> holds whatever keeps the transformer's weights alive — the checkpoint's memory map, which pass-through tensors still point into, and any copies the backend needed widened.</summary>
public sealed class Flux2RecipePipeline(Flux2Pipeline pipeline, Flux2Config config, Qwen3Tokenizer? qwenTokenizer, ErnieTokenizer? mistralTokenizer,
    string mistralSystemPrompt, LlamaStyleEncoder encoder, List<SafeTensorsLoader> loaders, IDisposable? checkpoint = null,
    MergedLoraStack? loraStack = null) : IRecipePipeline
{
    private readonly Flux2Pipeline _pipeline = pipeline;
    private readonly Flux2Config _config = config;
    private readonly Qwen3Tokenizer? _qwenTokenizer = qwenTokenizer;
    private readonly ErnieTokenizer? _mistralTokenizer = mistralTokenizer;
    private readonly string _mistralSystemPrompt = mistralSystemPrompt;
    private readonly LlamaStyleEncoder _encoder = encoder;
    private readonly List<SafeTensorsLoader> _loaders = loaders;
    private readonly IDisposable? _checkpoint = checkpoint;

    private readonly MergedLoraStack? _loraStack = loraStack;

    /// <summary>A Klein checkpoint (no guidance embedding) is CFG-distilled and few-step, so it resolves against <see cref="Flux2Recipe.KleinDefaults"/> rather than Dev's 50 steps.</summary>
    public ImageDefaults? VariantDefaults => _config.GuidanceEmbed ? Flux2Recipe.FamilyDefaults : Flux2Recipe.KleinDefaults;

    /// <inheritdoc/>
    public ImageResult Generate(ImageRequest request, IProgress<StepPreview>? progress, CancellationToken cancel)
    {
        cancel.ThrowIfCancellationRequested();
        string prompt = request.Prompt;
        int steps = request.Steps ?? (_config.GuidanceEmbed ? Flux2Recipe.FamilyDefaults.Steps : Flux2Recipe.KleinDefaults.Steps);
        // Flux.2 rounds image dims down to a multiple of 16 (VAE 8× × 2×2 patch).
        (int reqWidth, int reqHeight) = RecipeRequestMapper.Size(request);
        int width = (reqWidth / 16) * 16;
        int height = (reqHeight / 16) * 16;
        // Klein has no guidance embedding; Dev uses guidance ~3.5 (BFL distillation target).
        float guidance = _config.GuidanceEmbed ? 3.5f : 0f;

        // TODO(E-IMG-4/5): img2img, NegativePrompt/CfgScale mapping, and user component overrides are deferred.
        WeightedTokenSequence tokens = Tokenize(prompt);
        WeightedTokenSequence? promptWeights = tokens.IsUniformlyUnweighted ? null : tokens;

        // Resolved at the 16-rounded size Flux2Pipeline validates against.
        using Img2ImgResolver.Img2ImgSpec? img2img = RecipeImg2ImgBinder.Resolve(request, width, height);
        TextToImageRequest inner = RecipeImg2ImgBinder.Apply(
            new TextToImageRequest
            {
                SeamlessTiling = request.SeamlessTiling,
                    VariationSeed = request.VariationSeed?.Seed ?? -1,
                    VariationSeedStrength = request.VariationSeed?.Strength ?? 0,
                Prompt = prompt,
                Width = width,
                Height = height,
                Steps = steps,
                Seed = RecipeRequestMapper.MapSeed(request.Seed),
                // Routed through the resolver rather than read raw, so an unavailable sampler is refused by name here —
                // before the checkpoint loads — instead of deep inside the pipeline, or silently dropped.
                Scheduler = SamplingParamResolver.ResolveSchedulerName(request),
            },
            img2img);

        Action<GenerationProgress> bridge = RecipeProgressAdapter.Create(progress, cancel);

        RegionalPlan? regionalPlan = null;
        try
        {
            regionalPlan = BuildRegionalPlan(prompt, width, height, steps);

            (byte[] rgb, int outW, int outH, int usedSeed) = _pipeline.GenerateFromTokens(
                tokens.Tokens, inner, guidanceScale: guidance, onProgress: bridge, regionalPlan: regionalPlan,
                promptWeights: promptWeights);

            return new ImageResult
            {
                Rgb = rgb,
                Width = outW,
                Height = outH,
                Seed = usedSeed,
                Meta = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
                {
                    ["arch"] = "flux2",
                    ["size"] = $"{outW}x{outH}",
                    ["seed"] = usedSeed.ToString(CultureInfo.InvariantCulture),
                    ["steps"] = steps.ToString(CultureInfo.InvariantCulture),
                },
            };
        }
        finally
        {
            RegionalPromptResolver.DisposeRegions(regionalPlan);
        }
    }

    /// <summary>Builds a regional-conditioning plan when the prompt carries <c>&lt;region:&gt;</c>/<c>&lt;object:&gt;</c> parts, null otherwise (Tier 3.7). Mirrors <c>Flux1RecipePipeline.BuildRegionalPlan</c>: each region's text is tokenized the SAME way the base prompt was (Klein's chat template vs. Dev's Mistral splice) and encoded through <see cref="Flux2Pipeline.EncodeRegionText"/> — the same text-encoder instance + hidden-layer taps the base prompt uses. <see cref="RegionalPlan.BaseCond"/> is a required field on the resolver's signature that <see cref="Flux2Pipeline.GenerateFromTokens"/>'s regional path never reads (same as Flux.1 — confirmed by inspection: the pipeline builds its own background stream from the base <c>textEmbeddings</c>) — a throwaway placeholder tensor satisfies it.</summary>
    private RegionalPlan? BuildRegionalPlan(string prompt, int width, int height, int steps)
    {
        if (!RegionalPromptResolver.HasRegionParts(prompt))
        {
            return null;
        }
        using Tensor baseCondPlaceholder = new Tensor(new TensorShape(1), DType.F32);
        return RegionalPromptResolver.Resolve(prompt, baseCondPlaceholder, width, height, steps, encodeRegion: text =>
        {
            // Regions are not weighted yet (C.2 work): strip the emphasis rather than let the parens and digits
            // reach the encoder as prose, which is what SwarmUI does for anything it is not weighting.
            return _pipeline.EncodeRegionText(Tokenize(PromptWeighting.Join(PromptWeighting.Parse(text))).Tokens);
        });
    }

    /// <summary>Tokenizes <paramref name="prompt"/> through whichever text stack this checkpoint carries, carrying the
    /// per-token weights its emphasis grammar asks for. Klein's <see cref="Qwen3Tokenizer.EncodeChat"/> merges
    /// <c>user\n</c> with the prompt in ONE BPE call, which is load-bearing for a prompt that begins with whitespace —
    /// so an unweighted prompt keeps that call verbatim and only a weighted one splits per span, which is what SwarmUI
    /// itself does (<c>calc_leaf</c> tokenizes each leaf alone).</summary>
    private WeightedTokenSequence Tokenize(string prompt)
    {
        IReadOnlyList<WeightedSpan> spans = PromptWeighting.Parse(prompt);
        if (_config.TextEncoderType == Flux2TextEncoderType.Mistral)
        {
            return BuildMistralDevTokens(_mistralTokenizer!, spans);
        }
        if (!PromptWeighting.HasWeights(spans))
        {
            int[] plain = _qwenTokenizer!.EncodeChat(PromptWeighting.Join(spans));
            return new WeightedTokenSequence(plain, Ones(plain.Length)) { UniformWeight = 1f };
        }
        (int[] prefix, int[] suffix) = _qwenTokenizer!.ChatTemplateIds();
        WeightedTokenSequence built = WeightedTokenBuilder.Build(spans, _qwenTokenizer.EncodeRaw, prefix, suffix);
        return PadToWindow(built, _qwenTokenizer.MaxLength, Qwen3Tokenizer.BosTokenId);
    }

    /// <summary>Builds Flux.2 Dev conditioning ids: <c>&lt;s&gt;[SYSTEM_PROMPT]sys[/SYSTEM_PROMPT][INST]prompt[/INST]</c>. Special markers are spliced as raw ids (BOS=1, [SYSTEM_PROMPT]=17, [/SYSTEM_PROMPT]=18, [INST]=3, [/INST]=4) around byte-level BPE segments — special strings are pre-token boundaries in the HF reference, so segment-wise encoding is id-exact. No EOS (ComfyUI <c>has_end_token=False</c>). The byte-level map is per-byte, so encoding each emphasis span separately concatenates to the same string the whole-prompt call would have produced.</summary>
    private WeightedTokenSequence BuildMistralDevTokens(ErnieTokenizer tokenizer, IReadOnlyList<WeightedSpan> spans)
    {
        List<int> prefix = new List<int>(256) { 1, 17 };
        prefix.AddRange(tokenizer.EncodeRaw(ByteLevelCodec.Encode(_mistralSystemPrompt)));
        prefix.Add(18);
        prefix.Add(3);
        return WeightedTokenBuilder.Build(spans, text => tokenizer.EncodeRaw(ByteLevelCodec.Encode(text)),
            CollectionsMarshal.AsSpan(prefix), [4]);
    }

    /// <summary>Right-pads (or truncates) to the tokenizer's fixed window the way <see cref="Qwen3Tokenizer.EncodeChat"/>
    /// does, carrying the weights with the ids — pad rows weigh 1, and a truncation that cut only the ids would shift
    /// every emphasis off its own word.</summary>
    private static WeightedTokenSequence PadToWindow(WeightedTokenSequence sequence, int window, int padId)
    {
        int[] tokens = new int[window];
        float[] weights = Ones(window);
        int real = Math.Min(sequence.Tokens.Length, window);
        Array.Copy(sequence.Tokens, tokens, real);
        Array.Copy(sequence.Weights, weights, real);
        for (int i = real; i < window; i++)
        {
            tokens[i] = padId;
        }
        return new WeightedTokenSequence(tokens, weights) { UniformWeight = sequence.UniformWeight };
    }

    private static float[] Ones(int length)
    {
        float[] weights = new float[length];
        Array.Fill(weights, 1f);
        return weights;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _pipeline.Dispose();
        _qwenTokenizer?.Dispose();
        _mistralTokenizer?.Dispose();
        _encoder.Dispose();
        foreach (SafeTensorsLoader loader in _loaders)
        {
            loader.Dispose();
        }
        _checkpoint?.Dispose();
        // Last: the stack owns the merged weight tensors the transformer was serving.
        _loraStack?.Dispose();
    }
}
