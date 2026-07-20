using System.Globalization;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.Tokenizers;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>A constructed Kandinsky 5 pipeline driven against the native <see cref="ImageRequest"/>. <see cref="Kandinsky5Pipeline"/> takes only pre-computed embeddings, so this owns the dual text stack: it wraps the prompt in Kandinsky's fixed "promt engineer" ChatML template, runs Qwen2.5-VL-7B for the last hidden state and drops the template prefix, frees those weights, then takes the CLIP-L pooled embedding — the two inputs the reference <c>encode_prompt</c> produces. Ported from the diffusers reference, not from a SwarmUI loader (none exists); UNVERIFIED against real weights.</summary>
public sealed unsafe class Kandinsky5RecipePipeline : IRecipePipeline
{
    /// <summary>Kandinsky's fixed conditioning system prompt — verbatim from diffusers <c>pipeline_kandinsky_t2i.py</c> (the "promt" typo is in the original).</summary>
    private const string SystemPrompt = "You are a promt engineer. Describe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:";

    /// <summary>Maximum prompt tokens after the template prefix (diffusers <c>max_sequence_length</c> default).</summary>
    private const int MaxPromptTokens = 512;

    private readonly Kandinsky5Pipeline _pipeline;
    private readonly LlamaStyleEncoder _qwen;
    private readonly ClipTextEncoder _clipL;
    private readonly Qwen2Tokenizer _qwenTokenizer;
    private readonly ClipTokenizer _clipTokenizer;
    private readonly IBackend _backend;
    private readonly Kandinsky5Transformer _transformer;
    private readonly List<SafeTensorsLoader> _loaders;

    /// <summary>Wraps the constructed Kandinsky 5 pipeline plus its dual text stack, taking ownership of every disposable.</summary>
    public Kandinsky5RecipePipeline(Kandinsky5Pipeline pipeline, LlamaStyleEncoder qwen, ClipTextEncoder clipL, Qwen2Tokenizer qwenTokenizer, ClipTokenizer clipTokenizer, IBackend backend, Kandinsky5Transformer transformer, List<SafeTensorsLoader> loaders)
    {
        _pipeline = pipeline;
        _qwen = qwen;
        _clipL = clipL;
        _qwenTokenizer = qwenTokenizer;
        _clipTokenizer = clipTokenizer;
        _backend = backend;
        _transformer = transformer;
        _loaders = loaders;
    }

    /// <inheritdoc/>
    public ImageResult Generate(ImageRequest request, IProgress<StepPreview>? progress, CancellationToken cancel)
    {
        cancel.ThrowIfCancellationRequested();
        string prompt = request.Prompt;
        string negative = request.NegativePrompt ?? "";
        int steps = request.Steps <= 0 ? 50 : request.Steps;
        float cfg = request.CfgScale <= 0 ? 1.0f : request.CfgScale;
        bool useCfg = cfg > 1.0f;

        // TODO(E-IMG-4/5): img2img/inpaint, LoRA, ControlNet, IP-Adapter, refiner, regional prompting and
        // ImageRequest.Components overrides are deferred — text-to-image only.
        Tensor? qwenEmbeds = null;
        Tensor? negQwenEmbeds = null;
        Tensor? clipPooled = null;
        Tensor? negClipPooled = null;
        try
        {
            // Qwen2.5-VL is ~16 GB: bulk-upload, encode both branches, then free before the denoise loop preloads
            // the transformer (the ZImage/OmniGen2 preload -> encode -> free pattern).
            _backend.PreloadWeights(_qwen.EnumerateWeights());
            qwenEmbeds = EncodeQwen(prompt);
            if (useCfg)
            {
                negQwenEmbeds = EncodeQwen(negative);
            }
            _backend.FreeWeights(_qwen.EnumerateWeights());

            clipPooled = EncodeClipPooled(prompt);
            if (useCfg)
            {
                negClipPooled = EncodeClipPooled(negative);
            }

            TextToImageRequest inner = new TextToImageRequest
            {
                Prompt = prompt,
                NegativePrompt = negative,
                Width = request.Width,
                Height = request.Height,
                Steps = steps,
                CfgScale = cfg,
                Seed = request.Seed < 0 ? null : (int?)(int)(request.Seed & 0x7FFFFFFF),
            };

            Action<GenerationProgress> bridge = p =>
            {
                cancel.ThrowIfCancellationRequested();
                progress?.Report(new StepPreview { Step = p.Step, TotalSteps = p.TotalSteps });
            };

            (byte[] rgb, int outW, int outH, int usedSeed) = _pipeline.GenerateFromEmbeddings(
                qwenEmbeds, clipPooled, negQwenEmbeds, negClipPooled, inner, bridge);

            return new ImageResult
            {
                Rgb = rgb,
                Width = outW,
                Height = outH,
                Seed = usedSeed,
                Meta = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
                {
                    ["arch"] = "kandinsky5",
                    ["size"] = $"{outW}x{outH}",
                    ["seed"] = usedSeed.ToString(CultureInfo.InvariantCulture),
                    ["steps"] = steps.ToString(CultureInfo.InvariantCulture),
                    ["cfg"] = cfg.ToString(CultureInfo.InvariantCulture),
                },
            };
        }
        finally
        {
            qwenEmbeds?.Dispose();
            negQwenEmbeds?.Dispose();
            clipPooled?.Dispose();
            negClipPooled?.Dispose();
        }
    }

    /// <summary>Runs the Qwen2.5-VL tower over the templated prompt and returns the last hidden state with the fixed template prefix sliced off — the reference takes <c>hidden_states[-1][:, prompt_template_encode_start_idx:]</c> (start idx 41 for the shipped template; computed here from the actual prefix ids so a tokenizer revision can't silently misalign it).</summary>
    private Tensor EncodeQwen(string prompt)
    {
        int prefixLength = _qwenTokenizer.Encode($"<|im_start|>system\n{SystemPrompt}<|im_end|>\n<|im_start|>user\n", addSpecial: true).Length;
        int[] full = _qwenTokenizer.Encode($"<|im_start|>system\n{SystemPrompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>", addSpecial: true);
        if (full.Length > prefixLength + MaxPromptTokens)
        {
            full = full[..(prefixLength + MaxPromptTokens)];
        }
        Tensor encoded = _qwen.Encode(_backend, new[] { full });
        try
        {
            return SliceSequenceFrom(encoded, prefixLength);
        }
        finally
        {
            encoded.Dispose();
        }
    }

    /// <summary>Runs CLIP-L over the 77-token prompt and returns the pooled EOS embedding <c>[1, 768]</c> (CLIPTextModel's <c>pooler_output</c> — CLIP-L has no text_projection, so the pooled value is the raw post-final-LN EOS hidden state).</summary>
    private Tensor EncodeClipPooled(string prompt)
    {
        int[] tokens = _clipTokenizer.Encode(prompt);
        int eosPos = ClipTokenizer.FindEosPosition(tokens);
        (Tensor hidden, Tensor? pooled) = _clipL.EncodePenultimate(_backend, new[] { tokens }, new[] { eosPos }, layersFromEnd: 1);
        hidden.Dispose();
        if (pooled is null)
        {
            throw new InvalidOperationException("CLIP-L produced no pooled output; Kandinsky 5 requires the [1, 768] pooled embedding.");
        }
        return pooled;
    }

    /// <summary>Copies a <c>[batch, seq, hidden]</c> F32 tensor from <paramref name="start"/> to the end of the sequence axis, dropping the leading template tokens.</summary>
    private static Tensor SliceSequenceFrom(Tensor source, int start)
    {
        if (source.Shape.Rank != 3)
        {
            throw new ArgumentException($"Expected a rank-3 tensor, got rank {source.Shape.Rank}.", nameof(source));
        }
        if (source.DType != DType.F32)
        {
            throw new ArgumentException($"SliceSequenceFrom expects F32, got {source.DType}.", nameof(source));
        }
        long batch = source.Shape[0];
        long fullLen = source.Shape[1];
        long hidden = source.Shape[2];
        if (start < 0 || start >= fullLen)
        {
            throw new ArgumentOutOfRangeException(nameof(start), $"start {start} out of range [0..{fullLen - 1}].");
        }
        long keep = fullLen - start;
        Tensor result = new Tensor(new TensorShape(batch, keep, hidden), source.DType);
        long elemSize = source.DType.SizeInBytes;
        long fullRowBytes = fullLen * hidden * elemSize;
        long keepRowBytes = keep * hidden * elemSize;
        long offsetBytes = start * hidden * elemSize;
        byte* src = (byte*)source.DataPointer;
        byte* dst = (byte*)result.DataPointer;
        for (long b = 0; b < batch; b++)
        {
            Buffer.MemoryCopy(src + b * fullRowBytes + offsetBytes, dst + b * keepRowBytes, keepRowBytes, keepRowBytes);
        }
        return result;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _pipeline.Dispose();
        _qwen.Dispose();
        _qwenTokenizer.Dispose();
        _clipTokenizer.Dispose();
        _transformer.Dispose();
        foreach (SafeTensorsLoader loader in _loaders)
        {
            loader.Dispose();
        }
    }
}
