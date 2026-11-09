using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Multimodal;
using HartsyInference.LLM.Sampling;
using HartsyInference.LLM.Transformer;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.LLM.Models.MossVl;

/// <summary>One MOSS-VL request: a prefill over the prompt and one image, then single-token decode steps. Every step reads the
/// same vision features through the gated cross-attention layers, with the decode token's position continuing the prompt's.</summary>
/// <remarks>Holds the KV cache and the vision tensor; dispose it when the request ends.</remarks>
public sealed class MossVlDecodeSession : IDisposable
{
    private readonly MossVlTextDecoder _decoder;
    private readonly MossVlConfig _config;
    private readonly IBackend _backend;
    private readonly int[] _promptIds;
    private readonly MossVlPositionPlan _promptPlan;
    private readonly Tensor _vision;
    private readonly Tensor _visionCos;
    private readonly Tensor _visionSin;
    private readonly FixedKvCache _cache;
    private int _cacheLength;
    private int _nextPosition;
    private bool _prefilled;
    private int _disposed;

    /// <param name="mergedVision">The image's merged vision tokens <c>[h·w, hidden]</c> F32, from the vision tower.</param>
    /// <param name="mergedHeight">The image's merged grid height (patch height divided by the merge size).</param>
    /// <param name="mergedWidth">The image's merged grid width.</param>
    /// <param name="maxNewTokens">The most tokens the session will decode; sizes the KV cache.</param>
    public MossVlDecodeSession(MossVlTextDecoder decoder, MossVlConfig config, IBackend backend, int[] promptIds,
        Tensor mergedVision, int mergedHeight, int mergedWidth, int maxNewTokens)
    {
        _decoder = decoder;
        _config = config;
        _backend = backend;
        _promptIds = promptIds;
        _promptPlan = MossVlPositions.Plan(promptIds, config.ImageTokenId, [new MossVlImage(mergedHeight, mergedWidth)]);
        _vision = decoder.AppendSeparator(mergedVision);
        // The image's vision rotary tables are the same for every step, so they are built once here.
        Tensor? visionCos = null, visionSin = null;
        try
        {
            if (_vision.Shape[1] != _promptPlan.VisionLength)
                throw new ArgumentException($"Vision tokens {_vision.Shape[1]} do not match the image grid ({_promptPlan.VisionLength} with separator).", nameof(mergedVision));
            (visionCos, visionSin) = MossVlPositions.BuildVisionTables(_promptPlan, config);
            TransformerConfig text = decoder.TextConfig;
            _cache = new FixedKvCache(text.NumLayers, 1, text.NumKvHeads, text.HeadDim, promptIds.Length + maxNewTokens + 1);
            _visionCos = visionCos;
            _visionSin = visionSin;
        }
        catch
        {
            visionCos?.Dispose();
            visionSin?.Dispose();
            _vision.Dispose();
            throw;
        }
    }

    /// <summary>Runs the prompt through the model and returns the logits after its last token.</summary>
    public float[] Prefill()
    {
        ThrowIfDisposed();
        if (_prefilled) throw new InvalidOperationException("The prompt is already prefilled.");
        int t = _promptIds.Length;
        using MossVlRope rope = MossVlPositions.BuildRope(_promptPlan, _config);
        using Tensor embeds = new(new TensorShape(1, t, _decoder.TextConfig.HiddenSize), DType.F32);
        _decoder.Model.EmbedLookup(embeds, _promptIds);
        using Tensor hidden = _decoder.Model.ForwardEmbeds(_backend, embeds, t, 0, _cache, applyFinalNorm: true,
            crossStates: _vision, crossLen: _promptPlan.VisionLength, rope: rope.Rope);
        _prefilled = true;
        _cacheLength = t;
        _nextPosition = MaxPosition(_promptPlan.Text) + 1;
        return VlmDecodeOps.LastLogits(_backend, _decoder.Model, hidden, t);
    }

    /// <summary>Decodes one token after the last one and returns the logits after it.</summary>
    public float[] Step(int token)
    {
        ThrowIfDisposed();
        if (!_prefilled) throw new InvalidOperationException("Prefill must run before a decode step.");
        int position = _nextPosition++;
        // A decode token is text: all three axes at the next position, reading the prompt's vision positions with its row visible.
        MossVlPositionPlan stepPlan = new()
        {
            Text = [position, position, position],
            Vision = _promptPlan.Vision,
            VisionLength = _promptPlan.VisionLength,
            RowMask = [1f],
        };
        using MossVlRope rope = MossVlPositions.BuildDecodeRope(stepPlan, _config, _visionCos, _visionSin);
        using Tensor embeds = new(new TensorShape(1, 1, _decoder.TextConfig.HiddenSize), DType.F32);
        _decoder.Model.EmbedLookup(embeds, [token]);
        using Tensor hidden = _decoder.Model.ForwardEmbeds(_backend, embeds, 1, _cacheLength, _cache, applyFinalNorm: true,
            crossStates: _vision, crossLen: _promptPlan.VisionLength, rope: rope.Rope);
        _cacheLength++;
        return VlmDecodeOps.LastLogits(_backend, _decoder.Model, hidden, 1);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _cache.Dispose();
        _visionCos.Dispose();
        _visionSin.Dispose();
        _vision.Dispose();
    }

    private void ThrowIfDisposed()
    {
        if (Volatile.Read(ref _disposed) != 0) throw new ObjectDisposedException(nameof(MossVlDecodeSession));
    }

    private static int MaxPosition(int[] axisMajor)
    {
        int max = 0;
        foreach (int p in axisMajor) max = Math.Max(max, p);
        return max;
    }
}

/// <summary>Greedy or sampled MOSS-VL generation for one image and one question, in the Qwen-VL ChatML layout the checkpoint's chat
/// template uses.</summary>
public sealed class MossVlGenerator(MossVlTextDecoder decoder, MossVlConfig config, ILlmTokenizer tokenizer, IBackend backend)
{
    /// <summary>Generates a reply to <paramref name="question"/> about the image. Sampling defaults match the other VLM generators
    /// (<see cref="MllamaGenerator"/>).</summary>
    public string Generate(string question, Tensor mergedVision, int mergedHeight, int mergedWidth, int maxTokens = 64,
        SamplingOptions? sampling = null)
    {
        if (maxTokens <= 0) return string.Empty;
        int[] ids = tokenizer.Encode(
            "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>" + question + "<|im_end|>\n<|im_start|>assistant\n",
            addSpecial: true);
        SamplerChain sampler = SamplerChain.FromOptions(sampling ?? new SamplingOptions
        {
            Temperature = 0.4f, TopP = 0.9f, RepetitionPenalty = 1.1f, Seed = 1,
        }, tokenizer, config.Text.VocabSize);
        HashSet<int> stops = new(tokenizer.StopIds);
        List<int> history = [];
        List<int> generated = [];

        using MossVlDecodeSession session = new(decoder, config, backend, ids, mergedVision, mergedHeight, mergedWidth, maxTokens);
        int token = sampler.Next(session.Prefill(), history);
        while (!stops.Contains(token) && generated.Count < maxTokens)
        {
            generated.Add(token);
            history.Add(token);
            if (generated.Count == maxTokens) break;
            token = sampler.Next(session.Step(token), history);
        }
        return tokenizer.Decode([.. generated]);
    }
}
