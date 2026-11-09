using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Multimodal;
using HartsyInference.LLM.Sampling;
using HartsyInference.LLM.Transformer;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.LLM.Models.MossVl;

/// <summary>One image of a MOSS-VL request: its merged vision tokens from the vision tower and its merged grid.</summary>
/// <param name="MergedTokens">The image's merged vision tokens <c>[h·w, hidden]</c> F32.</param>
/// <param name="MergedHeight">The merged grid height (patch height divided by the merge size).</param>
/// <param name="MergedWidth">The merged grid width.</param>
public readonly record struct MossVlImageFeatures(Tensor MergedTokens, int MergedHeight, int MergedWidth);

/// <summary>One MOSS-VL request: a prefill over the prompt and its images, then single-token decode steps. Every step reads the
/// same vision features through the gated cross-attention layers, with the decode token's position continuing the prompt's.</summary>
/// <remarks>Holds the KV cache, the vision tensor and its projected cross-attention keys and values; dispose it when the request ends.
/// If a forward throws part-way, the session is faulted and refuses further calls, because its KV cache no longer matches the tokens.</remarks>
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
    private readonly CrossKvCache _crossKv;
    private int _cacheLength;
    private int _nextPosition;
    private bool _prefilled;
    private bool _faulted;
    private int _disposed;

    /// <param name="images">The request's images, in the order their placeholders appear in <paramref name="promptIds"/>.</param>
    /// <param name="maxNewTokens">The most tokens the session will decode; sizes the KV cache.</param>
    /// <exception cref="ArgumentException">No image, or a merged tensor that does not match its grid.</exception>
    public MossVlDecodeSession(MossVlTextDecoder decoder, MossVlConfig config, IBackend backend, int[] promptIds,
        IReadOnlyList<MossVlImageFeatures> images, int maxNewTokens)
    {
        if (images.Count == 0) throw new ArgumentException("A MOSS-VL session needs at least one image.", nameof(images));
        MossVlImage[] grids = new MossVlImage[images.Count];
        Tensor[] merged = new Tensor[images.Count];
        for (int k = 0; k < images.Count; k++)
        {
            // Each image's own token count must match its grid: a total that only matches in sum would misplace the separators.
            long expected = (long)images[k].MergedHeight * images[k].MergedWidth;
            if (images[k].MergedTokens.Shape[0] != expected)
                throw new ArgumentException($"Image {k} has {images[k].MergedTokens.Shape[0]} merged tokens for a {images[k].MergedHeight}x{images[k].MergedWidth} grid ({expected}).", nameof(images));
            grids[k] = new MossVlImage(images[k].MergedHeight, images[k].MergedWidth);
            merged[k] = images[k].MergedTokens;
        }

        _decoder = decoder;
        _config = config;
        _backend = backend;
        _promptIds = promptIds;
        _promptPlan = MossVlPositions.Plan(promptIds, config.ImageTokenId, grids);
        _vision = decoder.AppendSeparators(merged);
        // The images' vision rotary tables are the same for every step, so they are built once here.
        Tensor? visionCos = null, visionSin = null;
        try
        {
            if (_vision.Shape[1] != _promptPlan.VisionLength)
                throw new ArgumentException($"Vision tokens {_vision.Shape[1]} do not match the image grids ({_promptPlan.VisionLength} with separators).", nameof(images));
            (visionCos, visionSin) = MossVlPositions.BuildVisionTables(_promptPlan, config);
            TransformerConfig text = decoder.TextConfig;
            _cache = new FixedKvCache(text.NumLayers, 1, text.NumKvHeads, text.HeadDim, promptIds.Length + maxNewTokens + 1);
            _crossKv = new CrossKvCache(text.NumLayers);
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
    public float[] Prefill() =>
        RunPrefill(hidden => VlmDecodeOps.LastLogits(_backend, _decoder.Model, hidden, _promptIds.Length));

    /// <summary>Runs the prompt and selects the first generated token with <paramref name="sampler"/>; the logits are read on the device.</summary>
    public int Prefill(SamplerChain sampler, List<int> history) =>
        RunPrefill(hidden => VlmDecodeOps.SampleLast(_backend, _decoder.Model, hidden, _promptIds.Length, sampler, history));

    /// <summary>Decodes one token after the last one and returns the logits after it.</summary>
    public float[] Step(int token) =>
        RunStep(token, hidden => VlmDecodeOps.LastLogits(_backend, _decoder.Model, hidden, 1));

    /// <summary>Decodes one token after the last one and selects the next with <paramref name="sampler"/>; the logits are read on the device.</summary>
    public int Step(int token, SamplerChain sampler, List<int> history) =>
        RunStep(token, hidden => VlmDecodeOps.SampleLast(_backend, _decoder.Model, hidden, 1, sampler, history));

    /// <inheritdoc/>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _crossKv.Dispose();
        _cache.Dispose();
        _visionCos.Dispose();
        _visionSin.Dispose();
        _vision.Dispose();
    }

    /// <summary>Prefill forward, then <paramref name="read"/> over its hidden state. The rope tables and embeddings stay alive until the
    /// read has returned, so an asynchronous backend cannot free them under a queued kernel.</summary>
    private T RunPrefill<T>(Func<Tensor, T> read)
    {
        ThrowIfUnusable();
        if (_prefilled) throw new InvalidOperationException("The prompt is already prefilled.");
        int t = _promptIds.Length;
        try
        {
            // The images' vision tables are the session's own, borrowed here (see BuildRopeOverVisionTables).
            using MossVlRope rope = MossVlPositions.BuildRopeOverVisionTables(_promptPlan, _config, _visionCos, _visionSin);
            using Tensor embeds = new(new TensorShape(1, t, _decoder.TextConfig.HiddenSize), DType.F32);
            _decoder.Model.EmbedLookup(embeds, _promptIds);
            using Tensor hidden = _decoder.Model.ForwardEmbeds(_backend, embeds, t, 0, _cache, applyFinalNorm: true,
                crossStates: _vision, crossLen: _promptPlan.VisionLength, rope: rope.Rope, crossKv: _crossKv);
            _prefilled = true;
            _cacheLength = t;
            _nextPosition = MaxPosition(_promptPlan.Text) + 1;
            return read(hidden);
        }
        catch
        {
            _faulted = true;   // the cache may hold rows the session never counted
            throw;
        }
    }

    /// <summary>One decode forward for <paramref name="token"/>, then <paramref name="read"/> over its hidden state (see <see cref="RunPrefill{T}"/>).</summary>
    private T RunStep<T>(int token, Func<Tensor, T> read)
    {
        ThrowIfUnusable();
        if (!_prefilled) throw new InvalidOperationException("Prefill must run before a decode step.");
        int position = _nextPosition;
        // A decode token is text: all three axes at the next position. It follows every image, so its row sees all of them and
        // needs neither a row mask nor a cross bias.
        MossVlPositionPlan stepPlan = new()
        {
            Text = [position, position, position],
            Vision = _promptPlan.Vision,
            VisionLength = _promptPlan.VisionLength,
            RowMask = null,
            CrossGroups = null,
        };
        try
        {
            using MossVlRope rope = MossVlPositions.BuildRopeOverVisionTables(stepPlan, _config, _visionCos, _visionSin);
            using Tensor embeds = new(new TensorShape(1, 1, _decoder.TextConfig.HiddenSize), DType.F32);
            _decoder.Model.EmbedLookup(embeds, [token]);
            using Tensor hidden = _decoder.Model.ForwardEmbeds(_backend, embeds, 1, _cacheLength, _cache, applyFinalNorm: true,
                crossStates: _vision, crossLen: _promptPlan.VisionLength, rope: rope.Rope, crossKv: _crossKv);
            _nextPosition++;
            _cacheLength++;
            return read(hidden);
        }
        catch
        {
            _faulted = true;
            throw;
        }
    }

    private void ThrowIfUnusable()
    {
        if (Volatile.Read(ref _disposed) != 0) throw new ObjectDisposedException(nameof(MossVlDecodeSession));
        if (_faulted) throw new InvalidOperationException("An earlier forward failed part-way, so the KV cache no longer matches the decoded tokens; start a new session.");
    }

    private static int MaxPosition(int[] axisMajor)
    {
        int max = 0;
        foreach (int p in axisMajor) max = Math.Max(max, p);
        return max;
    }
}

/// <summary>Greedy or sampled MOSS-VL generation for one or more images and one question, in the Qwen-VL ChatML layout the checkpoint's
/// chat template uses.</summary>
public sealed class MossVlGenerator(MossVlTextDecoder decoder, MossVlConfig config, ILlmTokenizer tokenizer, IBackend backend)
{
    private const string ImagePlaceholder = "<|vision_start|><|image_pad|><|vision_end|>";

    /// <summary>Generates a reply to <paramref name="question"/> about one image. Sampling defaults match the other VLM generators
    /// (<see cref="MllamaGenerator"/>).</summary>
    public string Generate(string question, Tensor mergedVision, int mergedHeight, int mergedWidth, int maxTokens = 64,
        SamplingOptions? sampling = null) =>
        Generate(question, [new MossVlImageFeatures(mergedVision, mergedHeight, mergedWidth)], maxTokens, sampling);

    /// <summary>Generates a reply to <paramref name="question"/> about <paramref name="images"/>, shown in that order before the question.
    /// At least one image is required: a text-only prompt has no placeholder for the vision sequence.</summary>
    /// <exception cref="ArgumentException">No image is given.</exception>
    public string Generate(string question, IReadOnlyList<MossVlImageFeatures> images, int maxTokens = 64, SamplingOptions? sampling = null)
    {
        if (images.Count == 0) throw new ArgumentException("MOSS-VL generation needs at least one image.", nameof(images));
        if (maxTokens <= 0) return string.Empty;
        string placeholders = string.Concat(Enumerable.Repeat(ImagePlaceholder, images.Count));
        int[] ids = tokenizer.Encode("<|im_start|>user\n" + placeholders + question + "<|im_end|>\n<|im_start|>assistant\n", addSpecial: true);
        SamplerChain sampler = SamplerChain.FromOptions(sampling ?? new SamplingOptions
        {
            Temperature = 0.4f, TopP = 0.9f, RepetitionPenalty = 1.1f, Seed = 1,
        }, tokenizer, config.Text.VocabSize);
        HashSet<int> stops = new(tokenizer.StopIds);
        List<int> history = [];
        List<int> generated = [];

        using MossVlDecodeSession session = new(decoder, config, backend, ids, images, maxTokens);
        int token = session.Prefill(sampler, history);
        while (!stops.Contains(token) && generated.Count < maxTokens)
        {
            generated.Add(token);
            history.Add(token);
            if (generated.Count == maxTokens) break;
            token = session.Step(token, sampler, history);
        }
        return tokenizer.Decode([.. generated]);
    }
}
