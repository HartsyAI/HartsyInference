using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>FLUX.3 Klein diffusion transformer (ComfyUI <c>comfy/ldm/flux3/model.py</c>). Image, reference-image and text
/// tokens first run their own <see cref="Flux3Block"/> stacks under per-mode modulation, then one joint stack over
/// <c>[image, references, text]</c> in which every span keeps its own modulation; the image rows go through a Flux-style
/// final layer. Text and references sit at t = 0, so their early stacks depend on the conditioning alone and
/// <see cref="PrecomputeConditioning"/> runs them once. A flow-map (step-distilled) build also embeds the time each
/// step jumps to via <c>ref_time_in</c>, which makes those streams per-step. Batch 1; CFG runs separate passes.</summary>
public sealed class Flux3Transformer : IDisposable
{
    private const string ImageMode = "image";
    private const string ReferenceMode = "image_cond";
    private const string TextMode = "txt";

    private readonly Flux3Config _config;
    private readonly Flux3Block[] _imageBlocks;
    private readonly Flux3Block[] _referenceBlocks;
    private readonly Flux3Block[] _textBlocks;
    private readonly Flux3Block[] _jointBlocks;
    private readonly Dictionary<string, AdaLNModulation> _earlyModulations;
    private readonly Dictionary<string, AdaLNModulation> _jointModulations;
    private readonly AdaLNModulation _finalModulation;
    private readonly RopeSlot _imageRope;
    private readonly RopeSlot _referenceRope;
    private readonly RopeSlot _textRope;
    private readonly RopeSlot _jointRope;
    private Tensor? _imageEmbed;
    private Tensor? _referenceEmbed;
    private Tensor? _textEmbed;
    private Tensor? _timeIn1;
    private Tensor? _timeIn2;
    private Tensor? _targetTimeIn1;
    private Tensor? _targetTimeIn2;
    private Tensor? _finalLinear;
    private int _disposed;

    /// <summary>Creates an unloaded transformer for <paramref name="config"/>.</summary>
    public Flux3Transformer(Flux3Config config)
    {
        ArgumentNullException.ThrowIfNull(config);
        _config = config;
        _imageBlocks = CreateBlocks(config.Depth);
        _referenceBlocks = CreateBlocks(config.Depth);
        _textBlocks = CreateBlocks(config.Depth);
        _jointBlocks = CreateBlocks(config.DepthSingleBlocks);
        _earlyModulations = CreateModulations();
        _jointModulations = CreateModulations();
        _finalModulation = new AdaLNModulation(config.HiddenSize, 2);
        _imageRope = new RopeSlot(config);
        _referenceRope = new RopeSlot(config);
        _textRope = new RopeSlot(config);
        _jointRope = new RopeSlot(config);
    }

    /// <summary>The configuration this transformer was built for.</summary>
    public Flux3Config Config => _config;

    /// <summary>Binds the converted weights (see <c>Flux3CheckpointConverter</c>).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        ArgumentNullException.ThrowIfNull(weights);
        _imageEmbed = weights["emb_in.image.weight"];
        _referenceEmbed = weights["emb_in.image_cond.weight"];
        _textEmbed = weights["txt_in.weight"];
        _timeIn1 = weights["time_in.in_layer.weight"];
        _timeIn2 = weights["time_in.out_layer.weight"];
        if (_config.FlowMap)
        {
            _targetTimeIn1 = weights["ref_time_in.in_layer.weight"];
            _targetTimeIn2 = weights["ref_time_in.out_layer.weight"];
        }
        foreach ((string mode, AdaLNModulation modulation) in _earlyModulations)
            modulation.LoadWeights(weights[$"early_stream_modulations.{mode}.lin.weight"], null);
        foreach ((string mode, AdaLNModulation modulation) in _jointModulations)
            modulation.LoadWeights(weights[$"single_stream_modulations.{mode}.lin.weight"], null);
        for (int i = 0; i < _config.Depth; i++)
        {
            _imageBlocks[i].LoadWeights(weights, $"content_mode_blocks.{ImageMode}.{i}");
            _referenceBlocks[i].LoadWeights(weights, $"content_mode_blocks.{ReferenceMode}.{i}");
            _textBlocks[i].LoadWeights(weights, $"txt_mode_blocks.{i}");
        }
        for (int i = 0; i < _config.DepthSingleBlocks; i++)
            _jointBlocks[i].LoadWeights(weights, $"single_blocks.{i}");
        _finalModulation.LoadWeights(weights["final_layer.image.adaLN_modulation.1.weight"], null);
        _finalLinear = weights["final_layer.image.linear.weight"];
    }

    /// <summary>Every weight the transformer reads.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] singles = [_imageEmbed, _referenceEmbed, _textEmbed, _timeIn1, _timeIn2, _targetTimeIn1, _targetTimeIn2, _finalLinear];
        foreach (Tensor? w in singles)
        {
            if (w is not null) yield return w;
        }
        foreach (AdaLNModulation modulation in _earlyModulations.Values)
            foreach (Tensor w in modulation.EnumerateWeights()) yield return w;
        foreach (AdaLNModulation modulation in _jointModulations.Values)
            foreach (Tensor w in modulation.EnumerateWeights()) yield return w;
        foreach (Tensor w in _finalModulation.EnumerateWeights()) yield return w;
        foreach (Flux3Block[] stack in new[] { _imageBlocks, _referenceBlocks, _textBlocks, _jointBlocks })
        {
            foreach (Flux3Block block in stack)
                foreach (Tensor w in block.EnumerateWeights()) yield return w;
        }
    }

    /// <summary>Runs the text and reference streams through their per-mode blocks at t = 0. A flow-map build needs
    /// the step's <paramref name="targetSigma"/>; a guided build passes null and may reuse the result every step.</summary>
    public Flux3ConditionStates PrecomputeConditioning(IBackend backend, Flux3ConditionInput input, float? targetSigma = null)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(input);
        ValidateTarget(targetSigma);
        Tensor zeroVec = TimeVector(backend, 0f, targetSigma);
        try
        {
            Tensor text = EmbedTokens(backend, input.TextFeatures, _textEmbed!, _config.ContextInDim, "text features");
            int textLength = (int)text.Shape[1];
            FluxRope textRope = _textRope.Ensure(textLength, () => Flux3PosEmbed.TextIds(textLength));
            text = RunStack(backend, text, _textBlocks, _earlyModulations[TextMode], zeroVec, textRope);

            Tensor? references = null;
            if (input.ReferenceTokens is not null)
            {
                ValidateReferences(input);
                references = EmbedTokens(backend, input.ReferenceTokens, _referenceEmbed!, _config.InChannels, "reference tokens");
                FluxRope referenceRope = _referenceRope.Ensure(ReferenceSignature(input.ReferenceShapes),
                    () => Flux3PosEmbed.ReferenceIds(input.ReferenceShapes, _config.ImageRopeScale));
                references = RunStack(backend, references, _referenceBlocks, _earlyModulations[ReferenceMode], zeroVec, referenceRope);
            }
            return new Flux3ConditionStates(text, references);
        }
        finally
        {
            zeroVec.Dispose();
        }
    }

    /// <summary>Predicts the flow velocity <c>[1, H·W, 128]</c> for <paramref name="packedLatent"/> <c>[1, H·W, 128]</c>
    /// at <paramref name="sigma"/>. <paramref name="states"/> are the precomputed conditioning streams; a flow-map
    /// build passes null and the step's <paramref name="targetSigma"/>, and the streams are rebuilt for that target.</summary>
    public Tensor Forward(IBackend backend, Tensor packedLatent, int hPacked, int wPacked, float sigma,
        Flux3ConditionInput input, Flux3ConditionStates? states, float? targetSigma = null)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(packedLatent);
        ArgumentNullException.ThrowIfNull(input);
        ValidateTarget(targetSigma);
        int imageTokens = hPacked * wPacked;
        if (packedLatent.Shape.Rank != 3 || packedLatent.Shape[0] != 1 || packedLatent.Shape[1] != imageTokens
            || packedLatent.Shape[2] != _config.InChannels)
        {
            throw new ArgumentException(
                $"FLUX.3 expects a packed latent [1, {imageTokens}, {_config.InChannels}], got {packedLatent.Shape}.", nameof(packedLatent));
        }
        Flux3ConditionStates? owned = states is null ? PrecomputeConditioning(backend, input, targetSigma) : null;
        Flux3ConditionStates conditioning = states ?? owned!;
        Tensor imageVec = TimeVector(backend, sigma, targetSigma);
        Tensor zeroVec = TimeVector(backend, 0f, targetSigma);
        try
        {
            Tensor image = EmbedTokens(backend, packedLatent, _imageEmbed!, _config.InChannels, "packed latent");
            FluxRope imageRope = _imageRope.Ensure(GridSignature(hPacked, wPacked),
                () => Flux3PosEmbed.ImageIds(hPacked, wPacked, 0f, _config.ImageRopeScale));
            image = RunStack(backend, image, _imageBlocks, _earlyModulations[ImageMode], imageVec, imageRope);

            Tensor joint = JoinStreams(backend, image, conditioning, out int[] spanLengths);
            image.Dispose();
            int textLength = (int)conditioning.Text.Shape[1];
            long jointSignature = HashCode.Combine(GridSignature(hPacked, wPacked), ReferenceSignature(input.ReferenceShapes), textLength);
            FluxRope jointRope = _jointRope.Ensure(jointSignature, () => BuildJointIds(hPacked, wPacked, input.ReferenceShapes, textLength));
            joint = RunJoint(backend, joint, imageVec, zeroVec, spanLengths, conditioning.References is not null, jointRope);

            Tensor imageRows = new Tensor(new TensorShape(1, imageTokens, _config.HiddenSize), joint.DType);
            backend.SliceRows(imageRows, joint, 0);
            joint.Dispose();
            Tensor velocity = FinalLayer(backend, imageRows, imageVec);
            imageRows.Dispose();
            return velocity;
        }
        finally
        {
            imageVec.Dispose();
            zeroVec.Dispose();
            owned?.Dispose();
        }
    }

    private Flux3Block[] CreateBlocks(int count)
    {
        Flux3Block[] blocks = new Flux3Block[count];
        for (int i = 0; i < count; i++)
        {
            blocks[i] = new Flux3Block(_config.HiddenSize, _config.NumHeads, _config.MlpHiddenSize,
                _config.LayerNormEps, _config.QkNormEps);
        }
        return blocks;
    }

    private Dictionary<string, AdaLNModulation> CreateModulations() => new(StringComparer.Ordinal)
    {
        [ImageMode] = new AdaLNModulation(_config.HiddenSize, 3),
        [ReferenceMode] = new AdaLNModulation(_config.HiddenSize, 3),
        [TextMode] = new AdaLNModulation(_config.HiddenSize, 3),
    };

    private void ValidateTarget(float? targetSigma)
    {
        if (_config.FlowMap && targetSigma is null)
            throw new ArgumentException("A flow-map FLUX.3 build needs each step's target sigma.", nameof(targetSigma));
    }

    private void ValidateReferences(Flux3ConditionInput input)
    {
        if (input.ReferenceShapes.Count == 0 || input.ReferenceShapes.Count > _config.MaxReferenceImages)
        {
            throw new ArgumentException(
                $"FLUX.3 takes 1 to {_config.MaxReferenceImages} reference images, got {input.ReferenceShapes.Count}.", nameof(input));
        }
        if (input.ReferenceTokens!.Shape[1] != input.ReferenceTokenCount)
        {
            throw new ArgumentException(
                $"Reference tokens {input.ReferenceTokens.Shape} do not match the {input.ReferenceTokenCount} tokens their grids describe.",
                nameof(input));
        }
    }

    /// <summary><c>time_in(t) [+ ref_time_in(target)]</c> as <c>[1, D]</c>; both embeddings scale σ by 1000.</summary>
    private Tensor TimeVector(IBackend backend, float sigma, float? targetSigma)
    {
        Tensor vec = EmbedTime(backend, sigma, _timeIn1!, _timeIn2!);
        if (!_config.FlowMap)
            return vec;
        Tensor target = EmbedTime(backend, targetSigma!.Value, _targetTimeIn1!, _targetTimeIn2!);
        Tensor sum = new Tensor(vec.Shape, DType.F32);
        backend.Add(sum, vec, target);
        vec.Dispose();
        target.Dispose();
        return sum;
    }

    private Tensor EmbedTime(IBackend backend, float sigma, Tensor inLayer, Tensor outLayer)
    {
        TensorShape hiddenShape = new TensorShape(1, _config.HiddenSize);
        Tensor sinusoid = new Tensor(new TensorShape(1, _config.TimestepChannels), DType.F32);
        DiTUtils.SinusoidalTimestepEmbedding(sinusoid, sigma * 1000f, 1, _config.TimestepChannels);
        Tensor hidden = new Tensor(hiddenShape, DType.F32);
        backend.Linear(hidden, sinusoid, inLayer, null);
        sinusoid.Dispose();
        backend.Silu(hidden, hidden);
        Tensor vec = new Tensor(hiddenShape, DType.F32);
        backend.Linear(vec, hidden, outLayer, null);
        hidden.Dispose();
        return vec;
    }

    private Tensor EmbedTokens(IBackend backend, Tensor tokens, Tensor weight, int width, string what)
    {
        if (tokens.Shape.Rank != 3 || tokens.Shape[0] != 1 || tokens.Shape[2] != width)
            throw new ArgumentException($"FLUX.3 {what} must be [1, N, {width}], got {tokens.Shape}.", nameof(tokens));
        Tensor embedded = new Tensor(new TensorShape(1, tokens.Shape[1], _config.HiddenSize), DType.F32);
        backend.Linear(embedded, tokens, weight, null);
        return embedded;
    }

    private static Tensor RunStack(IBackend backend, Tensor x, Flux3Block[] blocks, AdaLNModulation projection, Tensor vec, FluxRope rope)
    {
        Tensor[] rows = projection.Forward(backend, vec);
        using Flux3Modulation modulation = Flux3Modulation.Uniform(rows[0], rows[1], rows[2]);
        foreach (Flux3Block block in blocks)
        {
            Tensor next = block.Forward(backend, x, modulation, rope);
            x.Dispose();
            x = next;
        }
        return x;
    }

    private Tensor RunJoint(IBackend backend, Tensor joint, Tensor imageVec, Tensor zeroVec, int[] spanLengths, bool hasReferences,
        FluxRope rope)
    {
        List<Tensor[]> projected = [_jointModulations[ImageMode].Forward(backend, imageVec)];
        if (hasReferences) projected.Add(_jointModulations[ReferenceMode].Forward(backend, zeroVec));
        projected.Add(_jointModulations[TextMode].Forward(backend, zeroVec));
        try
        {
            List<(Tensor Shift, Tensor Scale, Tensor Gate)> rows = new List<(Tensor, Tensor, Tensor)>(projected.Count);
            foreach (Tensor[] triple in projected) rows.Add((triple[0], triple[1], triple[2]));
            using Flux3Modulation modulation = Flux3Modulation.Spans(backend, rows, spanLengths);
            foreach (Flux3Block block in _jointBlocks)
            {
                Tensor next = block.Forward(backend, joint, modulation, rope);
                joint.Dispose();
                joint = next;
            }
            return joint;
        }
        finally
        {
            foreach (Tensor[] triple in projected)
                foreach (Tensor t in triple) t.Dispose();
        }
    }

    private static Tensor JoinStreams(IBackend backend, Tensor image, Flux3ConditionStates conditioning, out int[] spanLengths)
    {
        List<Tensor> parts = [image];
        if (conditioning.References is not null) parts.Add(conditioning.References);
        parts.Add(conditioning.Text);
        spanLengths = new int[parts.Count];
        long total = 0;
        for (int i = 0; i < parts.Count; i++)
        {
            spanLengths[i] = (int)parts[i].Shape[1];
            total += spanLengths[i];
        }
        Tensor joint = new Tensor(new TensorShape(1, total, image.Shape[2]), image.DType);
        backend.Concat(joint, parts.ToArray(), 1);
        return joint;
    }

    private Tensor BuildJointIds(int hPacked, int wPacked, IReadOnlyList<(int Height, int Width)> referenceShapes, int textLength)
    {
        using Tensor image = Flux3PosEmbed.ImageIds(hPacked, wPacked, 0f, _config.ImageRopeScale);
        using Tensor text = Flux3PosEmbed.TextIds(textLength);
        if (referenceShapes.Count == 0)
            return Flux3PosEmbed.Concat(image, text);
        using Tensor references = Flux3PosEmbed.ReferenceIds(referenceShapes, _config.ImageRopeScale);
        return Flux3PosEmbed.Concat(image, references, text);
    }

    /// <summary>Flux-style final layer: <c>linear((1 + scale)·LayerNorm(x) + shift)</c>, <c>[shift, scale]</c> from the image
    /// time vector.</summary>
    private Tensor FinalLayer(IBackend backend, Tensor imageRows, Tensor imageVec)
    {
        Tensor[] shiftScale = _finalModulation.Forward(backend, imageVec);
        Tensor modulated = new Tensor(imageRows.Shape, imageRows.DType);
        backend.LayerNormModulate(modulated, imageRows, shiftScale[1], shiftScale[0], _config.LayerNormEps);
        shiftScale[0].Dispose();
        shiftScale[1].Dispose();
        Tensor velocity = new Tensor(new TensorShape(1, imageRows.Shape[1], _config.InChannels), DType.F32);
        backend.Linear(velocity, modulated, _finalLinear!, null);
        modulated.Dispose();
        return velocity;
    }

    private static long GridSignature(int hPacked, int wPacked) => ((long)hPacked << 32) | (uint)wPacked;

    private static long ReferenceSignature(IReadOnlyList<(int Height, int Width)> shapes)
    {
        HashCode hash = new HashCode();
        hash.Add(shapes.Count);
        foreach ((int h, int w) in shapes)
        {
            hash.Add(h);
            hash.Add(w);
        }
        return hash.ToHashCode();
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        _imageEmbed = null;
        _referenceEmbed = null;
        _textEmbed = null;
        _timeIn1 = null;
        _timeIn2 = null;
        _targetTimeIn1 = null;
        _targetTimeIn2 = null;
        _finalLinear = null;
    }

    /// <summary>Two RoPE table sets keyed by position layout, so the conditional and unconditional passes of true CFG
    /// (whose prompts may pad to different lengths) do not rebuild tables every step. A miss re-precomputes the least
    /// recently used set in place, which also releases its previous device tables.</summary>
    private sealed class RopeSlot(Flux3Config config)
    {
        private readonly FluxRope[] _ropes = [new FluxRope(config.AxesDim, config.Theta), new FluxRope(config.AxesDim, config.Theta)];
        private readonly long[] _signatures = [long.MinValue, long.MinValue];
        private int _lastUsed;

        public FluxRope Ensure(long signature, Func<Tensor> buildIds)
        {
            for (int i = 0; i < _ropes.Length; i++)
            {
                if (_signatures[i] == signature)
                {
                    _lastUsed = i;
                    return _ropes[i];
                }
            }
            int victim = 1 - _lastUsed;
            using Tensor ids = buildIds();
            _ropes[victim].Precompute(ids);
            _signatures[victim] = signature;
            _lastUsed = victim;
            return _ropes[victim];
        }
    }
}
