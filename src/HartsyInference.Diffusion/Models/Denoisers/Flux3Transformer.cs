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
        bool f16 = UseF16Stream(backend);
        Tensor zeroVec = TimeVector(backend, 0f, targetSigma);
        Tensor? text = null;
        Tensor? references = null;
        try
        {
            text = EmbedTokens(backend, input.TextFeatures, _textEmbed!, _config.ContextInDim, "text features");
            text = ToStreamDtype(backend, text, f16);
            int textLength = (int)text.Shape[1];
            FluxRope textRope = _textRope.Ensure(textLength, () => Flux3PosEmbed.TextIds(textLength));
            text = RunStack(backend, text, _textBlocks, _earlyModulations[TextMode], zeroVec, textRope, f16);

            if (input.ReferenceTokens is not null)
            {
                ValidateReferences(input);
                references = EmbedTokens(backend, input.ReferenceTokens, _referenceEmbed!, _config.InChannels, "reference tokens");
                references = ToStreamDtype(backend, references, f16);
                FluxRope referenceRope = _referenceRope.Ensure(ReferenceSignature(input.ReferenceShapes),
                    () => Flux3PosEmbed.ReferenceIds(input.ReferenceShapes, _config.ImageRopeScale));
                references = RunStack(backend, references, _referenceBlocks, _earlyModulations[ReferenceMode], zeroVec, referenceRope, f16);
            }
            return new Flux3ConditionStates(text, references);
        }
        catch
        {
            // Ownership passes to the states only once they exist; a stream built before the failure dies here.
            // RunStack has already freed its own input when it throws, and Dispose is idempotent.
            text?.Dispose();
            references?.Dispose();
            throw;
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
        // Each intermediate is freed as soon as the next exists; the finally frees whichever were live when a stage threw.
        // A stack that throws has already freed its own input, and Dispose is idempotent.
        Flux3ConditionStates? owned = null;
        Tensor? imageVec = null;
        Tensor? zeroVec = null;
        Tensor? image = null;
        Tensor? joint = null;
        Tensor? imageRows = null;
        try
        {
            bool f16 = UseF16Stream(backend);
            // JoinStreams copies bytes without converting dtypes, so cached states must match this call's stream dtype. The
            // knob can change between a precompute and a later forward; a mismatch is rebuilt here and the caller's states are left alone.
            Flux3ConditionStates conditioning;
            if (states is not null && StatesMatch(states, f16))
            {
                conditioning = states;
            }
            else
            {
                owned = PrecomputeConditioning(backend, input, targetSigma);
                conditioning = owned;
            }
            imageVec = TimeVector(backend, sigma, targetSigma);
            zeroVec = TimeVector(backend, 0f, targetSigma);
            image = EmbedTokens(backend, packedLatent, _imageEmbed!, _config.InChannels, "packed latent");
            image = ToStreamDtype(backend, image, f16);
            FluxRope imageRope = _imageRope.Ensure(GridSignature(hPacked, wPacked),
                () => Flux3PosEmbed.ImageIds(hPacked, wPacked, 0f, _config.ImageRopeScale));
            image = RunStack(backend, image, _imageBlocks, _earlyModulations[ImageMode], imageVec, imageRope, f16);

            joint = JoinStreams(backend, image, conditioning, out int[] spanLengths);
            image.Dispose();
            int textLength = (int)conditioning.Text.Shape[1];
            long jointSignature = HashCode.Combine(GridSignature(hPacked, wPacked), ReferenceSignature(input.ReferenceShapes), textLength);
            FluxRope jointRope = _jointRope.Ensure(jointSignature, () => BuildJointIds(hPacked, wPacked, input.ReferenceShapes, textLength));
            joint = RunJoint(backend, joint, imageVec, zeroVec, spanLengths, conditioning.References is not null, jointRope, f16);

            imageRows = new Tensor(new TensorShape(1, imageTokens, _config.HiddenSize), joint.DType);
            backend.SliceRows(imageRows, joint, 0);
            joint.Dispose();
            if (imageRows.DType == DType.F16)
            {
                // Back to F32 for the final layer: its LayerNorm cancels the residual damp, and the velocity keeps F32 precision
                // across Euler steps (the Flux2 recipe).
                Tensor rowsF32 = new Tensor(imageRows.Shape, DType.F32);
                backend.CastToF32(rowsF32, imageRows);
                imageRows.Dispose();
                imageRows = rowsF32;
            }
            return FinalLayer(backend, imageRows, imageVec);
        }
        finally
        {
            imageRows?.Dispose();
            joint?.Dispose();
            image?.Dispose();
            imageVec?.Dispose();
            zeroVec?.Dispose();
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

    /// <summary>The F16 stream runs where the backend has F16 activation kernels and <c>numerics.ditF16</c> is on. The CPU
    /// backend keeps F32, so a default CPU build never reaches an F16 kernel.</summary>
    private static bool UseF16Stream(IBackend backend) => backend.SupportsF16Activations && DitDtype.Act == DType.F16;

    /// <summary>True when every cached stream already has the dtype this call's stream would produce.</summary>
    private static bool StatesMatch(Flux3ConditionStates states, bool f16)
    {
        DType expected = f16 ? DType.F16 : DType.F32;
        return states.Text.DType == expected && (states.References is null || states.References.DType == expected);
    }

    /// <summary>Scales an embedded F32 stream by the residual damp and casts it to F16. On success this takes ownership of
    /// <paramref name="f32"/> and returns the F16 stream; on the F32 path the stream is returned unchanged. On failure the
    /// caller still owns <paramref name="f32"/>, and every temporary this call created is freed.</summary>
    private static Tensor ToStreamDtype(IBackend backend, Tensor f32, bool f16)
    {
        if (!f16)
            return f32;
        TensorShape shape = f32.Shape;
        Tensor damped = new Tensor(shape, DType.F32);
        Tensor stream = new Tensor(shape, DType.F16);
        try
        {
            backend.Scale(damped, f32, ChromaF16.ResidualDamp);
            backend.CastToF16(stream, damped);
        }
        catch
        {
            damped.Dispose();
            stream.Dispose();
            throw;
        }
        damped.Dispose();
        f32.Dispose();
        return stream;
    }

    /// <summary>Scales a block's gate row by the residual damp. The gate multiplies the whole branch (attention, MLP and any
    /// runtime LoRA adjunct), so the damped stream sees the branch at the same scale on every backend and weight format.
    /// On success this takes ownership of <paramref name="gate"/>; on failure the caller still owns it.</summary>
    private static Tensor DampGate(IBackend backend, Tensor gate)
    {
        Tensor damped = new Tensor(gate.Shape, DType.F32);
        try
        {
            backend.Scale(damped, gate, ChromaF16.ResidualDamp);
        }
        catch
        {
            damped.Dispose();
            throw;
        }
        gate.Dispose();
        return damped;
    }

    /// <summary>Runs <paramref name="x"/> through <paramref name="blocks"/>, taking ownership of it: each block's input is
    /// freed once its output exists, and the current activation is freed if a block throws.</summary>
    private static Tensor RunStack(IBackend backend, Tensor x, Flux3Block[] blocks, AdaLNModulation projection, Tensor vec, FluxRope rope,
        bool f16)
    {
        try
        {
            Tensor[] rows = projection.Forward(backend, vec);
            try
            {
                if (f16) rows[2] = DampGate(backend, rows[2]);
            }
            catch
            {
                foreach (Tensor row in rows) row.Dispose();
                throw;
            }
            using Flux3Modulation modulation = Flux3Modulation.Uniform(rows[0], rows[1], rows[2]);
            foreach (Flux3Block block in blocks)
            {
                Tensor next = block.Forward(backend, x, modulation, rope);
                x.Dispose();
                x = next;
            }
            return x;
        }
        catch
        {
            x.Dispose();
            throw;
        }
    }

    /// <summary>Runs the joint stack over <paramref name="joint"/>, taking ownership of it as <see cref="RunStack"/> does:
    /// each block's input is freed once its output exists, and the current activation is freed if anything throws.</summary>
    private Tensor RunJoint(IBackend backend, Tensor joint, Tensor imageVec, Tensor zeroVec, int[] spanLengths, bool hasReferences,
        FluxRope rope, bool f16)
    {
        List<Tensor[]> projected = new List<Tensor[]>(3);
        try
        {
            projected.Add(_jointModulations[ImageMode].Forward(backend, imageVec));
            if (hasReferences) projected.Add(_jointModulations[ReferenceMode].Forward(backend, zeroVec));
            projected.Add(_jointModulations[TextMode].Forward(backend, zeroVec));
            if (f16)
            {
                foreach (Tensor[] triple in projected) triple[2] = DampGate(backend, triple[2]);
            }
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
        catch
        {
            joint.Dispose();
            throw;
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
