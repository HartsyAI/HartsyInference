using HartsyInference.Core.Backends;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Fibo 2.0 DiT block (ComfyUI <c>fibo2.Block</c>): Z-Image's RMSNorm sandwich with per-token adaLN groups and,
/// for injection blocks, a tanh-gated cross-attention to a text bundle. Each group holds one adaLN embedding: the
/// <see cref="TokenSegmentLayout"/> names the group of every token (gist and reference tokens take the clean embedding,
/// image tokens the noisy one), and a refiner uses a single group. Fused checkpoint tensors are split at load.</summary>
public sealed unsafe class Fibo2Block : IDisposable
{
    private readonly int _dim;
    private readonly int _ffnDim;
    private readonly float _eps;
    private readonly bool _hasCrossAttention;
    private readonly AttentionLayerConfig _attentionConfig;
    private readonly SwiGluFfn _ffn;
    private readonly List<Tensor> _ownedSplits = [];

    private Tensor? _adaLnWeight, _adaLnBias;
    private Tensor? _attentionNorm1, _attentionNorm2, _ffnNorm1, _ffnNorm2;
    private Tensor? _crossNorm1, _crossNorm2, _crossGate;
    private AttentionLayer? _selfAttention, _crossAttention;
    private int _disposed;

    /// <summary>Creates a block of the config's width and heads; <paramref name="hasCrossAttention"/> marks an injection block.</summary>
    public Fibo2Block(Fibo2Config config, bool hasCrossAttention)
    {
        ArgumentNullException.ThrowIfNull(config);
        _dim = config.Dim;
        _ffnDim = config.FfnDim;
        _eps = config.NormEps;
        _hasCrossAttention = hasCrossAttention;
        _attentionConfig = new AttentionLayerConfig
        {
            Heads = config.NumHeads,
            KvHeads = config.NumHeads,
            HeadDim = config.HeadDim,
            QkNorm = AttentionQkNorm.PerHead,
        };
        _ffn = new SwiGluFfn(config.Dim, config.FfnDim);
    }

    /// <summary>Loads the block under <paramref name="prefix"/> (for example <c>layers.6</c>), ComfyUI-native names.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        ArgumentNullException.ThrowIfNull(weights);
        _adaLnWeight = weights[$"{prefix}.adaLN_modulation.0.weight"];
        _adaLnBias = weights[$"{prefix}.adaLN_modulation.0.bias"];
        _attentionNorm1 = TensorCasts.LoadF32(weights, $"{prefix}.attention_norm1.weight");
        _attentionNorm2 = TensorCasts.LoadF32(weights, $"{prefix}.attention_norm2.weight");
        _ffnNorm1 = TensorCasts.LoadF32(weights, $"{prefix}.ffn_norm1.weight");
        _ffnNorm2 = TensorCasts.LoadF32(weights, $"{prefix}.ffn_norm2.weight");

        (Tensor q, Tensor k, Tensor v) = ZImageBlock.SplitQkv(weights[$"{prefix}.attention.qkv.weight"], _dim);
        _ownedSplits.Add(q);
        _ownedSplits.Add(k);
        _ownedSplits.Add(v);
        _selfAttention = new AttentionLayer(_attentionConfig, new AttentionLayerWeights
        {
            Query = q,
            Key = k,
            Value = v,
            Output = weights[$"{prefix}.attention.out.weight"],
            QueryNorm = TensorCasts.LoadF32(weights, $"{prefix}.attention.q_norm.weight"),
            KeyNorm = TensorCasts.LoadF32(weights, $"{prefix}.attention.k_norm.weight"),
            QueryNormEps = _eps,
            KeyNormEps = _eps,
        });

        Fibo2Weights.LoadSwiGlu(_ffn, weights, $"{prefix}.feed_forward.", _ffnDim, _ownedSplits);

        if (!_hasCrossAttention)
            return;
        _crossNorm1 = TensorCasts.LoadF32(weights, $"{prefix}.cross_attn_norm1.weight");
        _crossNorm2 = TensorCasts.LoadF32(weights, $"{prefix}.cross_attn_norm2.weight");
        _crossGate = TensorCasts.LoadF32(weights, $"{prefix}.cross_attn_gate");
        _crossAttention = Fibo2Weights.LoadCrossAttention(weights, $"{prefix}.cross_attention.", _attentionConfig, _dim, _eps,
            _ownedSplits);
    }

    /// <summary>Runs the block on <paramref name="x"/> <c>[1, S, dim]</c>. <paramref name="groupEmbeddings"/> holds one
    /// <c>[1, 256]</c> adaLN embedding per layout group. <paramref name="rope"/> must be precomputed for the S positions.
    /// The caller keeps <paramref name="x"/>, <paramref name="textBundle"/> and the rope; the result is new.</summary>
    public Tensor Forward(IBackend backend, Tensor x, IReadOnlyList<Tensor> groupEmbeddings, TokenSegmentLayout layout,
        ZImageRope rope, Tensor? textBundle)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(groupEmbeddings);
        ArgumentNullException.ThrowIfNull(layout);
        ArgumentNullException.ThrowIfNull(rope);
        if (x.Shape.Rank != 3 || x.Shape[0] != 1 || x.Shape[2] != _dim)
            throw new ArgumentException($"Fibo2 block input must be [1, S, {_dim}], got {x.Shape}.", nameof(x));
        if (layout.TotalLength != x.Shape[1])
            throw new ArgumentException($"Layout covers {layout.TotalLength} tokens but the input has {x.Shape[1]}.", nameof(layout));
        if (groupEmbeddings.Count != layout.GroupCount)
            throw new ArgumentException($"{groupEmbeddings.Count} adaLN embeddings for {layout.GroupCount} groups.", nameof(groupEmbeddings));
        if (_hasCrossAttention != (textBundle is not null))
            throw new ArgumentException("A text bundle is required for exactly the injection blocks.", nameof(textBundle));

        int seqLen = (int)x.Shape[1];
        TensorShape shape = new(1, seqLen, _dim);
        (SegmentModulation attentionMod, SegmentModulation ffnMod) = BuildModulation(backend, groupEmbeddings, layout);
        try
        {
            Tensor normed = new(shape, DType.F32);
            backend.RmsNorm(normed, x, _attentionNorm1!, _eps);
            Tensor modulated = attentionMod.Affine(backend, normed);
            normed.Dispose();

            AttentionCallContext selfContext = new() { Rotation = rope, ReleaseInputs = true };
            Tensor attended = _selfAttention!.Forward(backend, modulated, modulated, in selfContext);
            Tensor postAttention = new(shape, DType.F32);
            backend.RmsNorm(postAttention, attended, _attentionNorm2!, _eps);
            attended.Dispose();
            Tensor afterAttention = attentionMod.GatedResidual(backend, x, postAttention);
            postAttention.Dispose();

            Tensor current = afterAttention;
            if (textBundle is not null)
            {
                current = CrossAttend(backend, afterAttention, textBundle, shape);
                afterAttention.Dispose();
            }

            Tensor ffnNormed = new(shape, DType.F32);
            backend.RmsNorm(ffnNormed, current, _ffnNorm1!, _eps);
            Tensor ffnInput = ffnMod.Affine(backend, ffnNormed);
            ffnNormed.Dispose();
            Tensor ffnOut = _ffn.Forward(backend, ffnInput, 1, seqLen);
            ffnInput.Dispose();
            Tensor ffnPost = new(shape, DType.F32);
            backend.RmsNorm(ffnPost, ffnOut, _ffnNorm2!, _eps);
            ffnOut.Dispose();

            Tensor result = ffnMod.GatedResidual(backend, current, ffnPost);
            ffnPost.Dispose();
            current.Dispose();
            return result;
        }
        finally
        {
            attentionMod.Dispose();
            ffnMod.Dispose();
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        foreach (Tensor split in _ownedSplits)
            split.Dispose();
        _ownedSplits.Clear();
    }

    /// <summary>Per-group adaLN rows for the attention and FFN sub-blocks. The 1 + scale of ModulationSplit4 is undone
    /// because <see cref="SegmentModulation"/> applies it; its gates arrive already tanh'd.</summary>
    private (SegmentModulation Attention, SegmentModulation Ffn) BuildModulation(IBackend backend,
        IReadOnlyList<Tensor> embeddings, TokenSegmentLayout layout)
    {
        List<(Tensor Shift, Tensor Scale, Tensor Gate)> attentionRows = new(embeddings.Count);
        List<(Tensor Shift, Tensor Scale, Tensor Gate)> ffnRows = new(embeddings.Count);
        try
        {
            TensorShape rowShape = new(1, _dim);
            foreach (Tensor embedding in embeddings)
            {
                Tensor projected = new(new TensorShape(1, 4 * _dim), DType.F32);
                backend.Linear(projected, embedding, _adaLnWeight!, _adaLnBias);
                Tensor scaleMsa = new(rowShape, DType.F32);
                Tensor gateMsa = new(rowShape, DType.F32);
                Tensor scaleMlp = new(rowShape, DType.F32);
                Tensor gateMlp = new(rowShape, DType.F32);
                backend.ModulationSplit4(scaleMsa, gateMsa, scaleMlp, gateMlp, projected);
                projected.Dispose();
                attentionRows.Add((Zeros(rowShape), Unfolded(backend, scaleMsa, rowShape), gateMsa));
                ffnRows.Add((Zeros(rowShape), Unfolded(backend, scaleMlp, rowShape), gateMlp));
            }
            return (SegmentModulation.Build(backend, attentionRows, layout), SegmentModulation.Build(backend, ffnRows, layout));
        }
        finally
        {
            ReleaseRows(attentionRows);
            ReleaseRows(ffnRows);
        }
    }

    private Tensor CrossAttend(IBackend backend, Tensor x, Tensor bundle, TensorShape shape)
    {
        Tensor queryInput = new(shape, DType.F32);
        backend.RmsNorm(queryInput, x, _crossNorm1!, _eps);
        AttentionCallContext context = new() { ReleaseInputs = false };
        Tensor attended = _crossAttention!.Forward(backend, queryInput, bundle, in context);
        queryInput.Dispose();

        Tensor postNorm = new(shape, DType.F32);
        backend.RmsNorm(postNorm, attended, _crossNorm2!, _eps);
        attended.Dispose();

        Tensor gate = new(new TensorShape(1, _dim), DType.F32);
        float gateValue = MathF.Tanh(((float*)_crossGate!.DataPointer)[0]);
        float* gatePtr = (float*)gate.DataPointer;
        for (int d = 0; d < _dim; d++)
            gatePtr[d] = gateValue;
        Tensor output = new(shape, DType.F32);
        backend.GatedResidualLastDim(output, x, postNorm, gate);
        postNorm.Dispose();
        gate.Dispose();
        return output;
    }

    private static Tensor Unfolded(IBackend backend, Tensor scalePlusOne, TensorShape shape)
    {
        Tensor scale = new(shape, DType.F32);
        backend.AddScalar(scale, scalePlusOne, -1f);
        scalePlusOne.Dispose();
        return scale;
    }

    private static Tensor Zeros(TensorShape shape)
    {
        Tensor zeros = new(shape, DType.F32);
        new Span<float>((void*)zeros.DataPointer, (int)zeros.ElementCount).Clear();
        return zeros;
    }

    private static void ReleaseRows(List<(Tensor Shift, Tensor Scale, Tensor Gate)> rows)
    {
        foreach ((Tensor shift, Tensor scale, Tensor gate) in rows)
        {
            shift.Dispose();
            scale.Dispose();
            gate.Dispose();
        }
    }
}
