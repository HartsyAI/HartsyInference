using HartsyInference.Core.Backends;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Segmentation.Sam3;

/// <summary>One pre-norm residual block of the SAM 3 text tower (CLIP-style, batch-first): <c>x + attn(ln_1(x))</c>, then
/// <c>x + mlp(ln_2(x))</c> with an exact-GELU MLP. Attention is the shared <see cref="AttentionLayer"/> over the fused
/// <c>in_proj</c> weight, with the causal mask as an additive bias.</summary>
internal sealed class Sam3TextTransformerLayer
{
    private readonly AttentionLayer _attention;
    private readonly Tensor _ln1Weight;
    private readonly Tensor _ln1Bias;
    private readonly Tensor _ln2Weight;
    private readonly Tensor _ln2Bias;
    private readonly Tensor _fc1Weight;
    private readonly Tensor _fc1Bias;
    private readonly Tensor _fc2Weight;
    private readonly Tensor _fc2Bias;
    private readonly int _width;
    private readonly int _mlpWidth;
    private readonly float _eps;

    /// <summary>Loads one block from <paramref name="weights"/> under <paramref name="prefix"/> (for example
    /// <c>encoder.transformer.resblocks.3.</c>). The dictionary keeps ownership of the weights.</summary>
    /// <exception cref="InvalidDataException">A required weight is missing or has the wrong shape.</exception>
    public Sam3TextTransformerLayer(Sam3Config config, IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(weights);
        ArgumentException.ThrowIfNullOrEmpty(prefix);

        int width = config.TextWidth;
        int mlp = config.TextMlpHidden;
        _width = width;
        _mlpWidth = mlp;
        _eps = config.LayerNormEps;

        Tensor inProjWeight = Sam3WeightAccess.Require(weights, prefix + "attn.in_proj_weight", 3 * width, width);
        Tensor inProjBias = Sam3WeightAccess.Require(weights, prefix + "attn.in_proj_bias", 3 * width);
        Tensor outProjWeight = Sam3WeightAccess.Require(weights, prefix + "attn.out_proj.weight", width, width);
        Tensor outProjBias = Sam3WeightAccess.Require(weights, prefix + "attn.out_proj.bias", width);
        _ln1Weight = Sam3WeightAccess.Require(weights, prefix + "ln_1.weight", width);
        _ln1Bias = Sam3WeightAccess.Require(weights, prefix + "ln_1.bias", width);
        _ln2Weight = Sam3WeightAccess.Require(weights, prefix + "ln_2.weight", width);
        _ln2Bias = Sam3WeightAccess.Require(weights, prefix + "ln_2.bias", width);
        _fc1Weight = Sam3WeightAccess.Require(weights, prefix + "mlp.c_fc.weight", mlp, width);
        _fc1Bias = Sam3WeightAccess.Require(weights, prefix + "mlp.c_fc.bias", mlp);
        _fc2Weight = Sam3WeightAccess.Require(weights, prefix + "mlp.c_proj.weight", width, mlp);
        _fc2Bias = Sam3WeightAccess.Require(weights, prefix + "mlp.c_proj.bias", width);

        _attention = new AttentionLayer(
            new AttentionLayerConfig
            {
                Heads = config.TextHeads,
                KvHeads = config.TextHeads,
                HeadDim = width / config.TextHeads,
                Projection = AttentionProjection.FusedSlice,
                ComputeDType = DType.F32,
            },
            new AttentionLayerWeights
            {
                Qkv = inProjWeight,
                QkvBias = inProjBias,
                Output = outProjWeight,
                OutputBias = outProjBias,
            });
    }

    /// <summary>Runs the block on <paramref name="hidden"/> <c>[1, S, width]</c> with the additive <paramref name="causalMask"/>
    /// <c>[S, S]</c>. Borrows <paramref name="hidden"/>; the caller owns the returned tensor and the input.</summary>
    public Tensor Forward(IBackend backend, Tensor hidden, Tensor causalMask)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(hidden);
        ArgumentNullException.ThrowIfNull(causalMask);
        TensorShape shape = hidden.Shape;
        if (shape.Rank != 3 || shape[2] != _width)
        {
            throw new ArgumentException($"Text block takes [B, S, {_width}] input; got {shape}.", nameof(hidden));
        }

        int batch = (int)shape[0];
        int seq = (int)shape[1];
        TensorShape mlpShape = new TensorShape(batch, seq, _mlpWidth);

        Tensor normed1 = new Tensor(shape, DType.F32);
        backend.LayerNorm(normed1, hidden, _ln1Weight, _ln1Bias, _eps);
        AttentionCallContext context = new AttentionCallContext { Bias = causalMask };
        Tensor attnOut = _attention.Forward(backend, normed1, normed1, in context);
        normed1.Dispose();

        Tensor residual = new Tensor(shape, DType.F32);
        backend.Add(residual, hidden, attnOut);
        attnOut.Dispose();

        Tensor normed2 = new Tensor(shape, DType.F32);
        backend.LayerNorm(normed2, residual, _ln2Weight, _ln2Bias, _eps);

        Tensor fc1Out = new Tensor(mlpShape, DType.F32);
        backend.Linear(fc1Out, normed2, _fc1Weight, _fc1Bias);
        normed2.Dispose();

        Tensor activated = new Tensor(mlpShape, DType.F32);
        backend.GeluErf(activated, fc1Out);
        fc1Out.Dispose();

        Tensor fc2Out = new Tensor(shape, DType.F32);
        backend.Linear(fc2Out, activated, _fc2Weight, _fc2Bias);
        activated.Dispose();

        Tensor output = new Tensor(shape, DType.F32);
        backend.Add(output, residual, fc2Out);
        residual.Dispose();
        fc2Out.Dispose();
        return output;
    }
}
